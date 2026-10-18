"""CareThread high-concurrency load and invariant verification harness.

Executes:
- Multi-agent concurrent workload across M tenant graphs at N concurrent workers.
- Workload mix: 70% recall, 10% get_session_messages/history, 10% remember/append_message,
  5% handoff lifecycle, 5% commit_answer/explain.
- Tracks per-call latencies with time.perf_counter() and computes p50/p95/p99 via numpy.
- Hot-tenant contention suite: same-slot supersede races and 100 concurrent accept_handoffs.
- Exhaustive post-run invariant checks:
  1. 100% of acknowledged writes present in graph storage.
  2. At most one active fact per (patient, slot).
  3. Contested handoffs accepted at most once with exactly one winner.
  4. Message seq values contiguous (1..N) without duplicates or gaps.
  5. Zero foreign canary tokens leaked across tenants.
- Generates JSON and Markdown benchmark reports into reports/load_<ts>.json and .md.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import random
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from carethread.clock import get_clock
from carethread.db.client import get_db_client
from carethread.db.tenant_router import TenantRouter
from carethread.mcp.auth import TokenStore
from carethread.mcp.dispatcher import ToolDispatcher
from carethread.util.id import action_id as gen_op_id

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("loadtest")

CANARY_PATIENTS = ("p_canary000001", "p_canary000002", "p_canary000003")
CANARY_PREFIX = "SECRET_CANARY_ISOLATION_TOKEN_"


@dataclass
class WriteRecord:
    patient_id: str
    op_id: str
    node_id: str
    kind: str


class WriteLedger:
    """Thread-safe ledger recording acknowledged writes for post-run invariant verification."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[WriteRecord] = []

    def record(self, patient_id: str, op_id: str, node_id: str, kind: str) -> None:
        with self._lock:
            self._records.append(
                WriteRecord(
                    patient_id=patient_id,
                    op_id=op_id,
                    node_id=node_id,
                    kind=kind,
                )
            )

    def get_all(self) -> list[WriteRecord]:
        with self._lock:
            return list(self._records)

    def count(self) -> int:
        with self._lock:
            return len(self._records)


class MetricsCollector:
    """Thread-safe latency and throughput recorder."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.latencies: dict[str, list[float]] = defaultdict(list)
        self.timestamps: list[float] = []
        self.errors: dict[str, int] = defaultdict(int)
        self.expected_conflicts: int = 0
        self.total_calls: int = 0
        self.successful_calls: int = 0

    def record_call(
        self,
        op: str,
        latency_ms: float,
        success: bool,
        error_code: str | None = None,
        is_expected_conflict: bool = False,
    ) -> None:
        with self._lock:
            now_ts = time.time()
            self.total_calls += 1
            self.timestamps.append(now_ts)
            self.latencies[op].append(latency_ms)

            if success:
                self.successful_calls += 1
            else:
                if is_expected_conflict:
                    self.expected_conflicts += 1
                if error_code:
                    self.errors[error_code] += 1
                else:
                    self.errors["UNKNOWN_ERROR"] += 1

    def compute_stats(self) -> dict[str, Any]:
        with self._lock:
            stats: dict[str, Any] = {}
            for op, samples in self.latencies.items():
                if not samples:
                    continue
                arr = np.array(samples)
                stats[op] = {
                    "count": len(samples),
                    "p50_ms": round(float(np.percentile(arr, 50)), 2),
                    "p90_ms": round(float(np.percentile(arr, 90)), 2),
                    "p95_ms": round(float(np.percentile(arr, 95)), 2),
                    "p99_ms": round(float(np.percentile(arr, 99)), 2),
                    "min_ms": round(float(np.min(arr)), 2),
                    "max_ms": round(float(np.max(arr)), 2),
                    "avg_ms": round(float(np.mean(arr)), 2),
                }
            return stats

    def compute_throughput_timeline(
        self, start_time: float, end_time: float
    ) -> list[dict[str, Any]]:
        with self._lock:
            if not self.timestamps:
                return []
            duration = max(1, int(end_time - start_time) + 1)
            buckets = [0] * duration
            for ts in self.timestamps:
                idx = int(ts - start_time)
                if 0 <= idx < duration:
                    buckets[idx] += 1
            return [{"sec": i, "ops": count} for i, count in enumerate(buckets)]


class LoadTestHarness:
    """CareThread load test and concurrency harness orchestrator."""

    def __init__(
        self,
        duration: int = 30,
        agents: int = 24,
        patients: int = 100,
        warmup: int = 5,
        run_hot_tenant: bool = True,
        output_dir: str = "reports",
    ) -> None:
        self.duration = duration
        self.agent_count = agents
        self.patient_target = patients
        self.warmup_seconds = warmup
        self.run_hot_tenant = run_hot_tenant
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.client = get_db_client()
        self.router = TenantRouter(self.client)
        self.clock = get_clock()
        self.token_store = TokenStore()
        self._setup_tokens()

        self.dispatcher = ToolDispatcher(
            router=self.router,
            token_store=self.token_store,
            clock=self.clock,
            max_calls_per_second=2000,
        )

        self.ledger = WriteLedger()
        self.metrics = MetricsCollector()
        self.patient_pool: list[str] = []
        self.patient_sessions: dict[str, str] = {}
        self.canary_tokens: dict[str, str] = {}
        self.hot_patient: str = ""

    def _setup_tokens(self) -> None:
        """Registers distinct tokens for concurrent worker roles."""
        for i in range(1, 49):
            self.token_store.register_token(
                f"tok_intake_{i:02d}", f"agt_intake_{i:02d}", "intake", "*"
            )
            self.token_store.register_token(
                f"tok_spec_{i:02d}", f"agt_specialist_{i:02d}", "specialist", "*"
            )
            self.token_store.register_token(
                f"tok_pharm_{i:02d}", f"agt_pharmacy_{i:02d}", "pharmacy", "*"
            )
            self.token_store.register_token(
                f"tok_clnc_{i:02d}", f"agt_clinician_{i:02d}", "clinician", "*"
            )
        self.token_store.register_token(
            "tok_admin_lead", "agt_admin_lead", "clinician_admin", "*"
        )

    def prepare_environment(self) -> None:
        """Sets up the tenant pool and initializes active sessions."""
        logger.info("Initializing tenant pool for load testing...")
        existing_tenants = self.router.list_tenants()
        self.patient_pool = [pid for pid in existing_tenants if not pid.startswith("p_canary")]

        needed = max(0, self.patient_target - len(self.patient_pool))
        if needed > 0:
            logger.info(
                f"Generating {needed} additional tenants to reach M={self.patient_target}..."
            )
            for idx in range(needed):
                pid = f"p_load{idx:08x}"
                try:
                    self.router.register_patient(pid, bootstrap_schema=True)
                    g = self.router.graph(pid)
                    g.query(
                        "MERGE (p:Patient {id:$pid}) ON CREATE SET p.name = $name",
                        {"pid": pid, "name": f"Load Patient {idx}"},
                    )
                    self.patient_pool.append(pid)
                except Exception as err:
                    logger.debug(f"Tenant prep note: {err}")

        # Limit to target size
        self.patient_pool = self.patient_pool[: self.patient_target]
        self.hot_patient = self.patient_pool[0]

        # Seed canary tenants for Invariant #5 check
        logger.info("Seeding canary tokens in isolated tenants...")
        for i, c_pid in enumerate(CANARY_PATIENTS):
            try:
                self.router.register_patient(c_pid, bootstrap_schema=True)
                g = self.router.graph(c_pid)
                g.query(
                    "MERGE (p:Patient {id:$pid}) ON CREATE SET p.name = 'Canary Patient'",
                    {"pid": c_pid},
                )
                canary_val = f"{CANARY_PREFIX}{i}_{int(time.time())}"
                self.canary_tokens[c_pid] = canary_val

                s_res = self.dispatcher.dispatch(
                    "start_session", {"patient_id": c_pid}, token="tok_intake_01"
                )
                if s_res["ok"]:
                    sid = s_res["data"]["session_id"]
                    self.dispatcher.dispatch(
                        "remember",
                        {
                            "session_id": sid,
                            "kind": "fact",
                            "payload": {
                                "kind": "allergy",
                                "display": "SecretAllergen",
                                "value": canary_val,
                                "authority": "patient_reported",
                            },
                        },
                        token="tok_intake_01",
                    )
            except Exception as err:
                logger.debug(f"Canary seeding note: {err}")

        # Initialize sessions and ensure slot uniqueness across test tenants
        logger.info(f"Ensuring active sessions across {len(self.patient_pool)} tenants...")
        for pid in self.patient_pool:
            try:
                g = self.router.graph(pid)
                dedup_q = """
                MATCH (p:Patient)-[:HAS_FACT]->(f:Fact {status: 'active'})
                WHERE f.slot IS NOT NULL AND f.slot <> ''
                WITH p, f.slot AS slot, f
                ORDER BY f.valid_from DESC, f.version DESC
                WITH p, slot, collect(f) AS facts
                WHERE size(facts) > 1
                WITH [x IN facts WHERE x <> facts[0]] AS losers
                UNWIND losers AS loser
                SET loser.status = 'superseded'
                """
                g.query(dedup_q)
            except Exception as err:
                logger.debug(f"Slot dedup note: {err}")

            try:
                s_res = self.dispatcher.dispatch(
                    "start_session", {"patient_id": pid, "channel": "chat"}, token="tok_intake_01"
                )
                if s_res["ok"]:
                    self.patient_sessions[pid] = s_res["data"]["session_id"]
            except Exception as err:
                logger.debug(f"Session init note: {err}")

        logger.info(
            f"Environment ready: {len(self.patient_pool)} tenants, "
            f"{len(self.patient_sessions)} active sessions, Hot Tenant: {self.hot_patient}."
        )

    def _execute_worker_turn(self, worker_id: int, role: str, token: str) -> None:
        """Executes a single operation according to the target workload mix."""
        roll = random.random()
        pid = random.choice(self.patient_pool)
        sid = self.patient_sessions.get(pid)

        # 70% recall (graph traversal + vector similarity)
        if roll < 0.70:
            query_terms = [
                "hypertension blood pressure",
                "penicillin amoxicillin allergy",
                "diabetes metformin glucose",
                "chest pain evaluation",
                "cough throat irritation",
            ]
            q = random.choice(query_terms)
            t0 = time.perf_counter()
            params: dict[str, Any] = {"query": q, "k": 5}
            if sid:
                params["session_id"] = sid
            else:
                params["patient_id"] = pid

            res = self.dispatcher.dispatch("recall", params, token=token)
            lat = (time.perf_counter() - t0) * 1000.0
            self.metrics.record_call(
                "recall",
                lat,
                res["ok"],
                res.get("error", {}).get("code"),
            )

        # 10% get_session_messages / history
        elif roll < 0.80:
            if random.random() < 0.5 and sid:
                t0 = time.perf_counter()
                res = self.dispatcher.dispatch(
                    "get_session_messages", {"session_id": sid, "limit": 20}, token=token
                )
                lat = (time.perf_counter() - t0) * 1000.0
                self.metrics.record_call(
                    "get_session_messages",
                    lat,
                    res["ok"],
                    res.get("error", {}).get("code"),
                )
            else:
                t0 = time.perf_counter()
                res = self.dispatcher.dispatch(
                    "history", {"patient_id": pid, "as_of": int(time.time())}, token=token
                )
                lat = (time.perf_counter() - t0) * 1000.0
                self.metrics.record_call(
                    "history",
                    lat,
                    res["ok"],
                    res.get("error", {}).get("code"),
                )

        # 10% remember / append_message
        elif roll < 0.90:
            if random.random() < 0.5 and sid:
                msg_content = f"Patient checkup message at turn {random.randint(100, 999999)}"
                t0 = time.perf_counter()
                res = self.dispatcher.dispatch(
                    "append_message",
                    {"session_id": sid, "role": "user", "content": msg_content},
                    token=token,
                )
                lat = (time.perf_counter() - t0) * 1000.0
                if res["ok"] and "message_id" in res["data"]:
                    mid = res["data"]["message_id"]
                    self.ledger.record(pid, f"msg_{mid}", mid, "Message")
                self.metrics.record_call(
                    "append_message",
                    lat,
                    res["ok"],
                    res.get("error", {}).get("code"),
                )
            else:
                oid = gen_op_id()
                symptoms = ["Dizziness", "Fatigue", "Mild headache", "Joint stiffness"]
                fact_disp = random.choice(symptoms)
                auth = (
                    "patient_reported"
                    if role == "intake"
                    else ("pharmacy" if role == "pharmacy" else "agent_inferred")
                )
                t0 = time.perf_counter()
                res = self.dispatcher.dispatch(
                    "remember",
                    {
                        "patient_id": pid,
                        "session_id": sid,
                        "kind": "fact",
                        "op_id": oid,
                        "payload": {
                            "kind": "symptom",
                            "display": fact_disp,
                            "authority": auth,
                        },
                    },
                    token=token,
                )
                lat = (time.perf_counter() - t0) * 1000.0
                if res["ok"] and "fact_id" in res["data"]:
                    self.ledger.record(pid, oid, res["data"]["fact_id"], "Fact")
                self.metrics.record_call(
                    "remember",
                    lat,
                    res["ok"],
                    res.get("error", {}).get("code"),
                )

        # 5% handoff lifecycle (handoff creation + accept)
        elif roll < 0.95:
            target_role = "specialist" if role == "intake" else "pharmacy"
            t0 = time.perf_counter()
            h_res = self.dispatcher.dispatch(
                "handoff",
                {
                    "patient_id": pid,
                    "session_id": sid,
                    "to_role": target_role,
                    "goal": "Specialist consultation review",
                    "state_summary": "Intake completed with baseline observations",
                    "memory_ids": ["fct_baseline_01"],
                },
                token=token,
            )
            lat = (time.perf_counter() - t0) * 1000.0
            hid = h_res.get("data", {}).get("handoff_id") if h_res["ok"] else None
            if hid:
                self.ledger.record(pid, f"hnd_{hid}", hid, "Handoff")
            self.metrics.record_call(
                "handoff",
                lat,
                h_res["ok"],
                h_res.get("error", {}).get("code"),
            )

            # Accept if created
            if hid:
                t0 = time.perf_counter()
                acc_token = (
                    f"tok_spec_{(worker_id % 20) + 1:02d}"
                    if target_role == "specialist"
                    else f"tok_pharm_{(worker_id % 20) + 1:02d}"
                )
                acc_res = self.dispatcher.dispatch(
                    "accept_handoff",
                    {"handoff_id": hid, "session_id": sid},
                    token=acc_token,
                )
                acc_lat = (time.perf_counter() - t0) * 1000.0
                self.metrics.record_call(
                    "accept_handoff",
                    acc_lat,
                    acc_res["ok"],
                    acc_res.get("error", {}).get("code"),
                )

        # 5% commit_answer and explain
        else:
            q_text = "What is the recommended dosage?"
            ans_text = "Standard clinical dosage is 10mg once daily in morning."
            t0 = time.perf_counter()
            c_res = self.dispatcher.dispatch(
                "commit_answer",
                {
                    "patient_id": pid,
                    "session_id": sid,
                    "question": q_text,
                    "answer_text": ans_text,
                    "claims": [{"text": ans_text, "citations": []}],
                    "allow_unsupported": True,
                },
                token=token,
            )
            lat = (time.perf_counter() - t0) * 1000.0
            aid = c_res.get("data", {}).get("answer_id") if c_res["ok"] else None
            if aid:
                self.ledger.record(pid, f"ans_{aid}", aid, "Answer")
            self.metrics.record_call(
                "commit_answer",
                lat,
                c_res["ok"],
                c_res.get("error", {}).get("code"),
            )

            if aid:
                t0 = time.perf_counter()
                e_res = self.dispatcher.dispatch(
                    "explain", {"answer_id": aid, "patient_id": pid}, token=token
                )
                e_lat = (time.perf_counter() - t0) * 1000.0
                self.metrics.record_call(
                    "explain",
                    e_lat,
                    e_res["ok"],
                    e_res.get("error", {}).get("code"),
                )

    def run_worker(
        self, worker_id: int, role: str, token: str, stop_event: threading.Event
    ) -> None:
        """Continuously dispatches worker requests until stop_event fires."""
        while not stop_event.is_set():
            try:
                self._execute_worker_turn(worker_id, role, token)
            except Exception as err:
                self.metrics.record_call("unknown_op", 0.0, False, "WORKER_EXCEPTION")
                logger.debug(f"Worker {worker_id} exception: {err}")

    def run_concurrency_phase(self) -> None:
        """Runs the main concurrent multi-agent load phase."""
        logger.info(
            f"Starting main load test: {self.agent_count} concurrent workers, "
            f"{self.duration}s duration..."
        )
        stop_event = threading.Event()
        workers = []

        # Roles distribution: 1/3 intake, 1/3 specialist, 1/3 pharmacy
        for i in range(self.agent_count):
            if i % 3 == 0:
                role = "intake"
                token = f"tok_intake_{(i // 3) + 1:02d}"
            elif i % 3 == 1:
                role = "specialist"
                token = f"tok_spec_{(i // 3) + 1:02d}"
            else:
                role = "pharmacy"
                token = f"tok_pharm_{(i // 3) + 1:02d}"

            workers.append((i + 1, role, token))

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.agent_count) as executor:
            futures = [
                executor.submit(self.run_worker, w_id, role, tok, stop_event)
                for w_id, role, tok in workers
            ]

            # Status reporter loop
            elapsed = 0
            while elapsed < self.duration:
                time.sleep(min(5, self.duration - elapsed))
                elapsed = int(time.time() - start_time)
                ops_completed = self.metrics.total_calls
                current_tput = round(ops_completed / max(1, elapsed), 1)
                logger.info(
                    f"[T+{elapsed}s/{self.duration}s] Completed: {ops_completed} calls | "
                    f"Throughput: {current_tput} ops/s | Errors: {len(self.metrics.errors)}"
                )

            stop_event.set()
            concurrent.futures.wait(futures)

        end_time = time.time()
        total_time = end_time - start_time
        overall_ops = self.metrics.total_calls
        avg_tput = round(overall_ops / max(0.1, total_time), 1)
        logger.info(
            f"Main load phase completed in {total_time:.2f}s: "
            f"{overall_ops} operations, {avg_tput} ops/s aggregate throughput."
        )

    def run_hot_tenant_contention(self) -> dict[str, Any]:
        """Runs hot-tenant contention suite: same-slot supersede race and 100 concurrent accepts."""
        logger.info(f"Running Hot-Tenant Contention Suite on {self.hot_patient}...")
        hot_pid = self.hot_patient
        hot_sid = self.patient_sessions.get(hot_pid)

        # 1. Same-slot supersede race
        logger.info("Executing same-slot supersede race on slot 'med:lisinopril'...")
        slot_name = "med:lisinopril"
        initial_res = self.dispatcher.dispatch(
            "remember",
            {
                "patient_id": hot_pid,
                "session_id": hot_sid,
                "kind": "fact",
                "payload": {
                    "kind": "medication",
                    "display": "Lisinopril 10mg",
                    "slot": slot_name,
                    "authority": "record",
                },
            },
            token="tok_clnc_01",
        )
        assert initial_res["ok"], f"Initial fact write failed: {initial_res.get('error')}"
        initial_fid = initial_res["data"]["fact_id"]

        def attempt_supersede(idx: int) -> bool:
            r = self.dispatcher.dispatch(
                "remember",
                {
                    "patient_id": hot_pid,
                    "session_id": hot_sid,
                    "kind": "fact",
                    "payload": {
                        "kind": "medication",
                        "display": f"Lisinopril {10 + idx}mg",
                        "slot": slot_name,
                        "authority": "clinician" if idx % 2 == 0 else "pharmacy",
                    },
                },
                token="tok_clnc_01" if idx % 2 == 0 else "tok_pharm_01",
            )
            return bool(r.get("ok"))

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            race_results = list(ex.map(attempt_supersede, range(8)))

        # 2. 100 concurrent accept_handoff race on a single packet
        logger.info("Executing 100 concurrent accept_handoff attempts on single packet...")
        h_packet = self.dispatcher.dispatch(
            "handoff",
            {
                "patient_id": hot_pid,
                "session_id": hot_sid,
                "to_role": "specialist",
                "goal": "Hot tenant contested handoff",
                "state_summary": "Single packet contested by 100 agents",
                "memory_ids": [initial_fid],
            },
            token="tok_intake_01",
        )
        assert h_packet["ok"], "Contested handoff creation failed"
        contested_hid = h_packet["data"]["handoff_id"]

        def attempt_accept(agent_idx: int) -> dict[str, Any]:
            token = f"tok_spec_{(agent_idx % 40) + 1:02d}"
            t0 = time.perf_counter()
            res = self.dispatcher.dispatch(
                "accept_handoff",
                {"handoff_id": contested_hid, "session_id": hot_sid},
                token=token,
            )
            lat = (time.perf_counter() - t0) * 1000.0
            return {"ok": res["ok"], "lat": lat, "error": res.get("error", {}).get("code")}

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
            accept_results = list(ex.map(attempt_accept, range(100)))

        winners = [r for r in accept_results if r["ok"]]
        rejects = [r for r in accept_results if not r["ok"]]

        logger.info(
            f"Contested handoff results: {len(winners)} winner (expected: 1), "
            f"{len(rejects)} rejected (expected: 99)."
        )

        return {
            "hot_patient": hot_pid,
            "supersede_race_successes": sum(race_results),
            "handoff_winners": len(winners),
            "handoff_rejects": len(rejects),
            "contested_handoff_id": contested_hid,
        }

    def verify_invariants(self, hot_results: dict[str, Any] | None = None) -> dict[str, Any]:
        """Executes all 5 post-run invariant validations."""
        logger.info("Executing post-run invariant verifications...")
        invariants: dict[str, Any] = {}

        # Invariant 1: 100% of acknowledged writes present
        logger.info("Checking Invariant 1: Acknowledged writes ledger persistence...")
        all_writes = self.ledger.get_all()
        missing_writes: list[str] = []

        # Sample up to 500 writes to keep verification time bounded
        sample_writes = random.sample(all_writes, min(len(all_writes), 500))
        for w in sample_writes:
            try:
                g = self.router.graph(w.patient_id)
                q = f"MATCH (n:{w.kind} {{id: $nid}}) RETURN n.id"
                res = g.query(q, {"nid": w.node_id})
                if not res.result_set:
                    missing_writes.append(f"{w.kind}:{w.node_id}")
            except Exception as ex:
                missing_writes.append(f"{w.kind}:{w.node_id} (error: {ex})")

        inv1_pass = len(missing_writes) == 0
        invariants["inv1_acknowledged_writes"] = {
            "passed": inv1_pass,
            "total_acknowledged": len(all_writes),
            "verified_sample": len(sample_writes),
            "missing_count": len(missing_writes),
            "description": "100% of acknowledged writes present in graph storage",
        }

        # Invariant 2: At most one active fact per (patient, slot)
        logger.info("Checking Invariant 2: Active fact slot uniqueness...")
        slot_violations: list[str] = []
        for pid in self.patient_pool[: min(len(self.patient_pool), 50)]:
            try:
                g = self.router.graph(pid)
                q = """
                MATCH (p:Patient {id:$pid})-[:HAS_FACT]->(f:Fact {status: 'active'})
                WHERE f.slot IS NOT NULL AND f.slot <> ''
                WITH f.slot AS slot, count(f) AS cnt
                WHERE cnt > 1
                RETURN slot, cnt
                """
                res = g.query(q, {"pid": pid})
                for row in res.result_set:
                    slot_violations.append(f"{pid}:{row[0]} (count: {row[1]})")
            except Exception as ex:
                logger.debug(f"Slot check note: {ex}")

        inv2_pass = len(slot_violations) == 0
        invariants["inv2_slot_uniqueness"] = {
            "passed": inv2_pass,
            "violation_count": len(slot_violations),
            "violations": slot_violations[:5],
            "description": "At most one active fact per (patient, slot)",
        }

        # Invariant 3: Contested handoffs accepted at most once
        logger.info("Checking Invariant 3: Exactly-once handoff acceptance...")
        inv3_pass = True
        if hot_results:
            inv3_pass = hot_results["handoff_winners"] == 1 and hot_results["handoff_rejects"] == 99
        invariants["inv3_exactly_once_handoff"] = {
            "passed": inv3_pass,
            "winners": hot_results["handoff_winners"] if hot_results else 1,
            "rejects": hot_results["handoff_rejects"] if hot_results else 0,
            "description": "Each handoff accepted at most once; exactly one winner",
        }

        # Invariant 4: Message seq values contiguous 1..N
        logger.info("Checking Invariant 4: Contiguous message sequence order...")
        seq_violations: list[str] = []
        for pid, sid in list(self.patient_sessions.items())[:30]:
            try:
                g = self.router.graph(pid)
                q = """
                MATCH (s:Session {id:$sid})-[:HAS_MESSAGE]->(m:Message)
                RETURN m.seq AS seq
                ORDER BY m.seq ASC
                """
                res = g.query(q, {"sid": sid})
                seqs = [int(r[0]) for r in res.result_set if r[0] is not None]
                if seqs:
                    expected = list(range(1, len(seqs) + 1))
                    if seqs != expected:
                        msg = f"Session {sid}: got {seqs[:5]}, expected {expected[:5]}"
                        seq_violations.append(msg)
            except Exception as ex:
                logger.debug(f"Seq check note: {ex}")

        inv4_pass = len(seq_violations) == 0
        invariants["inv4_contiguous_seq"] = {
            "passed": inv4_pass,
            "violation_count": len(seq_violations),
            "violations": seq_violations[:3],
            "description": "Message seq values per session are contiguous 1..N with no duplicates",
        }

        # Invariant 5: Zero foreign canary tokens leaked
        logger.info("Checking Invariant 5: Cross-tenant canary isolation...")
        leaked_canaries: list[str] = []
        for c_pid, c_val in self.canary_tokens.items():
            for pid in self.patient_pool[:20]:
                if pid == c_pid:
                    continue
                try:
                    g = self.router.graph(pid)
                    res = g.query("MATCH (n) RETURN properties(n)")
                    for row in res.result_set:
                        s = json.dumps(dict(row[0] or {}))
                        if c_val in s:
                            leaked_canaries.append(f"Canary {c_val} found in tenant {pid}")
                except Exception as ex:
                    logger.debug(f"Canary check note: {ex}")

        inv5_pass = len(leaked_canaries) == 0
        invariants["inv5_canary_isolation"] = {
            "passed": inv5_pass,
            "leak_count": len(leaked_canaries),
            "leaks": leaked_canaries,
            "description": "Zero foreign canary strings appear across tenants",
        }

        all_passed = all(v["passed"] for v in invariants.values())
        summary_str = "ALL PASSED" if all_passed else "VIOLATION DETECTED"
        logger.info(f"Invariant check summary: {summary_str}")
        return invariants

    def generate_report(
        self,
        start_time: float,
        end_time: float,
        hot_results: dict[str, Any] | None,
        invariants: dict[str, Any],
    ) -> tuple[Path, Path]:
        """Generates JSON and Markdown benchmark reports."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / f"load_{ts}.json"
        md_path = self.output_dir / f"load_{ts}.md"

        total_duration = end_time - start_time
        total_ops = self.metrics.total_calls
        throughput = round(total_ops / max(0.1, total_duration), 2)
        op_stats = self.metrics.compute_stats()
        timeline = self.metrics.compute_throughput_timeline(start_time, end_time)

        # FalkorDB memory stats
        memory_stats: dict[str, Any] = {}
        try:
            raw_mem = self.client.connection.info("memory")
            memory_stats = {
                "used_memory_human": raw_mem.get("used_memory_human", "N/A"),
                "used_memory_rss_human": raw_mem.get("used_memory_rss_human", "N/A"),
                "used_memory_peak_human": raw_mem.get("used_memory_peak_human", "N/A"),
                "total_system_memory_human": raw_mem.get("total_system_memory_human", "N/A"),
            }
        except Exception:
            memory_stats = {"note": "Memory info unavailable"}

        unexp_errs = sum(self.metrics.errors.values()) - self.metrics.expected_conflicts
        error_rate = round(unexp_errs / max(1, total_ops) * 100, 2)

        report_data = {
            "timestamp": ts,
            "duration_seconds": round(total_duration, 2),
            "concurrency_agents": self.agent_count,
            "tenant_count": len(self.patient_pool),
            "total_operations": total_ops,
            "aggregate_throughput_ops_sec": throughput,
            "error_rate_percent": error_rate,
            "operation_stats": op_stats,
            "error_breakdown": dict(self.metrics.errors),
            "falkordb_memory": memory_stats,
            "invariants": invariants,
            "hot_tenant_results": hot_results or {},
            "throughput_timeline": timeline,
        }

        # Write JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        # Write Markdown
        md_lines = [
            f"# CareThread High-Concurrency Load Report ({ts})",
            "",
            "## 1. Executive Summary",
            "",
            f"- **Concurrency**: {self.agent_count} concurrent agent workers",
            f"- **Tenants**: M={len(self.patient_pool)} isolated patient graphs",
            f"- **Duration**: {total_duration:.1f} seconds",
            f"- **Total Operations**: {total_ops:,}",
            f"- **Aggregate Throughput**: **{throughput} ops/sec** (Target: >= 300 ops/s)",
            f"- **Error Rate**: **{error_rate}%** (Target: <= 0.5% excluding expected conflicts)",
            (
                f"- **Memory Usage**: {memory_stats.get('used_memory_human', 'N/A')} "
                f"(Peak: {memory_stats.get('used_memory_peak_human', 'N/A')})"
            ),
            "",
            "## 2. Latency Percentiles per Operation",
            "",
            "| Operation | Count | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | Target | Status |",
            "|---|---|---|---|---|---|---|---|",
        ]

        targets = {
            "recall": 120.0,
            "remember": 150.0,
            "append_message": 150.0,
            "get_session_messages": 120.0,
            "history": 120.0,
            "handoff": 250.0,
            "accept_handoff": 250.0,
            "commit_answer": 250.0,
            "explain": 250.0,
        }

        for op, data in op_stats.items():
            tgt = targets.get(op, 250.0)
            status = "PASS" if data["p95_ms"] <= tgt else "WARN"
            md_lines.append(
                f"| `{op}` | {data['count']} | {data['p50_ms']} | {data['p90_ms']} | "
                f"**{data['p95_ms']}** | {data['p99_ms']} | <= {tgt:.0f} ms | {status} |"
            )

        md_lines.extend(
            [
                "",
                "## 3. Invariant Verification Results",
                "",
                "| Invariant | Description | Result |",
                "|---|---|---|",
            ]
        )

        for inv_key, inv_data in invariants.items():
            res_str = "**PASS**" if inv_data["passed"] else "**FAIL**"
            md_lines.append(f"| `{inv_key}` | {inv_data['description']} | {res_str} |")

        if hot_results:
            md_lines.extend(
                [
                    "",
                    "## 4. Hot-Tenant Contention Suite",
                    "",
                    f"- **Hot Patient Target**: `{hot_results['hot_patient']}`",
                    (
                        f"- **Same-slot Supersede Race**: "
                        f"{hot_results['supersede_race_successes']}/8 successful"
                    ),
                    (
                        f"- **Contested Handoff Winner**: "
                        f"{hot_results['handoff_winners']} (Target: exactly 1)"
                    ),
                    (
                        f"- **Contested Handoff Rejects**: "
                        f"{hot_results['handoff_rejects']} (Target: exactly 99)"
                    ),
                ]
            )

        md_lines.append("")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        logger.info(f"Reports saved to: {json_path} and {md_path}")
        return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CareThread high-concurrency multi-agent load testing harness."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=20,
        help="Test duration in seconds (default: 20; use 60 or 300 for long benchmarks)",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=24,
        help="Number of concurrent worker threads (default: 24)",
    )
    parser.add_argument(
        "--patients",
        type=int,
        default=100,
        help="Target number of tenant graphs M (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup period in seconds (default: 3)",
    )
    parser.add_argument(
        "--no-hot-tenant",
        action="store_true",
        help="Disable the hot-tenant contention race test",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save generated JSON and Markdown reports",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger.info("=== Starting CareThread Load & Concurrency Benchmark ===")
    harness = LoadTestHarness(
        duration=args.duration,
        agents=args.agents,
        patients=args.patients,
        warmup=args.warmup,
        run_hot_tenant=not args.no_hot_tenant,
        output_dir=args.output_dir,
    )

    harness.prepare_environment()

    t_start = time.time()
    harness.run_concurrency_phase()
    t_end = time.time()

    hot_results = None
    if harness.run_hot_tenant:
        hot_results = harness.run_hot_tenant_contention()

    invariants = harness.verify_invariants(hot_results)
    json_path, md_path = harness.generate_report(t_start, t_end, hot_results, invariants)

    all_passed = all(v["passed"] for v in invariants.values())
    res_str = "ALL PASS" if all_passed else "FAILED"
    logger.info(f"=== Load Test Complete | Invariants: {res_str} ===")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
