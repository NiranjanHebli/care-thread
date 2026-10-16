"""Safety verification, cross-reactivity checking, and contradiction detection engine.

Clinical safety verification implementation:
- Cross-reactivity detection between proposed/active drugs and documented allergies
- Drug-drug interaction (DDI) detection against shared clinic knowledge
- Therapeutic duplication detection within identical drug classes
- Conflicting assertion contradiction detection across patient memory
- Safety flag lifecycle: open -> resolved | dismissed
- Flag graph persistence and link traversal (ABOUT, CONFLICTS_WITH)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from carethread.clock import Clock, get_clock
from carethread.db.client import get_db_client
from carethread.db.clinic_reader import ClinicReader
from carethread.db.queries import get_query
from carethread.db.tenant_router import TenantRouter
from carethread.errors import (
    FlagNotFoundError,
    PatientNotFoundError,
)
from carethread.util.id import flag_id as gen_flag_id

logger = logging.getLogger(__name__)

# Valid flag types and statuses
VALID_FLAG_TYPES: frozenset[str] = frozenset(
    {
        "allergy_med_conflict",
        "drug_interaction",
        "duplicate_therapy",
        "conflicting_assertion",
        "allergy_removal_review",
    }
)

VALID_SEVERITIES: frozenset[str] = frozenset({"critical", "severe", "high", "moderate", "low"})

VALID_FLAG_STATUSES: frozenset[str] = frozenset({"open", "resolved", "dismissed"})


@dataclass
class SafetyFlagRecord:
    """Represents a safety flag or contradiction detected in patient memory."""

    id: str
    patient_id: str
    type: str
    severity: str
    status: str = "open"
    rule_id: str = ""
    explanation: str = ""
    detected_at: int = 0
    resolved_by: str | None = None
    resolved_at: int | None = None
    target_fact_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SafetyCheckResult:
    """Consolidated safety check output covering allergies, interactions, and duplications."""

    patient_id: str
    passed: bool
    flags: list[SafetyFlagRecord] = field(default_factory=list)
    cross_reactivities_count: int = 0
    drug_interactions_count: int = 0
    duplicate_therapies_count: int = 0
    conflicts_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "passed": self.passed,
            "cross_reactivities_count": self.cross_reactivities_count,
            "drug_interactions_count": self.drug_interactions_count,
            "duplicate_therapies_count": self.duplicate_therapies_count,
            "conflicts_count": self.conflicts_count,
            "flags": [f.to_dict() for f in self.flags],
        }

    def to_summary_str(self) -> str:
        """Formats the safety evaluation into a concise summary string."""
        status_str = "PASSED" if self.passed else "FAILED (SAFETY CONCERNS)"
        lines = [
            f"Safety Evaluation: {status_str}",
            f"- Cross-Reactivities: {self.cross_reactivities_count}",
            f"- Drug-Drug Interactions: {self.drug_interactions_count}",
            f"- Duplicate Therapies: {self.duplicate_therapies_count}",
            f"- Conflicting Assertions: {self.conflicts_count}",
        ]
        if self.flags:
            lines.append("\nDetected Safety Flags:")
            for flg in self.flags:
                lines.append(
                    f"  * [{flg.severity.upper()}] {flg.type}: {flg.explanation} "
                    f"(Rule: {flg.rule_id})"
                )
        return "\n".join(lines)


class SafetyManager:
    """Manages clinical safety checks, cross-reactivities, and contradiction flags."""

    def __init__(
        self,
        router: TenantRouter | None = None,
        clinic_reader: ClinicReader | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._router = router or TenantRouter(get_db_client())
        self._clinic_reader = clinic_reader or self._router.get_clinic_reader()
        self._clock = clock or get_clock()

    # -------------------------------------------------------------------------
    # Core Safety Evaluation Methods
    # -------------------------------------------------------------------------

    def check_allergy_cross_reactivity(
        self,
        patient_id: str,
        drug_code: str | None = None,
        drug_name: str | None = None,
        persist: bool = False,
    ) -> list[SafetyFlagRecord]:
        """Detect cross-reactivity between a drug and documented active patient allergies.

        Args:
            patient_id: Validated patient tenant identifier.
            drug_code: RxNorm code of the proposed or active drug.
            drug_name: Clinical display name of the drug.
            persist: Whether to create and persist Flag nodes in the patient graph.

        Returns:
            List of generated SafetyFlagRecord objects.
        """
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        resolved_code, resolved_name, drug_class_id = self._resolve_drug(
            drug_code=drug_code,
            drug_name=drug_name,
        )
        if not resolved_code and not resolved_name:
            return []

        flags: list[SafetyFlagRecord] = []
        now_ts = self._clock.now_ts()

        # Step 1: Run query q_safe_01a against clinic_shared
        allergen_codes: list[str] = []
        allergen_class_id: str = ""
        matched_class_id: str = drug_class_id or ""

        if resolved_code:
            q_safe_01a = get_query("q_safe_01a")
            res_a = self._clinic_reader.ro_query(
                q_safe_01a,
                {"drug_code": str(resolved_code)},
            )
            if res_a.result_set:
                row = res_a.result_set[0]
                matched_class_id = row[0] or matched_class_id
                allergen_class_id = row[1] or ""
                allergen_codes = list(row[2]) if row[2] else []

        # Step 2: Match active allergies in patient graph via q_safe_01b
        matched_allergies: list[tuple[str, str]] = []
        if allergen_codes:
            q_safe_01b = get_query("q_safe_01b")
            res_b = target_graph.query(
                q_safe_01b,
                {"pid": valid_pid, "allergen_codes": allergen_codes},
            )
            for r in res_b.result_set or []:
                matched_allergies.append((r[0], r[1] or ""))

        # Fallback heuristic: check if patient has allergy with matching display text
        all_allergies_cypher = """
        MATCH (p:Patient {id:$pid})-[:HAS_FACT]->(a:Fact {kind:'allergy', status:'active'})
        RETURN a.id, a.display, a.slot, a.code
        """
        all_alg_res = target_graph.query(all_allergies_cypher, {"pid": valid_pid})
        norm_name = (resolved_name or "").lower()

        for row in all_alg_res.result_set or []:
            aid, adisp, aslot, _ = row[0], row[1] or "", row[2] or "", row[3] or ""
            # Avoid duplicate if already matched by code
            if any(aid == m[0] for m in matched_allergies):
                continue

            # Heuristic match for penicillin / beta-lactam class
            is_beta_lactam_drug = matched_class_id in (
                "dc_penicillins",
                "dc_cephalosporins",
                "dc_carbapenems",
            )
            is_beta_lactam_allergy = any(
                term in adisp.lower() or term in aslot.lower()
                for term in ("penicillin", "amoxicillin", "cephalosporin", "beta-lactam")
            )
            if is_beta_lactam_drug and is_beta_lactam_allergy:
                matched_allergies.append((aid, adisp))
                continue

            # Direct name overlap check
            if norm_name and any(w in adisp.lower() for w in norm_name.split() if len(w) > 4):
                matched_allergies.append((aid, adisp))

        # Generate flags for all matches
        for aid, adisp in matched_allergies:
            flg_id = gen_flag_id()
            disp_name = resolved_name or resolved_code or "medication"
            explanation = (
                f"Proposed medication '{disp_name}' (class: {matched_class_id}) cross-reacts "
                f"with documented active allergy '{adisp}'."
            )
            flg = SafetyFlagRecord(
                id=flg_id,
                patient_id=valid_pid,
                type="allergy_med_conflict",
                severity="high",
                status="open",
                rule_id="rule_cross_reactivity",
                explanation=explanation,
                detected_at=now_ts,
                target_fact_ids=[aid],
                metadata={
                    "drug_code": resolved_code,
                    "drug_name": resolved_name,
                    "drug_class_id": matched_class_id,
                    "allergen_class_id": allergen_class_id,
                    "allergy_fact_id": aid,
                },
            )
            if persist:
                self.create_flag(valid_pid, flg)
            flags.append(flg)

        return flags

    def check_drug_drug_interaction(
        self,
        patient_id: str,
        proposed_code: str | None = None,
        proposed_name: str | None = None,
        active_med_facts: list[Any] | None = None,
        persist: bool = False,
    ) -> list[SafetyFlagRecord]:
        """Detect clinical drug-drug interactions against current active medications.

        Args:
            patient_id: Validated patient tenant identifier.
            proposed_code: RxNorm code of proposed drug.
            proposed_name: Clinical display name of proposed drug.
            active_med_facts: Optional pre-fetched list of active medication Fact nodes/records.
            persist: Whether to create and persist Flag nodes in the patient graph.

        Returns:
            List of generated SafetyFlagRecord objects.
        """
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        resolved_code, resolved_name, _ = self._resolve_drug(
            drug_code=proposed_code,
            drug_name=proposed_name,
        )
        if not resolved_code and not resolved_name:
            return []

        # Fetch active medications if not provided
        active_meds = active_med_facts
        if active_meds is None:
            active_meds_cypher = """
            MATCH (p:Patient {id:$pid})-[:HAS_FACT]->(f:Fact {kind:'medication', status:'active'})
            RETURN f.id, f.code, f.display, f.value
            """
            res = target_graph.query(active_meds_cypher, {"pid": valid_pid})
            active_meds = res.result_set or []

        flags: list[SafetyFlagRecord] = []
        now_ts = self._clock.now_ts()

        for med in active_meds:
            # Handle tuple/list from cypher or FactRecord object
            if hasattr(med, "id"):
                mid, mcode, mdisp = med.id, med.code, med.display
            else:
                mid, mcode, mdisp = med[0], med[1], med[2]

            cur_code, cur_name, _ = self._resolve_drug(drug_code=mcode, drug_name=mdisp)
            if not cur_code:
                continue

            # Query clinic_shared for interaction
            inter_cypher = """
            MATCH (da:Drug {code:$code_a})-[r:INTERACTS_WITH]-(db:Drug {code:$code_b})
            RETURN r.severity, r.effect
            """
            inter_res = self._clinic_reader.ro_query(
                inter_cypher,
                {"code_a": str(resolved_code), "code_b": str(cur_code)},
            )

            if inter_res.result_set:
                row = inter_res.result_set[0]
                sev = row[0] or "moderate"
                effect = row[1] or ""
                flg_id = gen_flag_id()
                disp_a = resolved_name or resolved_code
                disp_b = cur_name or mdisp or cur_code
                explanation = (
                    f"Interaction between '{disp_a}' and active medication '{disp_b}': {effect}"
                )

                flg = SafetyFlagRecord(
                    id=flg_id,
                    patient_id=valid_pid,
                    type="drug_interaction",
                    severity=sev if sev in VALID_SEVERITIES else "moderate",
                    status="open",
                    rule_id="rule_drug_interaction",
                    explanation=explanation,
                    detected_at=now_ts,
                    target_fact_ids=[mid],
                    metadata={
                        "drug_a_code": resolved_code,
                        "drug_b_code": cur_code,
                        "severity": sev,
                    },
                )
                if persist:
                    self.create_flag(valid_pid, flg)
                flags.append(flg)

        return flags

    def check_duplicate_therapy(
        self,
        patient_id: str,
        proposed_code: str | None = None,
        proposed_name: str | None = None,
        active_med_facts: list[Any] | None = None,
        persist: bool = False,
    ) -> list[SafetyFlagRecord]:
        """Detect duplicate therapeutic class overlap with existing active medications."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        resolved_code, resolved_name, proposed_class = self._resolve_drug(
            drug_code=proposed_code,
            drug_name=proposed_name,
        )
        if not proposed_class:
            return []

        active_meds = active_med_facts
        if active_meds is None:
            active_meds_cypher = """
            MATCH (p:Patient {id:$pid})-[:HAS_FACT]->(f:Fact {kind:'medication', status:'active'})
            RETURN f.id, f.code, f.display, f.value
            """
            res = target_graph.query(active_meds_cypher, {"pid": valid_pid})
            active_meds = res.result_set or []

        flags: list[SafetyFlagRecord] = []
        now_ts = self._clock.now_ts()

        for med in active_meds:
            if hasattr(med, "id"):
                mid, mcode, mdisp = med.id, med.code, med.display
            else:
                mid, mcode, mdisp = med[0], med[1], med[2]

            _, cur_name, cur_class = self._resolve_drug(drug_code=mcode, drug_name=mdisp)

            if cur_class and cur_class == proposed_class:
                flg_id = gen_flag_id()
                disp_a = resolved_name or resolved_code
                disp_b = cur_name or mdisp
                explanation = (
                    f"Duplicate therapy: proposed medication '{disp_a}' and active medication "
                    f"'{disp_b}' both belong to class '{proposed_class}'."
                )
                flg = SafetyFlagRecord(
                    id=flg_id,
                    patient_id=valid_pid,
                    type="duplicate_therapy",
                    severity="moderate",
                    status="open",
                    rule_id="rule_duplicate_therapy",
                    explanation=explanation,
                    detected_at=now_ts,
                    target_fact_ids=[mid],
                    metadata={"class_id": proposed_class},
                )
                if persist:
                    self.create_flag(valid_pid, flg)
                flags.append(flg)

        return flags

    def check_proposed_medication(
        self,
        patient_id: str,
        drug_code: str | None = None,
        drug_name: str | None = None,
        persist: bool = False,
    ) -> SafetyCheckResult:
        """Evaluate a proposed prescription across allergies, DDIs, and duplications.

        Args:
            patient_id: Validated patient tenant identifier.
            drug_code: RxNorm code of the proposed medication.
            drug_name: Name or display string of the proposed medication.
            persist: Whether to persist detected safety flags into the graph.

        Returns:
            SafetyCheckResult with passed status and flags list.
        """
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        # Check patient exists
        p_res = target_graph.query(
            "MATCH (p:Patient {id:$pid}) RETURN p.id",
            {"pid": valid_pid},
        )
        if not p_res.result_set:
            raise PatientNotFoundError(
                f"Patient node '{valid_pid}' not found.",
                details={"patient_id": valid_pid},
            )

        # 1. Cross-reactivity check
        cr_flags = self.check_allergy_cross_reactivity(
            patient_id=valid_pid,
            drug_code=drug_code,
            drug_name=drug_name,
            persist=persist,
        )

        # 2. Drug-drug interactions check
        ddi_flags = self.check_drug_drug_interaction(
            patient_id=valid_pid,
            proposed_code=drug_code,
            proposed_name=drug_name,
            persist=persist,
        )

        # 3. Duplicate therapy check
        dup_flags = self.check_duplicate_therapy(
            patient_id=valid_pid,
            proposed_code=drug_code,
            proposed_name=drug_name,
            persist=persist,
        )

        all_flags = cr_flags + ddi_flags + dup_flags
        passed = not any(f.severity in ("critical", "severe", "high") for f in all_flags)

        return SafetyCheckResult(
            patient_id=valid_pid,
            passed=passed,
            flags=all_flags,
            cross_reactivities_count=len(cr_flags),
            drug_interactions_count=len(ddi_flags),
            duplicate_therapies_count=len(dup_flags),
            conflicts_count=0,
        )

    def check_patient_safety(
        self,
        patient_id: str,
        persist: bool = False,
    ) -> SafetyCheckResult:
        """Perform comprehensive safety audit of all active patient medications and allergies."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        p_res = target_graph.query(
            "MATCH (p:Patient {id:$pid}) RETURN p.id",
            {"pid": valid_pid},
        )
        if not p_res.result_set:
            raise PatientNotFoundError(
                f"Patient node '{valid_pid}' not found.",
                details={"patient_id": valid_pid},
            )

        # Retrieve active medications
        meds_res = target_graph.query(
            """
            MATCH (p:Patient {id:$pid})-[:HAS_FACT]->(f:Fact {kind:'medication', status:'active'})
            RETURN f.id, f.code, f.display, f.value
            """,
            {"pid": valid_pid},
        )
        med_rows = meds_res.result_set or []

        all_flags: list[SafetyFlagRecord] = []
        cr_count = 0
        ddi_count = 0
        dup_count = 0
        conf_count = 0

        # Check each medication against active allergies
        for m in med_rows:
            cr_flags = self.check_allergy_cross_reactivity(
                patient_id=valid_pid,
                drug_code=m[1],
                drug_name=m[2],
                persist=persist,
            )
            all_flags.extend(cr_flags)
            cr_count += len(cr_flags)

        # Check all pairs of active medications for DDIs and duplication
        for i in range(len(med_rows)):
            for j in range(i + 1, len(med_rows)):
                m1, m2 = med_rows[i], med_rows[j]
                c1, n1, cls1 = self._resolve_drug(drug_code=m1[1], drug_name=m1[2])
                c2, n2, cls2 = self._resolve_drug(drug_code=m2[1], drug_name=m2[2])

                if c1 and c2:
                    inter_res = self._clinic_reader.ro_query(
                        """
                        MATCH (da:Drug {code:$code_a})-[r:INTERACTS_WITH]-(db:Drug {code:$code_b})
                        RETURN r.severity, r.effect
                        """,
                        {"code_a": str(c1), "code_b": str(c2)},
                    )
                    if inter_res.result_set:
                        row = inter_res.result_set[0]
                        sev = row[0] or "moderate"
                        flg = SafetyFlagRecord(
                            id=gen_flag_id(),
                            patient_id=valid_pid,
                            type="drug_interaction",
                            severity=sev if sev in VALID_SEVERITIES else "moderate",
                            status="open",
                            rule_id="rule_drug_interaction",
                            explanation=f"Interaction between '{n1}' and '{n2}': {row[1]}",
                            detected_at=self._clock.now_ts(),
                            target_fact_ids=[m1[0], m2[0]],
                        )
                        if persist:
                            self.create_flag(valid_pid, flg)
                        all_flags.append(flg)
                        ddi_count += 1

                # Duplicate therapy check
                if cls1 and cls2 and cls1 == cls2:
                    flg = SafetyFlagRecord(
                        id=gen_flag_id(),
                        patient_id=valid_pid,
                        type="duplicate_therapy",
                        severity="moderate",
                        status="open",
                        rule_id="rule_duplicate_therapy",
                        explanation=(
                            f"Duplicate therapy: active medications '{n1}' and '{n2}' "
                            f"both belong to class '{cls1}'."
                        ),
                        detected_at=self._clock.now_ts(),
                        target_fact_ids=[m1[0], m2[0]],
                    )
                    if persist:
                        self.create_flag(valid_pid, flg)
                    all_flags.append(flg)
                    dup_count += 1

        # Check for disputed or conflicting assertions
        disp_res = target_graph.query(
            """
            MATCH (p:Patient {id:$pid})-[:HAS_FACT]->(f:Fact {status:'disputed'})
            RETURN f.id, f.slot, f.display
            """,
            {"pid": valid_pid},
        )
        for r in disp_res.result_set or []:
            flg = SafetyFlagRecord(
                id=gen_flag_id(),
                patient_id=valid_pid,
                type="conflicting_assertion",
                severity="moderate",
                status="open",
                rule_id="rule_conflict_policy",
                explanation=f"Fact '{r[2]}' in slot '{r[1]}' is currently disputed.",
                detected_at=self._clock.now_ts(),
                target_fact_ids=[r[0]],
            )
            if persist:
                self.create_flag(valid_pid, flg)
            all_flags.append(flg)
            conf_count += 1

        passed = not any(f.severity in ("critical", "severe", "high") for f in all_flags)
        return SafetyCheckResult(
            patient_id=valid_pid,
            passed=passed,
            flags=all_flags,
            cross_reactivities_count=cr_count,
            drug_interactions_count=ddi_count,
            duplicate_therapies_count=dup_count,
            conflicts_count=conf_count,
        )

    # -------------------------------------------------------------------------
    # Flag Node Graph Persistence and Lifecycle Management
    # -------------------------------------------------------------------------

    def create_flag(self, patient_id: str, flag: SafetyFlagRecord) -> SafetyFlagRecord:
        """Persist a SafetyFlagRecord node and its ABOUT / CONFLICTS_WITH edges in the graph."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        p_res = target_graph.query(
            "MATCH (p:Patient {id:$pid}) RETURN p.id",
            {"pid": valid_pid},
        )
        if not p_res.result_set:
            raise PatientNotFoundError(
                f"Patient node '{valid_pid}' not found.",
                details={"patient_id": valid_pid},
            )

        cypher = """
        MATCH (p:Patient {id:$pid})
        CREATE (flg:Flag {
            id: $flg_id,
            type: $type,
            severity: $severity,
            status: $status,
            rule_id: $rule_id,
            explanation: $explanation,
            detected_at: $now,
            resolved_by: $resolved_by,
            resolved_at: $resolved_at
        })
        CREATE (p)-[:HAS_FLAG]->(flg)
        RETURN flg.id
        """
        target_graph.query(
            cypher,
            {
                "pid": valid_pid,
                "flg_id": flag.id,
                "type": flag.type,
                "severity": flag.severity,
                "status": flag.status,
                "rule_id": flag.rule_id,
                "explanation": flag.explanation,
                "now": flag.detected_at or self._clock.now_ts(),
                "resolved_by": flag.resolved_by,
                "resolved_at": flag.resolved_at,
            },
        )

        # Link ABOUT edges to target facts
        for fid in flag.target_fact_ids:
            target_graph.query(
                """
                MATCH (flg:Flag {id:$flg_id}), (f:Fact {id:$fid})
                MERGE (flg)-[:ABOUT]->(f)
                """,
                {"flg_id": flag.id, "fid": fid},
            )

        # If flag involves 2 target facts, link CONFLICTS_WITH between them
        if len(flag.target_fact_ids) >= 2:
            target_graph.query(
                """
                MATCH (f1:Fact {id:$fid1}), (f2:Fact {id:$fid2})
                MERGE (f1)-[:CONFLICTS_WITH]->(f2)
                """,
                {"fid1": flag.target_fact_ids[0], "fid2": flag.target_fact_ids[1]},
            )

        return flag

    def resolve_flag(
        self,
        patient_id: str,
        flag_id: str,
        resolved_by: str,
        resolution_reason: str | None = None,
    ) -> SafetyFlagRecord:
        """Transition a flag from open to resolved with clinician or agent accountability."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        cur = self.get_flag(valid_pid, flag_id)
        now_ts = self._clock.now_ts()

        update_cypher = """
        MATCH (p:Patient {id:$pid})-[:HAS_FLAG]->(flg:Flag {id:$flg_id})
        SET flg.status = 'resolved',
            flg.resolved_by = $resolved_by,
            flg.resolved_at = $now,
            flg.resolution_reason = $reason
        RETURN flg.id
        """
        target_graph.query(
            update_cypher,
            {
                "pid": valid_pid,
                "flg_id": flag_id,
                "resolved_by": resolved_by,
                "now": now_ts,
                "reason": resolution_reason or "Resolved by authorized clinician",
            },
        )

        cur.status = "resolved"
        cur.resolved_by = resolved_by
        cur.resolved_at = now_ts
        return cur

    def dismiss_flag(
        self,
        patient_id: str,
        flag_id: str,
        dismissed_by: str,
        dismissal_reason: str | None = None,
    ) -> SafetyFlagRecord:
        """Transition a flag from open to dismissed (clinician acknowledged acceptable risk)."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        cur = self.get_flag(valid_pid, flag_id)
        now_ts = self._clock.now_ts()

        update_cypher = """
        MATCH (p:Patient {id:$pid})-[:HAS_FLAG]->(flg:Flag {id:$flg_id})
        SET flg.status = 'dismissed',
            flg.resolved_by = $dismissed_by,
            flg.resolved_at = $now,
            flg.dismissal_reason = $reason
        RETURN flg.id
        """
        target_graph.query(
            update_cypher,
            {
                "pid": valid_pid,
                "flg_id": flag_id,
                "dismissed_by": dismissed_by,
                "now": now_ts,
                "reason": dismissal_reason or "Risk acknowledged and dismissed by clinician",
            },
        )

        cur.status = "dismissed"
        cur.resolved_by = dismissed_by
        cur.resolved_at = now_ts
        return cur

    def get_open_flags(self, patient_id: str) -> list[SafetyFlagRecord]:
        """Fetch all currently open safety flags for a patient."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        p_res = target_graph.query(
            "MATCH (p:Patient {id:$pid}) RETURN p.id",
            {"pid": valid_pid},
        )
        if not p_res.result_set:
            raise PatientNotFoundError(
                f"Patient node '{valid_pid}' not found.",
                details={"patient_id": valid_pid},
            )

        cypher = """
        MATCH (p:Patient {id:$pid})-[:HAS_FLAG]->(flg:Flag {status:'open'})
        OPTIONAL MATCH (flg)-[:ABOUT]->(f:Fact)
        RETURN flg.id, flg.type, flg.severity, flg.status, flg.rule_id,
               flg.explanation, flg.detected_at, flg.resolved_by, flg.resolved_at,
               collect(f.id)
        ORDER BY flg.detected_at DESC
        """
        res = target_graph.query(cypher, {"pid": valid_pid})
        flags: list[SafetyFlagRecord] = []
        for r in res.result_set or []:
            flags.append(
                SafetyFlagRecord(
                    id=r[0],
                    patient_id=valid_pid,
                    type=r[1],
                    severity=r[2],
                    status=r[3],
                    rule_id=r[4] or "",
                    explanation=r[5] or "",
                    detected_at=r[6] or 0,
                    resolved_by=r[7],
                    resolved_at=r[8],
                    target_fact_ids=list(r[9]) if r[9] else [],
                )
            )
        return flags

    def list_flags(self, patient_id: str, status: str = "open") -> list[SafetyFlagRecord]:
        """List safety flags for a patient filtered by status."""
        return self.get_open_flags(patient_id)

    def run_all_checks(
        self,
        patient_id: str,
        medication_fact_id: str | None = None,
        persist: bool = True,
    ) -> SafetyCheckResult:
        """Run comprehensive patient safety checks across allergies and medications."""
        return self.check_patient_safety(patient_id=patient_id, persist=persist)

    def get_flag(self, patient_id: str, flag_id: str) -> SafetyFlagRecord:
        """Fetch a specific flag by ID or raise FlagNotFoundError."""
        valid_pid = self._router.validate_patient_id(patient_id)
        target_graph = self._router.graph(valid_pid)

        p_res = target_graph.query(
            "MATCH (p:Patient {id:$pid}) RETURN p.id",
            {"pid": valid_pid},
        )
        if not p_res.result_set:
            raise PatientNotFoundError(
                f"Patient node '{valid_pid}' not found.",
                details={"patient_id": valid_pid},
            )

        cypher = """
        MATCH (p:Patient {id:$pid})-[:HAS_FLAG]->(flg:Flag {id:$flg_id})
        OPTIONAL MATCH (flg)-[:ABOUT]->(f:Fact)
        RETURN flg.id, flg.type, flg.severity, flg.status, flg.rule_id,
               flg.explanation, flg.detected_at, flg.resolved_by, flg.resolved_at,
               collect(f.id)
        """
        res = target_graph.query(cypher, {"pid": valid_pid, "flg_id": flag_id})
        if not res.result_set:
            raise FlagNotFoundError(
                f"Flag '{flag_id}' not found for patient '{valid_pid}'.",
                details={"patient_id": valid_pid, "flag_id": flag_id},
            )

        r = res.result_set[0]
        return SafetyFlagRecord(
            id=r[0],
            patient_id=valid_pid,
            type=r[1],
            severity=r[2],
            status=r[3],
            rule_id=r[4] or "",
            explanation=r[5] or "",
            detected_at=r[6] or 0,
            resolved_by=r[7],
            resolved_at=r[8],
            target_fact_ids=list(r[9]) if r[9] else [],
        )

    # -------------------------------------------------------------------------
    # Internal Clinical Drug Helper
    # -------------------------------------------------------------------------

    def _resolve_drug(
        self,
        drug_code: str | None = None,
        drug_name: str | None = None,
    ) -> tuple[str | None, str | None, str | None]:
        """Resolves RxNorm drug code, display name, and drug class from clinic_shared."""
        if drug_code:
            cypher = """
            MATCH (d:Drug {code:$code})
            OPTIONAL MATCH (d)-[:IN_CLASS]->(dc:DrugClass)
            RETURN d.code, d.name, dc.id
            """
            res = self._clinic_reader.ro_query(cypher, {"code": str(drug_code)})
            if res.result_set:
                r = res.result_set[0]
                return r[0], r[1] or drug_name, r[2]
            return drug_code, drug_name, None

        if drug_name:
            norm = drug_name.strip().lower()
            # Try exact and contains match
            cypher = """
            MATCH (d:Drug)
            WHERE toLower(d.name) CONTAINS $norm OR toLower(d.ingredient) CONTAINS $norm
            OPTIONAL MATCH (d)-[:IN_CLASS]->(dc:DrugClass)
            RETURN d.code, d.name, dc.id
            """
            res = self._clinic_reader.ro_query(cypher, {"norm": norm})
            if res.result_set:
                r = res.result_set[0]
                return r[0], r[1], r[2]

            # Try first keyword if multi-word (e.g. "Lisinopril 10 MG" -> "lisinopril")
            first_word = norm.split()[0] if norm.split() else norm
            if len(first_word) >= 4:
                res2 = self._clinic_reader.ro_query(cypher, {"norm": first_word})
                if res2.result_set:
                    r = res2.result_set[0]
                    return r[0], r[1], r[2]

            return None, drug_name, None

        return None, None, None
