"""CareThread 3-Way Benchmark Runner.

Executes comparative evaluation across Stateless, Vector-only, and CareThread conditions.
Usage:
    python benchmark/run.py --conditions stateless,vector,carethread --runs 3 --seed 7
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure workspace root is in sys.path when run directly
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

from benchmark.baselines.carethread import CareThreadCondition
from benchmark.baselines.stateless import StatelessCondition
from benchmark.baselines.vector_only import VectorOnlyCondition
from benchmark.judge import BenchmarkJudge
from benchmark.report import BenchmarkReporter
from carethread.agents.llm import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")


def load_yaml(file_path: Path) -> dict[str, Any]:
    """Safely loads YAML data from the filesystem."""
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class BenchmarkOrchestrator:
    """Orchestrates scenario hydration, multi-run query evaluation, and scoring."""

    def __init__(
        self,
        conditions: list[str],
        runs_count: int = 3,
        seed: int = 7,
        llm_mode: str = "mock",
        output_dir: str = "reports",
    ) -> None:
        self.condition_names = [c.strip().lower() for c in conditions]
        self.runs_count = max(1, runs_count)
        self.seed = seed
        self.llm_mode = llm_mode
        self.output_dir = output_dir

        self.judge = BenchmarkJudge()
        self.reporter = BenchmarkReporter(output_dir)

        base_dir = Path(__file__).resolve().parent
        self.scenarios_file = base_dir / "scenarios" / "scenarios.yaml"
        self.questions_file = base_dir / "questions.yaml"

        self.scenarios: list[dict[str, Any]] = []
        self.questions: list[dict[str, Any]] = []

    def load_data(self) -> None:
        """Loads benchmark scenarios and question evaluation bank."""
        logger.info(f"Loading benchmark scenarios from {self.scenarios_file}...")
        scen_data = load_yaml(self.scenarios_file)
        self.scenarios = scen_data.get("scenarios", [])

        logger.info(f"Loading gold questions from {self.questions_file}...")
        q_data = load_yaml(self.questions_file)
        self.questions = q_data.get("questions", [])

        logger.info(f"Loaded {len(self.scenarios)} scenarios and {len(self.questions)} questions.")

    def _init_condition(self, name: str, llm: LLMClient) -> Any:
        """Instantiates the specified baseline condition runner."""
        if name in ("stateless", "a"):
            return StatelessCondition(llm_client=llm)
        elif name in ("vector", "vector_only", "b"):
            return VectorOnlyCondition(llm_client=llm)
        elif name in ("carethread", "graph", "c"):
            return CareThreadCondition(llm_client=llm)
        else:
            raise ValueError(f"Unknown benchmark condition: '{name}'")

    def run(self) -> tuple[Path, Path]:
        """Executes all conditions across multiple runs and outputs reports."""
        random.seed(self.seed)
        llm_client = LLMClient(mode=self.llm_mode)

        # Map condition names to canonical labels
        canonical_map = {
            "stateless": "stateless",
            "a": "stateless",
            "vector": "vector_only",
            "vector_only": "vector_only",
            "b": "vector_only",
            "carethread": "carethread",
            "graph": "carethread",
            "c": "carethread",
        }
        targets = [canonical_map.get(c, c) for c in self.condition_names]

        # Results container: condition -> list of run evaluations
        all_results: dict[str, list[list[dict[str, Any]]]] = {t: [] for t in targets}

        for run_idx in range(1, self.runs_count + 1):
            logger.info(f"=== Starting Benchmark Run {run_idx}/{self.runs_count} ===")

            for cond_name in targets:
                logger.info(f"Evaluating condition '{cond_name}' (Run {run_idx})...")
                runner = self._init_condition(cond_name, llm_client)

                # Step 1: Setup scenarios
                for scen in self.scenarios:
                    runner.setup_scenario(scen)

                run_evals: list[dict[str, Any]] = []

                # Step 2: Answer all questions
                for q in self.questions:
                    ans_result = runner.answer_question(q)

                    # For CareThread, evaluate ablation sensitivity on a subset
                    ablated_res = None
                    if cond_name == "carethread" and q.get("required_memory_ids"):
                        ablated_res = runner.answer_question(q, ablate_citations=True)

                    score = self.judge.evaluate_answer(q, ans_result, ablated_res)
                    run_evals.append(score)

                # Step 3: Teardown
                runner.teardown()
                all_results[cond_name].append(run_evals)

                avg_acc = sum(e["recall_accuracy"] for e in run_evals) / max(1, len(run_evals))
                logger.info(
                    f"Condition '{cond_name}' Run {run_idx} complete: "
                    f"Mean Recall Accuracy = {avg_acc * 100:.1f}%"
                )

        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        metadata = {
            "timestamp": ts_str,
            "conditions": targets,
            "runs_count": self.runs_count,
            "seed": self.seed,
            "llm_mode": self.llm_mode,
            "llm_model": llm_client.get_model_for_role("agent"),
            "embedder": "BAAI/bge-small-en-v1.5",
            "scenarios_count": len(self.scenarios),
            "questions_count": len(self.questions),
        }

        json_path, md_path = self.reporter.generate_report(metadata, all_results, ts_str)
        logger.info(f"Reports saved to {json_path} and {md_path}")
        return json_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CareThread 3-Way Benchmark Runner (Stateless vs Vector vs Graph)."
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="stateless,vector,carethread",
        help="Comma-separated conditions: stateless, vector, carethread (default: all)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of evaluation runs per condition (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducibility (default: 7)",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="mock",
        choices=["mock", "replay", "live"],
        help="LLM inference mode (default: mock)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Output report directory (default: reports)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger.info("=== Initializing CareThread 3-Way Benchmark Suite ===")
    orchestrator = BenchmarkOrchestrator(
        conditions=args.conditions.split(","),
        runs_count=args.runs,
        seed=args.seed,
        llm_mode=args.llm,
        output_dir=args.output_dir,
    )
    orchestrator.load_data()

    t_start = time.time()
    json_path, md_path = orchestrator.run()
    elapsed = time.time() - t_start

    logger.info(f"=== Benchmark Completed in {elapsed:.1f}s ===")
    logger.info(f"JSON Report: {json_path}")
    logger.info(f"Markdown Report: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
