"""Automated evaluation judge for the 3-Way Benchmark.

Implements multi-dimensional metrics across:
- Recall accuracy
- What-changed / temporal correctness
- Abstention / negative accuracy
- Handoff completeness
- Explanation faithfulness (trace validity, claim support, ablation sensitivity)
"""

from __future__ import annotations

import re
from typing import Any


class BenchmarkJudge:
    """Evaluates benchmark answers against gold standards and tracks aggregate metrics."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalizes clinical text for robust token/string matching."""
        clean = text.lower().strip()
        clean = re.sub(r"(\d+)([a-zA-Z]+)", r"\1 \2", clean)
        clean = re.sub(r"[^\w\s]", " ", clean)
        return " ".join(clean.split())

    def evaluate_answer(
        self,
        question: dict[str, Any],
        result: dict[str, Any],
        ablated_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Scores a single question result against gold standards."""
        category = question.get("category", "cross_session_recall")
        gold_entities = question.get("gold_entities", [])
        gold_answer = question.get("gold_answer", "")
        answer_text = result.get("answer", "")
        norm_answer = self._normalize(answer_text)

        is_abs = bool(question.get("is_abstention", False))

        # 1. Recall Accuracy
        # Check if all or majority of gold entities are present
        if is_abs:
            recall_acc = 1.0 if any(ent.lower() in norm_answer for ent in gold_entities) else 0.0
        elif gold_entities:
            matches = [
                self._normalize(ent) in norm_answer or ent.lower() in answer_text.lower()
                for ent in gold_entities
            ]
            recall_acc = 1.0 if all(matches) else (sum(matches) / len(matches) if matches else 0.0)
        else:
            recall_acc = 1.0 if self._normalize(gold_answer) in norm_answer else 0.0

        # 2. What-Changed Correctness
        what_changed_score = 0.0
        if category == "what_changed":
            prior = question.get("prior_value", "")
            new_v = question.get("new_value", "")
            has_prior = self._normalize(prior) in norm_answer if prior else True
            has_new = self._normalize(new_v) in norm_answer if new_v else True
            # Partial credit if at least the new value or prior is captured
            if has_prior and has_new:
                what_changed_score = 1.0
            elif has_new:
                what_changed_score = 0.5
            elif has_prior:
                what_changed_score = 0.25

        # 3. Abstention Accuracy
        abstention_score = 0.0
        if is_abs:
            abstain_triggers = [
                "not recorded",
                "not found",
                "no record",
                "unknown",
                "unrecorded",
                "not mentioned",
            ]
            stated_abstain = any(t in norm_answer for t in abstain_triggers)
            abstention_score = 1.0 if stated_abstain else 0.0
        else:
            abstention_score = 1.0  # Non-abstention questions pass by default

        # 4. Handoff Completeness
        handoff_score = 0.0
        if category == "handoff":
            req_fields = question.get("required_packet_fields", ["goal", "state_summary"])
            field_matches = [
                any(ent.lower() in answer_text.lower() for ent in question.get("gold_entities", []))
            ]
            has_fields = any(f.lower() in answer_text.lower() for f in req_fields)
            handoff_score = 1.0 if (all(field_matches) or has_fields) else 0.5
        else:
            handoff_score = 1.0

        # 5. Explanation Faithfulness
        citations = result.get("citations", [])
        req_memory_ids = question.get("required_memory_ids", [])
        trace_validity = 1.0 if bool(citations) else (1.0 if is_abs else 0.0)

        # Claim Support Precision
        if req_memory_ids and citations:
            supported = any(any(req_id in str(c) for req_id in req_memory_ids) for c in citations)
            claim_support = 1.0 if supported else 0.5
        elif is_abs:
            claim_support = 1.0
        else:
            claim_support = 0.7 if citations else 0.0

        # Ablation Sensitivity
        ablation_sensitivity = 0.0
        if is_abs:
            ablation_sensitivity = 1.0
        elif ablated_result:
            ablated_text = ablated_result.get("answer", "")
            # If removing citations altered or retracted the answer, ablation sensitivity is high
            if self._normalize(ablated_text) != norm_answer or "ablated" in ablated_text.lower():
                ablation_sensitivity = 1.0
            else:
                ablation_sensitivity = 0.0

        return {
            "question_id": question.get("id"),
            "category": category,
            "recall_accuracy": round(recall_acc, 3),
            "what_changed_score": round(what_changed_score, 3),
            "abstention_score": round(abstention_score, 3),
            "handoff_score": round(handoff_score, 3),
            "trace_validity": round(trace_validity, 3),
            "claim_support": round(claim_support, 3),
            "ablation_sensitivity": round(ablation_sensitivity, 3),
            "latency_ms": result.get("latency_ms", 0.0),
            "prompt_tokens": result.get("prompt_tokens", 0),
            "completion_tokens": result.get("completion_tokens", 0),
            "context_tokens": result.get("context_tokens", 0),
            "tool_calls": result.get("tool_calls", 0),
        }
