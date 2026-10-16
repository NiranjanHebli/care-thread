"""Model-neutral LLM client supporting mock, replay, and live execution modes.

Provides deterministic responses for offline clinical agent workflows, caching
for replay verification, and optional connectivity to OpenCode / OpenAI-compatible
inference servers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_MOCK_MODEL: str = "opencode/nemotron-3-ultra-free"
DEFAULT_CACHE_DIR: str = "fixtures/llm_cache"
DEFAULT_CONFIG_PATH: str = "config/llm.yaml"


@dataclass
class LLMResponse:
    """Structured response from LLM generation."""

    text: str
    structured: dict[str, Any] = field(default_factory=dict)
    model: str = DEFAULT_MOCK_MODEL
    usage: dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "structured": self.structured,
            "model": self.model,
            "usage": self.usage,
            "cached": self.cached,
        }


class LLMClient:
    """Provider-neutral LLM client supporting mock, replay, and live inference."""

    def __init__(
        self,
        mode: str | None = None,
        backend: str | None = None,
        base_url: str | None = None,
        cache_dir: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        self.mode = (mode or os.environ.get("CT_LLM_MODE", "mock")).strip().lower()
        self.backend = (backend or os.environ.get("CT_LLM_BACKEND", "opencode_serve")).strip()
        self.base_url = (
            base_url or os.environ.get("OPENCODE_BASE_URL", "http://localhost:4096")
        ).strip()
        self.cache_dir = Path(cache_dir or os.environ.get("CT_LLM_CACHE_DIR", DEFAULT_CACHE_DIR))
        self.config_path = Path(config_path or os.environ.get("CT_LLM_CONFIG", DEFAULT_CONFIG_PATH))
        self._role_models: dict[str, list[str]] = self._load_role_models()

    def _load_role_models(self) -> dict[str, list[str]]:
        """Loads configured model priorities by agent role from yaml config."""
        if not self.config_path.exists():
            return {}
        try:
            with open(self.config_path, "r", encoding="utf-8") as fp:
                data = yaml.safe_load(fp) or {}
                return data.get("roles", {})
        except Exception as e:
            logger.warning(f"Could not load LLM role models from {self.config_path}: {e}")
            return {}

    def get_model_for_role(self, role: str) -> str:
        """Returns the primary model identifier for the given clinical role."""
        models = self._role_models.get(role) or self._role_models.get("agent")
        if models and len(models) > 0:
            return models[0]
        return DEFAULT_MOCK_MODEL

    def _cache_key(self, prompt: str, system: str | None, role: str) -> str:
        """Computes a deterministic SHA256 cache key for the prompt and context."""
        content = f"{role}:::{system or ''}:::{prompt}".encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        role: str = "agent",
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Generates a clinical response via mock, replay cache, or live inference."""
        model_name = self.get_model_for_role(role)

        if self.mode == "replay":
            cached = self._read_cache(prompt, system, role)
            if cached is not None:
                cached.cached = True
                return cached
            # If not in cache during replay, fallback to mock generation and cache it
            resp = self._mock_generate(prompt, system, role, model_name)
            self._write_cache(prompt, system, role, resp)
            return resp

        if self.mode == "live":
            try:
                return self._live_generate(prompt, system, role, model_name, temperature)
            except Exception as e:
                logger.warning(f"Live LLM generation failed ({e}); falling back to mock response.")
                return self._mock_generate(prompt, system, role, model_name)

        # Default: deterministic mock mode
        return self._mock_generate(prompt, system, role, model_name)

    def _read_cache(self, prompt: str, system: str | None, role: str) -> LLMResponse | None:
        """Reads a previously recorded LLM response from disk cache."""
        key = self._cache_key(prompt, system, role)
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            return LLMResponse(
                text=data.get("text", ""),
                structured=data.get("structured", {}),
                model=data.get("model", DEFAULT_MOCK_MODEL),
                usage=data.get("usage", {}),
                cached=True,
            )
        except Exception as e:
            logger.debug(f"Cache read error for {key}: {e}")
            return None

    def _write_cache(
        self,
        prompt: str,
        system: str | None,
        role: str,
        response: LLMResponse,
    ) -> None:
        """Persists an LLM response to disk cache for reproducible replay."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            key = self._cache_key(prompt, system, role)
            cache_file = self.cache_dir / f"{key}.json"
            with open(cache_file, "w", encoding="utf-8") as fp:
                json.dump(response.to_dict(), fp, indent=2)
        except Exception as e:
            logger.debug(f"Cache write error for {key}: {e}")

    def _live_generate(
        self,
        prompt: str,
        system: str | None,
        role: str,
        model_name: str,
        temperature: float,
    ) -> LLMResponse:
        """Executes HTTP request to live inference server (OpenAI / OpenCode format)."""
        import httpx

        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 1024,
        }

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        msg_content = choice["message"]["content"]
        usage = data.get(
            "usage",
            {"prompt_tokens": 50, "completion_tokens": 50, "total_tokens": 100},
        )

        structured = self._extract_json_block(msg_content)
        return LLMResponse(
            text=msg_content,
            structured=structured,
            model=model_name,
            usage=usage,
            cached=False,
        )

    def _extract_json_block(self, text: str) -> dict[str, Any]:
        """Attempts to parse JSON object from markdown fenced blocks or raw text."""
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        candidate = match.group(1) if match else text.strip()
        try:
            return json.loads(candidate)
        except Exception:
            return {}

    def _mock_generate(
        self,
        prompt: str,
        system: str | None,
        role: str,
        model_name: str,
    ) -> LLMResponse:
        """Produces deterministic, medically realistic clinical completions."""
        p_lower = prompt.lower()

        # 1. Intake role: Extraction & observation
        if role in ("intake", "extract"):
            extracted_facts = []
            if "throat" in p_lower or "pharyngitis" in p_lower or "fever" in p_lower:
                extracted_facts.append(
                    {
                        "kind": "symptom",
                        "display": "Sore throat and fever",
                        "value": "acute_onset_severe",
                        "authority": "patient_reported",
                    }
                )
            elif "knee" in p_lower or "joint" in p_lower or "pain" in p_lower:
                extracted_facts.append(
                    {
                        "kind": "symptom",
                        "display": "Right knee pain",
                        "value": "moderate_activity_aggravated",
                        "authority": "patient_reported",
                    }
                )
            else:
                extracted_facts.append(
                    {
                        "kind": "symptom",
                        "display": "Reported clinical concern",
                        "value": "patient_disclosed",
                        "authority": "patient_reported",
                    }
                )

            structured = {
                "summary": "Patient presented with acute clinical symptoms requiring evaluation.",
                "facts": extracted_facts,
                "note": "Intake assessment recorded reported symptoms and vital signs.",
                "goal": "Evaluate clinical presentation and provide diagnostic workup",
                "questions": [
                    "Are symptoms acute or chronic?",
                    "Are there contraindications or drug allergies?",
                ],
            }
            text = (
                f"Intake Assessment:\n"
                f"- Reported: {extracted_facts[0]['display']}\n"
                f"- Recommended: Specialist clinical consultation."
            )
            return LLMResponse(
                text=text,
                structured=structured,
                model=model_name,
                usage={"prompt_tokens": 120, "completion_tokens": 65, "total_tokens": 185},
            )

        # 2. Specialist role: Assessment & prescription
        if role in ("specialist", "agent"):
            if "throat" in p_lower or "pharyngitis" in p_lower or "fever" in p_lower:
                diagnosis = "Acute Bacterial Pharyngitis"
                med_display = "Amoxicillin 500mg"
                med_code = "amox_500"
                dosage = "500mg PO TID x 10 days"
            elif "knee" in p_lower or "joint" in p_lower:
                diagnosis = "Knee Osteoarthritis Flare"
                med_display = "Meloxicam 15mg"
                med_code = "melox_15"
                dosage = "15mg PO daily"
            else:
                diagnosis = "Upper Respiratory Symptoms"
                med_display = "Amoxicillin 500mg"
                med_code = "amox_500"
                dosage = "500mg PO TID x 10 days"

            structured = {
                "diagnosis": diagnosis,
                "clinical_assessment": (
                    f"Clinical presentation is consistent with {diagnosis}. "
                    f"Patient is hemodynamically stable. Initiating targeted therapy."
                ),
                "proposed_medication": {
                    "kind": "medication",
                    "display": med_display,
                    "code": med_code,
                    "dosage": dosage,
                    "authority": "agent_inferred",
                },
                "pharmacy_goal": (
                    f"Screen proposed prescription for {med_display} against allergies"
                ),
                "questions": [
                    f"Is {med_display} safe given patient allergy and formulary history?",
                ],
            }
            text = (
                f"Specialist Assessment:\n"
                f"Diagnosis: {diagnosis}\n"
                f"Prescription Proposed: {med_display} ({dosage})\n"
                f"Plan: Delegated to Pharmacy Agent for allergy and cross-reactivity screening."
            )
            return LLMResponse(
                text=text,
                structured=structured,
                model=model_name,
                usage={"prompt_tokens": 160, "completion_tokens": 90, "total_tokens": 250},
            )

        # 3. Pharmacy role: Safety check & provenance claims
        if role in ("pharmacy", "safety"):
            has_penicillin_allergy = "penicillin" in p_lower or "beta-lactam" in p_lower
            is_amoxicillin = "amoxicillin" in p_lower or "amox" in p_lower

            if has_penicillin_allergy and is_amoxicillin:
                is_safe = False
                recommendation = "Azithromycin 500mg PO day 1, then 250mg PO days 2-5"
                warning = (
                    "CRITICAL CONTRAINDICATION: Patient has documented Penicillin allergy. "
                    "Amoxicillin belongs to penicillin class and has cross-reactivity. "
                    "Recommend switching to Azithromycin."
                )
                flag_kind = "allergy_cross_reactivity"
            else:
                is_safe = True
                recommendation = "Approved as prescribed. Instruct patient to take with meals."
                warning = "No contraindications identified against active profile."
                flag_kind = "none"

            structured = {
                "is_safe": is_safe,
                "flag_kind": flag_kind,
                "warning": warning,
                "recommendation": recommendation,
                "claims": [
                    {
                        "text": (
                            "Proposed Amoxicillin has cross-reactivity with "
                            "documented Penicillin allergy"
                        ),
                        "conflicts_with": "Penicillin",
                    }
                ]
                if not is_safe
                else [
                    {
                        "text": "Prescription verified safe against known allergies",
                        "conflicts_with": None,
                    }
                ],
            }
            text = (
                f"Pharmacy Review:\n"
                f"Safe: {is_safe}\n"
                f"Evaluation: {warning}\n"
                f"Recommendation: {recommendation}"
            )
            return LLMResponse(
                text=text,
                structured=structured,
                model=model_name,
                usage={"prompt_tokens": 210, "completion_tokens": 110, "total_tokens": 320},
            )

        # Generic default
        return LLMResponse(
            text=f"Processed clinical instruction for {role}.",
            structured={"status": "completed", "role": role},
            model=model_name,
            usage={"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        )
