"""Pydantic models for CareThread REST API requests and responses.

Provides validated request schemas for agent chat runs, Cytoscape graph
payloads, Why-Not counterfactual explainability, and safety flag resolution.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request payload for initiating an agent or workflow chat interaction."""

    patient_id: str = Field(..., description="Target patient identifier (p_...)")
    message: str = Field(..., description="Clinical message or chief complaint")
    session_id: str | None = Field(default=None, description="Existing session identifier")
    handoff_id: str | None = Field(default=None, description="Handoff ID for specialist/pharmacy")
    mock: bool = Field(default=False, description="Whether to force simulated LLM output")


class WhyNotRequest(BaseModel):
    """Request payload for querying why a candidate was not selected in an answer."""

    node_id: str | None = Field(default=None, description="Memory node identifier")
    code: str | None = Field(default=None, description="Drug or clinical condition code")
    name: str | None = Field(default=None, description="Human-readable entity name")
    label: str | None = Field(default=None, description="Graph label (e.g. Drug, Fact)")
    target: Any | None = Field(default=None, description="Arbitrary target descriptor")


class ResolveFlagRequest(BaseModel):
    """Request payload for resolving an active clinical safety flag."""

    resolved_by: str = Field(..., description="Clinician identifier resolving the flag")
    patient_id: str | None = Field(default=None, description="Patient identifier")


class AdvanceClockRequest(BaseModel):
    """Request payload for advancing the synthetic simulation clock."""

    seconds: int = Field(..., description="Number of seconds to advance")


class ToolInvokeRequest(BaseModel):
    """Request payload for internal tool execution harness."""

    params: dict[str, Any] = Field(default_factory=dict, description="Tool parameter payload")
    token: str = Field(default="tok_admin_01", description="Authentication token")
