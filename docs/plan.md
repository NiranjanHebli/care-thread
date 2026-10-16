# CareThread Implementation Plan & Roadmap

## Phase 0: Foundations & Setup (Completed)
- [x] **Step 00**: Scaffold repo layout, pyproject.toml, Makefile, AGENTS.md, opencode.json, .env.example, initial documentation.
- [x] **Step 01**: Docker Compose setup for FalkorDB with persistent AOF, healthcheck, and connectivity verification.
- [x] **Step 02**: `TenantRouter`, `ClinicReader` (strictly read-only), and tenant registry in `ct_control`.
- [x] **Step 03**: Schema bootstrap (`db/schema.py`) creating range indexes, unique constraints, and 384-dim vector indexes.

## Phase 1: Data Ingestion & Clinical Knowledge
- [x] **Step 04**: Synthea CSV parser with column allowlist (PII stripping) and offline synthetic test fixtures (`tests/fixtures/synthea_csv`).
- [x] **Step 05**: Ingestion pipeline (`src/carethread/ingest/`) mapping Synthea entities to per-patient graphs (`patient_<id>`).
- [x] **Step 06**: Clinic graph seeding (`clinic_shared`) from YAML data (formularies, guidelines, allergen classes).

## Phase 2: Core Memory Layer
- [x] **Step 07**: Sessions, atomically sequenced messages, episodic chains (`NEXT`), injectable clock (`Clock.now()`), and ULID generators.
- [x] **Step 08**: Semantic memory (`memory/facts.py`): bitemporal facts, CAS supersession, as-of point-in-time queries.

- [x] **Step 09**: Hybrid recall (`memory/recall.py`): fastembed vector search + Cypher graph traversal, score ranking, rejected reason classification.
- [x] **Step 10**: Safety checks & contradiction detection (`memory/safety.py`): cross-reactivity flags, drug-drug conflicts.

## Phase 3: Coordination & MCP Interface
- [x] **Step 11**: Handoff state machine (`memory/handoff.py`): draft -> open -> accepted -> completed, CAS atomic accept for inter-agent delegation.
- [x] **Step 12**: Audit logging engine (`memory/audit.py`): synchronous writes, batched read flushing.
- [ ] **Step 13**: Procedural memory & background worker: consolidation runs, playbook promotions, decay marking.
- [ ] **Step 14**: Deletion & retention (`forget_request`, cryptographically secure tombstones).
- [ ] **Step 15**: Unified tool dispatcher and FastMCP server (`src/carethread/mcp/`).

## Phase 4: Agents, Provenance & UI
- [ ] **Step 16**: Clinical agents (`intake.py`, `specialist.py`, `pharmacy.py`) with `LLMClient` (mock/replay/live) executing the multi-agent interaction workflow (Intake -> Specialist -> Pharmacy as diagrammed in README).
- [ ] **Step 17**: Provenance engine (`commit_answer`, machine-verifiable traces, "Why Not?" analyzer).
- [ ] **Step 18**: FastAPI REST & SSE endpoints.
- [ ] **Step 19**: Cytoscape.js visual graph UI, trace highlighting, agent console, and handoff board.

## Phase 5: Verification & Benchmarking
- [ ] **Step 20**: Isolation test suite (`tests/isolation/test_iso_*.py`).
- [ ] **Step 21**: Load & concurrency test harness (`loadtest/`).
- [ ] **Step 22**: 3-Way Benchmark report (Stateless vs Vector-only vs CareThread Graph Memory).
