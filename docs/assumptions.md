# CareThread Assumptions and Verified Findings

This document tracks technical assumptions (A1-A14) and verified findings.

| ID | Assumption | Status | Notes |
|---|---|---|---|
| A1 | Synthea CSV export is opt-in | Verified | `tests/fixtures/synthea_csv` fixture will be committed for instant offline testing without Java requirement. |
| A2 | FalkorDB vector index syntax and cosine similarity | To Verify | `CREATE VECTOR INDEX FOR (n:Note) ON (n.embedding) OPTIONS {dimension:384, similarityFunction:'cosine'}`. Need to confirm score direction (distance vs similarity). |
| A3 | FalkorDB Docker configuration | Verified | Verified with `falkordb/falkordb:latest` (v4.22.1); AOF enabled (`appendonly: yes`, `appendfsync: everysec`); persistent volume mapped to `/var/lib/falkordb/data`. |
| A4 | Unique constraint syntax | To Verify | Range index must be created before `GRAPH.CONSTRAINT CREATE ... UNIQUE NODE`. Fallback to app-level checks in atomic statements if needed. |
| A5 | Cypher features (`UNWIND`, `coalesce`, `CASE`, variable hops) | Standard openCypher | Will be verified in unit tests against live container in Step 01-03. |
| A6 | Cypher transaction atomicity | Native | Each query is atomic in FalkorDB. Multi-graph operations coordinated via application logic. |
| A7 | Redis ACL for graph isolation | P2 Stretch | Standard application-level `TenantRouter` provides primary boundary. |
| A8 | Multi-tenant graph scaling | Architectural | FalkorDB handles hundreds of graphs with minimal memory overhead per empty graph. |
| A9 | Embeddings caching | Local ONNX | `fastembed` with `bge-small-en-v1.5` (384 dimensions) for local execution without API dependency. Deterministic hash embedder for unit tests. |
| A10 | FastMCP Python SDK | Installed | Version 2.3.0 installed and verified compatible with python 3.12. |
| A11 | OpenCode CLI & model availability | To Verify | OpenCode CLI present at `/Users/niranjan/.opencode/bin/opencode`. Free models will be tested in Step 00/01. |
| A12 | Performance targets | Target | Read p95 <= 120ms, write p95 <= 150ms under N=24, M=100 load. |
| A13 | Clinic rules nature | Non-clinical | Mock rules authored for demo and testing only; banner displayed at all times. |
| A14 | Model provider neutrality | Handled | Abstracted behind `LLMClient` with `mock`, `replay`, and `live` modes. |
