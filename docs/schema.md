# CareThread FalkorDB Graph Schema Specification

FalkorDB is the system of record for all patient and clinical graph operations.

## 1. Graphs Overview
- `patient_<id>`: Isolated tenant graph per patient (`^p_[0-9a-f]{12}$`).
- `clinic_shared`: Read-only shared clinic graph containing medical guidelines, drug classifications, formulary rules, and allergen cross-reactivities.
- `ct_control`: Control graph containing patient registries, deletion tokens, and audit tombstones.

---

## 2. Patient Graph Schema (`patient_<id>`)

### 2.1 Node Labels

| Label | Memory Tier | Properties |
|---|---|---|
| `Patient` | Anchor | `id, synthea_id, name, gender, birthdate, data_provenance='synthea', dirty, created_at` |
| `Episode` | Episodic | `id, kind ('synthea_encounter'\|'live_visit'\|'pharmacy_review'\|'specialist_review'), start, end, summary, summary_embedding, outcome, episode_type, seq, source ('synthea'\|'live')` |
| `Session` | Conversation | `id, started_at, ended_at, status ('open'\|'closed'), channel, agent_id, msg_count, summary` |
| `Message` | Conversation | `id, session_id, seq, role ('user'\|'agent'\|'tool'\|'system'), agent_id, content, ts, embedding, redacted_at` |
| `Fact` | Semantic | `id, kind ('condition'\|'medication'\|'allergy'\|'preference'\|'symptom'), slot, code, system, display, value, status ('active'\|'superseded'\|'disputed'\|'pending_review'\|'retracted'), valid_from, valid_to, recorded_at, superseded_at, last_confirmed_at, confidence, authority, evidence_count, stale, half_life_days, version, source_episode_id, asserted_by, op_id` |
| `FactCandidate` | Semantic (pre) | `id, kind, slot, code, display, value, episode_id, message_id, status ('pending'\|'promoted'\|'rejected'), created_at` |
| `Note` | Semantic/Episodic Text | `id, text, embedding, kind ('intake_note'\|'clinical_draft'\|'pharmacy_note'\|'canary'), author_agent_id, status, clinician_review_required, ts` |
| `Observation` | Episodic Evidence | `id, code, display, value, unit, ts, category` |
| `ClinicalProcedure` | Episodic Evidence | `id, code, display, start, end` |
| `Provider` | Context | `id, name, specialty` |
| `Organization` | Context | `id, name, city` |
| `Agent` | Actor | `id, role ('intake'\|'specialist'\|'pharmacy'\|'consolidator'\|'clinician'), name` |
| `Handoff` | Coordination | `id, status ('draft'\|'open'\|'accepted'\|'completed'\|'rejected'\|'expired'), from_role, to_role, goal, state_summary, created_at, ttl_at, accepted_at, completed_at, clinician_review_required, result_summary, version` |
| `OpenQuestion` | Coordination | `id, text, status ('open'\|'answered'), answer_text` |
| `Action` | Coordination/Procedural | `id, type, description, ts, status, episode_id, order` |
| `Playbook` | Procedural | `id, name, episode_type, status, version, success_count, failure_count, score, last_outcome_at, guideline_ref, embedding` |
| `PlaybookStep` | Procedural | `id, idx, action_type, text` |
| `Flag` | Safety | `id, type ('allergy_med_conflict'\|'drug_interaction'\|'duplicate_therapy'\|'conflicting_assertion'), severity ('high'\|'moderate'\|'low'), status ('open'\|'resolved'\|'dismissed'), rule_id, explanation, detected_at, resolved_by, resolved_at` |
| `Answer` | Provenance | `id, question, text, answer_type, clinician_review_required, overall_confidence, agent_id, session_id, created_at, retrieval_summary` |
| `AuditEvent` | Audit | `id, ts, agent_id, role, tool, op ('read'\|'write'\|'denied'), outcome, args_hash, read_ids, trace_id, latency_ms` |
| `ConsolidationRun` | Maintenance | `id, started_at, finished_at, trigger, stats` |

### 2.2 Relationship Types

| Type | From -> To | Meaning |
|---|---|---|
| `HAS_EPISODE` | `Patient -> Episode` | Ownership of visit/episode |
| `NEXT` | `Episode -> Episode` | Singly linked chronological chain |
| `HANDLED_BY` | `Episode / Session -> Agent` | Responsible agent |
| `WITH_PROVIDER` | `Episode -> Provider` | Synthea encounter provider |
| `AT_ORG` | `Episode -> Organization` | Synthea encounter organization |
| `PRODUCED` | `Episode -> Observation` | Evidence metrics produced in episode |
| `PERFORMED` | `Episode -> ClinicalProcedure` | Procedures performed in episode |
| `HAS_SESSION` | `Patient -> Session` | Patient interaction sessions |
| `OF_EPISODE` | `Session -> Episode` | Session bound to clinical episode |
| `HAS_MESSAGE` | `Session -> Message` | Ordered chat messages |
| `SENT_BY` | `Message -> Agent` | Sender entity |
| `HAS_FACT` | `Patient -> Fact` | Active & historical facts |
| `ASSERTED_IN` | `Fact / Note / FactCandidate -> Episode` | Source episode of fact assertion |
| `SUPERSEDES` | `Fact(new) -> Fact(old)` | Fact replacement with audit reason |
| `CONFLICTS_WITH`| `Fact -> Fact` | Open contradiction flag |
| `HAS_NOTE` | `Episode -> Note` | Clinical notes |
| `ABOUT` | `Note / Flag -> Fact / Episode` | Target entities of note/flag |
| `HAS_HANDOFF` | `Patient -> Handoff` | Active/completed handoffs |
| `CARRIES` | `Handoff -> Fact / Episode / Message / Note / Flag / Playbook` | State packet payload |
| `HAS_QUESTION`| `Handoff -> OpenQuestion` | Unanswered questions across agents |
| `HAS_ACTION` | `Handoff / Episode -> Action` | Actions taken or needed |
| `DERIVED_FROM` | `Playbook -> Episode` | Origin episodes for procedural playbook |
| `FOLLOWED` | `Episode -> Playbook` | Playbook usage tracking |
| `HAS_STEP` | `Playbook -> PlaybookStep` | Steps within procedural playbook |
| `USED` | `Answer -> any memory node` | Provenance citation supporting answer |
| `REJECTED` | `Answer -> any memory node` | Evaluated but rejected candidates |

---

## 3. Shared Clinic Graph (`clinic_shared`)

| Label | Key Properties | Connected Via |
|---|---|---|
| `Guideline` | `id, title, version, condition_code, text, embedding` | `-[:HAS_RULE]->(Rule)`, `-[:RECOMMENDS]->(DrugClass)` |
| `Rule` | `id, kind ('escalation'\|'checklist'\|'contraindication'), text, severity` | Guideline rules |
| `ConditionCode`| `code, system, display` | SNOMED / ICD mapping |
| `Drug` | `code, name, ingredient, formulary_tier, formulary_status` | `-[:IN_CLASS]->(DrugClass)`, `-[:INTERACTS_WITH]->(Drug)` |
| `DrugClass` | `id, name` | Class categorization |
| `AllergenClass`| `id, name` | `-[:CROSS_REACTS_WITH]->(DrugClass)` |
| `AllergenCode` | `code, system, display` | `-[:MEMBER_OF]->(AllergenClass)` |
| `PlaybookTemplate` | `id, name, episode_type, steps` | Standardized clinical care pathways |

---

## 4. Indexes & Constraints

1. **Unique Node Constraints & Range Indexes**:
   - `Patient.id`, `Episode.id`, `Fact.id`, `Handoff.id`, `Answer.id`, `Flag.id`, `Note.id`, `Session.id`, `Playbook.id`, `Agent.id`, `AuditEvent.id`.
2. **Compound Order Constraint**:
   - `Message(session_id, seq)` unique.
3. **Vector Indexes (384-dim, Cosine)**:
   - `Note.embedding`
   - `Message.embedding`
   - `Episode.summary_embedding`
   - `Playbook.embedding`
