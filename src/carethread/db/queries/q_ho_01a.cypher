MATCH (p:Patient {id:$pid}), (frm:Agent {id:$from_agent}), (s:Session {id:$session_id})
CREATE (h:Handoff {id:$hid, status:'draft', from_role:$from_role, to_role:$to_role, goal:$goal,
                   state_summary:$state, created_at:$now, ttl_at:$ttl_at,
                   clinician_review_required:$cr, version:1})
CREATE (p)-[:HAS_HANDOFF]->(h)
CREATE (h)-[:FROM]->(frm)
CREATE (h)-[:CREATED_IN]->(s)
WITH h
UNWIND $fact_ids AS fid
MATCH (f:Fact {id:fid})
CREATE (h)-[:CARRIES {rel:'supporting'}]->(f)
RETURN h.id AS id, count(f) AS carried_facts
