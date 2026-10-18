MATCH (h:Handoff {id:$hid})
WHERE h.status = 'open' AND h.to_role = $role AND h.ttl_at > $now
MATCH (a:Agent {id:$agent_id}), (s:Session {id:$session_id})
SET h.status = 'accepted', h.accepted_at = $now, h.version = h.version + 1
CREATE (h)-[:TO]->(a)
CREATE (h)-[:ACCEPTED_IN]->(s)
RETURN h.id AS id
