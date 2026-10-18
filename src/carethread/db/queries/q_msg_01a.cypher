MATCH (s:Session {id:$sid})
WITH s, s.msg_count + 1 AS seq
SET s.msg_count = seq
CREATE (m:Message {id:$mid, session_id:$sid, seq:seq, role:$role, agent_id:$agent_id,
                   content:$content, ts:$now, embedding:vecf32($emb)})
CREATE (s)-[:HAS_MESSAGE]->(m)
RETURN m.seq AS seq
