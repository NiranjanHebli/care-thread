MATCH (s:Session {id:$sid})-[:HAS_MESSAGE]->(m:Message)
RETURN m.seq AS seq, m.role AS role, m.agent_id AS agent_id, m.ts AS ts, m.content AS content
ORDER BY m.seq ASC
