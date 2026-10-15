MATCH (p:Patient {id:$pid})-[:HAS_EPISODE]->(e:Episode)
WHERE e.start <= $now
RETURN e.id AS id, e.kind AS kind, e.start AS start, e.summary AS summary, e.episode_type AS episode_type
ORDER BY e.start DESC LIMIT $n
