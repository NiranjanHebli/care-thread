MATCH (p:Patient {id:$pid})-[:HAS_EPISODE]->(e:Episode)
WHERE e.start <= $now
RETURN e.id AS id, e.kind AS kind, e.start AS start, e.end AS end, e.summary AS summary, e.outcome AS outcome, e.episode_type AS episode_type, e.seq AS seq, e.source AS source
ORDER BY e.start DESC LIMIT $n

