MATCH (p:Patient {id:$pid})-[:HAS_CANDIDATE]->(c:FactCandidate {status:'pending'})
WITH c.kind AS kind, c.slot AS slot, collect(c.id) AS cids, count(DISTINCT c.episode_id) AS n_eps
WHERE n_eps >= 2
RETURN kind, slot, n_eps, cids
