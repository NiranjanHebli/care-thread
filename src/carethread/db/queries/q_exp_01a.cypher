MATCH (ans:Answer {id:$aid})-[u:USED]->(n)
RETURN n.id AS id, labels(n)[0] AS label, u.role AS role, u.rank AS rank, u.score AS score,
       u.retrieval_method AS method, n.status AS status, n.confidence AS confidence,
       n.valid_from AS valid_from, n.valid_to AS valid_to, n.last_confirmed_at AS last_confirmed_at,
       n.display AS display, n.text AS text
ORDER BY u.rank
