MATCH (ans:Answer {id:$aid})-[r:REJECTED]->(n)
OPTIONAL MATCH (n)<-[:SUPERSEDES]-(newer:Fact)
RETURN n.id AS id, labels(n)[0] AS label, r.reason_code AS reason_code, r.reason AS reason,
       r.score AS score, newer.id AS superseded_by
