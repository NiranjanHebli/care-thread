MATCH (e:Episode {id:$eid})-[:NEXT*1..3]->(n:Episode)
RETURN n.id AS id, n.start AS start
ORDER BY n.start
