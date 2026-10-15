MATCH (e:Episode {id:$eid})-[:NEXT*1..3]->(n:Episode)
RETURN n.id AS id, n.start AS start, n.summary AS summary, n.episode_type AS episode_type, n.seq AS seq
ORDER BY n.start ASC

