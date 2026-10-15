MATCH (a)-[r]->(b)
WHERE a.id IN $used_ids AND b.id IN $used_ids AND NOT type(r) IN ['USED','REJECTED','WROTE']
RETURN a.id AS from_id, type(r) AS type, b.id AS to_id
