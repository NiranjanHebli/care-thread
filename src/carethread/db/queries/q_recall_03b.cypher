MATCH (n:Note) WHERE n.id IN $ids AND n.status = 'active' AND n.kind <> 'canary'
OPTIONAL MATCH (ep:Episode)-[:HAS_NOTE]->(n)
OPTIONAL MATCH (n)-[:ABOUT]->(t)
RETURN n.id AS id, n.text AS text, n.author_agent_id AS author, ep.id AS episode_id,
       collect(DISTINCT t.id) AS about_ids
