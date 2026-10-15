MATCH (h:Handoff {id:$hid})
OPTIONAL MATCH (h)-[c:CARRIES]->(m)
OPTIONAL MATCH (h)-[:HAS_QUESTION]->(q:OpenQuestion)
OPTIONAL MATCH (h)-[:HAS_ACTION]->(a:Action)
RETURN h,
       collect(DISTINCT {id:m.id, labels:labels(m), rel:c.rel, version:m.version}) AS memory,
       collect(DISTINCT q) AS questions, collect(DISTINCT a) AS actions
