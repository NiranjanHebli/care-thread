MATCH (cur:Fact {id:$fact_id})-[:SUPERSEDES*0..10]->(h:Fact)
RETURN h.id AS id, h.value AS value, h.status AS status, h.valid_from AS valid_from,
       h.recorded_at AS recorded_at, h.superseded_at AS superseded_at, h.asserted_by AS asserted_by
ORDER BY h.recorded_at DESC
