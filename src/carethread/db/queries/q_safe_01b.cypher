MATCH (p:Patient {id:$pid})-[:HAS_FACT]->(a:Fact {kind:'allergy', status:'active'})
WHERE a.code IN $allergen_codes
RETURN a.id AS allergy_fact_id, a.display AS display
