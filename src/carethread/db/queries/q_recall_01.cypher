MATCH (p:Patient {id:$pid})-[:HAS_FACT]->(f:Fact)
WHERE f.status = 'active' AND f.kind IN $kinds
  AND f.valid_from <= $now AND (f.valid_to IS NULL OR f.valid_to > $now)
RETURN f.id AS id, f.kind AS kind, f.display AS display, f.value AS value,
       f.confidence AS confidence, f.last_confirmed_at AS last_confirmed_at,
       f.half_life_days AS half_life_days, f.authority AS authority, f.version AS version
ORDER BY f.kind, f.valid_from DESC
