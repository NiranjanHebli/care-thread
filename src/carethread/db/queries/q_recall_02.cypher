MATCH (p:Patient {id:$pid})-[:HAS_FACT]->(f:Fact)
WHERE ($kind IS NULL OR f.kind = $kind)
  AND f.recorded_at <= $asof AND (f.superseded_at IS NULL OR f.superseded_at > $asof)
  AND f.valid_from <= $asof AND (f.valid_to IS NULL OR f.valid_to > $asof)
RETURN f.id AS id, f.kind AS kind, f.slot AS slot, f.display AS display, f.value AS value,
       f.status AS status, f.valid_from AS valid_from, f.valid_to AS valid_to,
       f.recorded_at AS recorded_at, f.superseded_at AS superseded_at,
       f.confidence AS confidence, f.authority AS authority, f.version AS version
ORDER BY f.slot ASC

