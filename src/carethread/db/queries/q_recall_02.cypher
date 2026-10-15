MATCH (p:Patient {id:$pid})-[:HAS_FACT]->(f:Fact)
WHERE f.kind = $kind
  AND f.recorded_at <= $asof AND (f.superseded_at IS NULL OR f.superseded_at > $asof)
  AND f.valid_from <= $asof AND (f.valid_to IS NULL OR f.valid_to > $asof)
RETURN f.id AS id, f.slot AS slot, f.display AS display, f.value AS value,
       f.recorded_at AS recorded_at, f.superseded_at AS superseded_at
ORDER BY f.slot
