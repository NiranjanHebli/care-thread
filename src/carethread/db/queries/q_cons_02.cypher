MATCH (p:Patient {id:$pid})-[:HAS_FACT]->(f:Fact {status:'active'})
WHERE f.half_life_days IS NOT NULL AND f.stale = false
WITH f, f.confidence * pow(0.5, (($now - f.last_confirmed_at) / 86400.0) / f.half_life_days) AS eff
WHERE eff < $stale_threshold
SET f.stale = true
RETURN f.id AS id, eff
