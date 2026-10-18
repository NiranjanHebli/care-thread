MATCH (p:Patient {id:$pid})-[:HAS_FACT]->(old:Fact {id:$old_id})
WHERE old.status = 'active' AND old.version = $expected_version
MATCH (ep:Episode {id:$episode_id})
SET old.status = 'superseded', old.superseded_at = $now,
    old.valid_to = coalesce(old.valid_to, $valid_from), old.version = old.version + 1
CREATE (new:Fact {id:$new_id, kind:old.kind, slot:old.slot, code:$code, system:$system, display:$display,
                  value:$value, status:'active', valid_from:$valid_from, valid_to:null,
                  recorded_at:$now, superseded_at:null, last_confirmed_at:$now, confidence:$confidence,
                  authority:$authority, evidence_count:1, stale:false, half_life_days:$half_life,
                  version:1, source_episode_id:$episode_id, asserted_by:$agent_id, op_id:$op_id})
CREATE (p)-[:HAS_FACT]->(new)
CREATE (new)-[:SUPERSEDES {reason:$reason, at:$now}]->(old)
CREATE (new)-[:ASSERTED_IN {message_id:$message_id}]->(ep)
RETURN new.id AS new_id, old.id AS old_id
