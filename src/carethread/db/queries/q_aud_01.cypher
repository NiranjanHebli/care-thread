MATCH (p:Patient {id:$pid})
CREATE (e:AuditEvent {id:$eid, ts:$now, agent_id:$agent_id, role:$role, tool:$tool, op:$op,
                      outcome:$outcome, args_hash:$args_hash, read_ids:$read_ids, trace_id:$trace_id,
                      latency_ms:$latency_ms})
RETURN e.id
