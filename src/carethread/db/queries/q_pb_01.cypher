MATCH (pb:Playbook {id:$pbid})
WITH pb, pb.success_count + $s AS ns, pb.failure_count + $f AS nf
SET pb.success_count = ns, pb.failure_count = nf,
    pb.score = (ns + 1.0) / (ns + nf + 2.0), pb.last_outcome_at = $now,
    pb.status = CASE WHEN pb.status = 'candidate' AND ns >= 3 THEN 'active' ELSE pb.status END
RETURN pb.id AS id, ns AS success_count, nf AS failure_count, pb.score AS score, pb.status AS status
