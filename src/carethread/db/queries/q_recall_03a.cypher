CALL db.idx.vector.queryNodes('Note', 'embedding', $k, vecf32($qvec)) YIELD node, score
RETURN node.id AS id, score
