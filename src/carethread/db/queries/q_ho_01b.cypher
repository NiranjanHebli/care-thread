MATCH (h:Handoff {id:$hid}) WHERE h.status = 'draft'
SET h.status = 'open', h.version = h.version + 1
RETURN h.id
