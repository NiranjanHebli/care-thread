MATCH (d:Drug {code:$drug_code})-[:IN_CLASS]->(dc:DrugClass)<-[:CROSS_REACTS_WITH]-(ac:AllergenClass)<-[:MEMBER_OF]-(code:AllergenCode)
RETURN dc.id AS drug_class_id, ac.id AS allergen_class_id, collect(code.code) AS allergen_codes
