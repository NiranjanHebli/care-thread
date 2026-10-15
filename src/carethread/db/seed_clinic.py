"""Seeds the shared clinic knowledge graph (`clinic_shared`) from YAML clinical datasets.

Populates:
- AllergenClass & AllergenCode (with CROSS_REACTS_WITH and MEMBER_OF)
- DrugClass & Drug (with IN_CLASS and INTERACTS_WITH)
- ConditionCode
- Guideline & Rule (with HAS_RULE, TARGETS, and RECOMMENDS, plus 384-dim Guideline embeddings)
- PlaybookTemplate
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from falkordb import FalkorDB, Graph

from carethread.db.client import get_db_client
from carethread.db.schema import bootstrap_clinic_graph
from carethread.errors import DatabaseError, ValidationError
from carethread.memory.embeddings import Embedder, get_embedder

logger = logging.getLogger(__name__)

DEFAULT_CLINIC_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "clinic"


@dataclass
class ClinicSeedResult:
    """Summary of clinic graph seeding execution."""

    graph_name: str = "clinic_shared"
    guidelines_count: int = 0
    rules_count: int = 0
    condition_codes_count: int = 0
    drugs_count: int = 0
    drug_classes_count: int = 0
    allergen_classes_count: int = 0
    allergen_codes_count: int = 0
    playbook_templates_count: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return len(self.errors) == 0 and self.total_nodes > 0


class ClinicSeeder:
    """Loads curated clinical knowledge YAML files and seeds `clinic_shared` in FalkorDB."""

    def __init__(
        self,
        client: FalkorDB | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._client = client or get_db_client()
        self._graph: Graph = self._client.select_graph("clinic_shared")
        self._embedder = embedder or get_embedder()

    def seed_from_directory(
        self,
        data_dir: Path | str,
        bootstrap_schema: bool = True,
    ) -> ClinicSeedResult:
        """Parses all clinical YAML files in directory and seeds the clinic_shared graph."""
        path = Path(data_dir).resolve()
        if not path.is_dir():
            raise ValidationError(
                f"Clinic data directory does not exist: {path}",
                code="CLINIC_DATA_DIR_NOT_FOUND",
            )

        if bootstrap_schema:
            bootstrap_clinic_graph(self._graph, self._client)

        logger.info(f"Seeding clinic_shared graph from {path}...")

        # 1. Load Allergen Classes and Codes
        allergen_file = path / "allergen_classes.yaml"
        if allergen_file.exists():
            self._seed_allergen_classes(allergen_file)

        # 2. Load Drug Formularies, Drug Classes, and Interactions
        formulary_file = path / "formularies.yaml"
        if formulary_file.exists():
            self._seed_formularies(formulary_file)

        # 3. Connect Cross-Reactivities between AllergenClass and DrugClass
        if allergen_file.exists():
            self._link_cross_reactivities(allergen_file)

        # 4. Load Clinical Practice Guidelines and Rules
        guideline_file = path / "guidelines.yaml"
        if guideline_file.exists():
            self._seed_guidelines(guideline_file)

        # 5. Load Playbook Templates
        playbook_file = path / "playbooks.yaml"
        if playbook_file.exists():
            self._seed_playbooks(playbook_file)

        # 6. Verify and compute graph summary
        return self._verify_and_summarize()

    def _seed_allergen_classes(self, file_path: Path) -> None:
        """Seeds AllergenClass and AllergenCode nodes with MEMBER_OF relationships."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        classes = data.get("allergen_classes", [])
        for ac in classes:
            self._graph.query(
                """
                MERGE (a:AllergenClass {id: $id})
                ON CREATE SET a.name = $name, a.description = $desc
                ON MATCH SET a.name = $name, a.description = $desc
                """,
                {
                    "id": ac["id"],
                    "name": ac["name"],
                    "desc": ac.get("description", ""),
                },
            )

            codes = ac.get("allergen_codes", [])
            for cd in codes:
                self._graph.query(
                    """
                    MERGE (c:AllergenCode {code: $code})
                    ON CREATE SET c.system = $system, c.display = $display
                    ON MATCH SET c.system = $system, c.display = $display
                    WITH c
                    MATCH (a:AllergenClass {id: $ac_id})
                    MERGE (c)-[:MEMBER_OF]->(a)
                    """,
                    {
                        "code": str(cd["code"]),
                        "system": cd.get("system", "SNOMED-CT"),
                        "display": cd.get("display", ""),
                        "ac_id": ac["id"],
                    },
                )

    def _seed_formularies(self, file_path: Path) -> None:
        """Seeds DrugClass, Drug nodes, and drug-drug interactions."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Drug classes
        drug_classes = data.get("drug_classes", [])
        for dc in drug_classes:
            self._graph.query(
                """
                MERGE (c:DrugClass {id: $id})
                ON CREATE SET c.name = $name, c.description = $desc
                ON MATCH SET c.name = $name, c.description = $desc
                """,
                {
                    "id": dc["id"],
                    "name": dc["name"],
                    "desc": dc.get("description", ""),
                },
            )

        # Drugs
        drugs = data.get("drugs", [])
        for d in drugs:
            self._graph.query(
                """
                MERGE (dr:Drug {code: $code})
                ON CREATE SET dr.name = $name,
                              dr.ingredient = $ingredient,
                              dr.formulary_tier = $tier,
                              dr.formulary_status = $status
                ON MATCH SET dr.name = $name,
                             dr.ingredient = $ingredient,
                             dr.formulary_tier = $tier,
                             dr.formulary_status = $status
                WITH dr
                MATCH (dc:DrugClass {id: $class_id})
                MERGE (dr)-[:IN_CLASS]->(dc)
                """,
                {
                    "code": str(d["code"]),
                    "name": d["name"],
                    "ingredient": d.get("ingredient", d["name"]),
                    "tier": int(d.get("formulary_tier", 1)),
                    "status": d.get("formulary_status", "preferred"),
                    "class_id": d["class_id"],
                },
            )

        # Drug-drug interactions
        interactions = data.get("drug_interactions", [])
        for inter in interactions:
            self._graph.query(
                """
                MATCH (da:Drug {code: $code_a}), (db:Drug {code: $code_b})
                MERGE (da)-[r:INTERACTS_WITH]->(db)
                ON CREATE SET r.severity = $severity, r.effect = $effect
                ON MATCH SET r.severity = $severity, r.effect = $effect
                """,
                {
                    "code_a": str(inter["drug_a_code"]),
                    "code_b": str(inter["drug_b_code"]),
                    "severity": inter.get("severity", "moderate"),
                    "effect": inter.get("effect", ""),
                },
            )

    def _link_cross_reactivities(self, file_path: Path) -> None:
        """Connects AllergenClass nodes to cross-reactive DrugClass nodes."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        classes = data.get("allergen_classes", [])
        for ac in classes:
            cross_reacts = ac.get("cross_reacts_with", [])
            for dc_id in cross_reacts:
                self._graph.query(
                    """
                    MATCH (a:AllergenClass {id: $ac_id}), (dc:DrugClass {id: $dc_id})
                    MERGE (a)-[:CROSS_REACTS_WITH]->(dc)
                    """,
                    {
                        "ac_id": ac["id"],
                        "dc_id": dc_id,
                    },
                )

    def _seed_guidelines(self, file_path: Path) -> None:
        """Seeds Guideline, ConditionCode, and Rule nodes with embeddings."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        guidelines = data.get("guidelines", [])
        for g in guidelines:
            text = g.get("text", g.get("title", ""))
            # Generate 384-dimensional dense vector embedding
            embedding = self._embedder.embed(text)

            # Condition code mapping
            cond_dict = g.get("condition_code", {})
            cond_code = str(cond_dict.get("code", ""))
            cond_system = cond_dict.get("system", "SNOMED-CT")
            cond_display = cond_dict.get("display", "")

            if cond_code:
                self._graph.query(
                    """
                    MERGE (cc:ConditionCode {code: $code})
                    ON CREATE SET cc.system = $system, cc.display = $display
                    ON MATCH SET cc.system = $system, cc.display = $display
                    """,
                    {
                        "code": cond_code,
                        "system": cond_system,
                        "display": cond_display,
                    },
                )

            # Guideline node with vector embedding
            self._graph.query(
                """
                MERGE (g:Guideline {id: $id})
                ON CREATE SET g.title = $title,
                              g.version = $version,
                              g.text = $text,
                              g.condition_code = $cond_code,
                              g.embedding = vecf32($embedding)
                ON MATCH SET g.title = $title,
                             g.version = $version,
                             g.text = $text,
                             g.condition_code = $cond_code,
                             g.embedding = vecf32($embedding)
                """,
                {
                    "id": g["id"],
                    "title": g["title"],
                    "version": str(g.get("version", "1.0")),
                    "text": text,
                    "cond_code": cond_code,
                    "embedding": embedding,
                },
            )

            # Link Guideline to ConditionCode
            if cond_code:
                self._graph.query(
                    """
                    MATCH (g:Guideline {id: $g_id}), (cc:ConditionCode {code: $code})
                    MERGE (g)-[:TARGETS]->(cc)
                    """,
                    {"g_id": g["id"], "code": cond_code},
                )

            # Link Guideline to recommended DrugClass
            recommends = g.get("recommends_classes", [])
            for dc_id in recommends:
                self._graph.query(
                    """
                    MATCH (g:Guideline {id: $g_id}), (dc:DrugClass {id: $dc_id})
                    MERGE (g)-[:RECOMMENDS]->(dc)
                    """,
                    {"g_id": g["id"], "dc_id": dc_id},
                )

            # Rules
            rules = g.get("rules", [])
            for r in rules:
                self._graph.query(
                    """
                    MERGE (rule:Rule {id: $id})
                    ON CREATE SET rule.kind = $kind, rule.text = $text, rule.severity = $severity
                    ON MATCH SET rule.kind = $kind, rule.text = $text, rule.severity = $severity
                    WITH rule
                    MATCH (g:Guideline {id: $g_id})
                    MERGE (g)-[:HAS_RULE]->(rule)
                    """,
                    {
                        "id": r["id"],
                        "kind": r.get("kind", "checklist"),
                        "text": r["text"],
                        "severity": r.get("severity", "moderate"),
                        "g_id": g["id"],
                    },
                )

    def _seed_playbooks(self, file_path: Path) -> None:
        """Seeds PlaybookTemplate nodes."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        templates = data.get("playbook_templates", [])
        for pt in templates:
            steps_json = json.dumps(pt.get("steps", []))
            self._graph.query(
                """
                MERGE (p:PlaybookTemplate {id: $id})
                ON CREATE SET p.name = $name,
                              p.episode_type = $episode_type,
                              p.steps = $steps
                ON MATCH SET p.name = $name,
                             p.episode_type = $episode_type,
                             p.steps = $steps
                """,
                {
                    "id": pt["id"],
                    "name": pt["name"],
                    "episode_type": pt.get("episode_type", "ambulatory"),
                    "steps": steps_json,
                },
            )

    def _verify_and_summarize(self) -> ClinicSeedResult:
        """Queries counts to verify graph contents and summarize."""
        q = """
        OPTIONAL MATCH (g:Guideline) WITH count(g) AS g_cnt
        OPTIONAL MATCH (r:Rule) WITH g_cnt, count(r) AS r_cnt
        OPTIONAL MATCH (cc:ConditionCode) WITH g_cnt, r_cnt, count(cc) AS cc_cnt
        OPTIONAL MATCH (d:Drug) WITH g_cnt, r_cnt, cc_cnt, count(d) AS d_cnt
        OPTIONAL MATCH (dc:DrugClass) WITH g_cnt, r_cnt, cc_cnt, d_cnt, count(dc) AS dc_cnt
        OPTIONAL MATCH (ac:AllergenClass)
        WITH g_cnt, r_cnt, cc_cnt, d_cnt, dc_cnt, count(ac) AS ac_cnt
        OPTIONAL MATCH (code:AllergenCode)
        WITH g_cnt, r_cnt, cc_cnt, d_cnt, dc_cnt, ac_cnt, count(code) AS code_cnt
        OPTIONAL MATCH (pt:PlaybookTemplate)
        WITH g_cnt, r_cnt, cc_cnt, d_cnt, dc_cnt, ac_cnt, code_cnt, count(pt) AS pt_cnt
        RETURN g_cnt, r_cnt, cc_cnt, d_cnt, dc_cnt, ac_cnt, code_cnt, pt_cnt
        """
        res = self._graph.query(q)
        row = res.result_set[0]

        # Total nodes and edges
        node_cnt_res = self._graph.query("MATCH (n) RETURN count(n)")
        total_nodes = node_cnt_res.result_set[0][0] if node_cnt_res.result_set else 0

        edge_cnt_res = self._graph.query("MATCH ()-[r]->() RETURN count(r)")
        total_edges = edge_cnt_res.result_set[0][0] if edge_cnt_res.result_set else 0

        result = ClinicSeedResult(
            graph_name=self._graph.name,
            guidelines_count=row[0],
            rules_count=row[1],
            condition_codes_count=row[2],
            drugs_count=row[3],
            drug_classes_count=row[4],
            allergen_classes_count=row[5],
            allergen_codes_count=row[6],
            playbook_templates_count=row[7],
            total_nodes=total_nodes,
            total_edges=total_edges,
        )

        if total_nodes == 0:
            err = "Clinic seeding resulted in 0 nodes."
            result.errors.append(err)
            raise DatabaseError(err, code="CLINIC_SEED_EMPTY")

        logger.info(
            f"Seeded clinic_shared: {total_nodes} nodes, {total_edges} edges "
            f"({result.guidelines_count} guidelines, {result.rules_count} rules, "
            f"{result.drugs_count} drugs, {result.drug_classes_count} drug classes, "
            f"{result.allergen_classes_count} allergen classes)."
        )

        return result


def seed_clinic(
    client: FalkorDB | None = None,
    clinic_data_dir: Path | str | None = None,
    embedder: Embedder | None = None,
    bootstrap_schema: bool = True,
) -> ClinicSeedResult:
    """Convenience functional interface to seed the clinic_shared graph."""
    target_dir = Path(clinic_data_dir or DEFAULT_CLINIC_DATA_DIR)
    seeder = ClinicSeeder(client=client, embedder=embedder)
    return seeder.seed_from_directory(target_dir, bootstrap_schema=bootstrap_schema)
