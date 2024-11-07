import json
import logging
import os
import re
import time
from typing import Generator, Iterator, Optional, Tuple

import spacy
from llama_index.core.schema import NodeWithScore, TextNode
from neo4j import GraphDatabase, Session
from sentence_transformers import SentenceTransformer, util

from src.ask_question import get_llm
from src.logging_config import setup_logging
from src.token_tracking import TokenTracker

setup_logging()
logger = logging.getLogger(__name__)

class BuildGraphIndex:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.token_tracker = TokenTracker()

    def _get_triplet_extraction_prompt(self, max_triplets: int = 3) -> str:
        return f"""Some text is provided below. Extract up to {max_triplets} key knowledge triplets in (subject, predicate, object) format.

    Important guidelines:
    - Only extract facts explicitly stated in the text
    - Combine numeric ranges into single values (e.g., "6.8-6.9" instead of separate numbers)
    - Use exact values and terms from the text (no synonyms or interpretations)
    - Each concept should appear in only one meaningful relationship
    - For cause-effect relationships, the cause must be the subject and the effect must be the object
    - Format predicates as UPPERCASE_KEYWORDS using only:
    * Letters A-Z
    * Numbers 0-9
    * Underscores _
    * NO spaces, apostrophes, or special characters
    * Examples: OPTIMAL_PH_IS, REQUIRES, CAUSES, INCREASES_BY

    Return the triplets in the following format:
    (subject1, predicate1, object1)
    (subject2, predicate2, object2)
    ...

    Examples:
    Text: The ideal pH for tomatoes is between 6.0 and 6.8.
    Triplets:
    (tomato_plants, OPTIMAL_PH_IS, 6.0-6.8)

    Text: Low phosphorus can cause slower plant growth.
    Triplets:
    (low_phosphorus, CAUSES, slower_plant_growth)

    Text: {{text}}
    Triplets:"""

    def _clear_neo4j_database(self, uri="bolt://localhost:7687",
                            auth=("neo4j", None),
                            database="soiltestknowledge"):
        """
        Delete all nodes and relationships from a Neo4j database.

        Args:
            uri (str): Neo4j connection URI
            auth (tuple): (username, password) tuple
            database (str): Name of the database to clear

        Returns:
            int: Number of nodes remaining after deletion
        """
        username, password = auth
        if password is None:
            password = os.getenv("NEO4J_PASSWORD")
            if not password:
                error_message = "NEO4J_PASSWORD environment variable not set"
                raise ValueError(error_message)
            auth = (username, password)

        driver = GraphDatabase.driver(uri, auth=auth)

        try:
            with driver.session(database=database) as session:
                # Delete all nodes and relationships
                session.run("MATCH (n) DETACH DELETE n")

                # Verify it's empty
                result = session.run("MATCH (n) RETURN count(n) as count")
                count = result.single()["count"]
                logger.debug(f"Deleted all nodes in database {database}. Verify 0 nodes remaining: {count}")
                return count

        finally:
            driver.close()

    def _parse_triplet(self, line: str) -> Optional[Tuple[str, str, str]]:
        """Extract a single triplet from a line of text."""
        # If the line is empty or doesn't have both ( and ), skip it
        if not (line and "(" in line and ")" in line):
            return None

        try:
            # From: "(subject, predicate, object)"
            # Get just: "subject, predicate, object"
            triplet_text = line.split("(")[1].split(")")[0]

            # Split by comma and remove extra spaces
            subject, predicate, obj = [t.strip() for t in triplet_text.split(",")]

            # Reject triplets where subject or object are purely numeric
            if subject.isdigit() or obj.isdigit():
                logger.warning(f"Rejecting triplet with numeric entity: {line}")
                return None
            return subject, predicate, obj
        except ValueError:  # If anything goes wrong, return None
            return None

    def _is_valid_neo4j_relationship_type(self,rel_type: str) -> bool:
        """
        Check if a string is a valid Neo4j relationship type.
        Rules:
        - Must start with a letter
        - Can only contain letters, numbers, and underscores
        - No spaces or special characters allowed
        """
        pattern = r"^[A-Za-z][A-Za-z0-9_]*$"
        return bool(re.match(pattern, rel_type))

    def _validate_triplets(self, triplets: list[Tuple[str, str, str]]) -> bool:
        """Check if all triplets have valid relationship types."""
        # Look at each triplet's predicate (the middle part)
        for _, predicate, _ in triplets:
            # Check if it follows Neo4j's rules (letters, numbers, underscores only)
            if not self._is_valid_neo4j_relationship_type(predicate.upper()):
                logger.error(f"Invalid predicate found: {predicate}")
                return False
        return True  # All predicates were good!



    def _extract_triplets_with_retries(self, text: str, max_triplets: int = 3,
                                model_name: str = None,
                                max_retries: int = 3,
                                retry_delay: int = 2) -> dict:
        cumulative_tokens = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        # The model name is caught by a previous call. But we'll check here too.
        if not model_name:
            error_message = "model_name is required"
            raise ValueError(error_message)


        for attempt in range(max_retries):
            try:
                prompt = self._get_triplet_extraction_prompt(max_triplets)
                formatted_prompt = prompt.format(text=text)

                # Get response with token info
                llm = get_llm(model_name)
                response = llm.ask(formatted_prompt)
                # Update token counts
                cumulative_tokens["prompt_tokens"] += response["token_info"]["prompt_tokens"]
                cumulative_tokens["completion_tokens"] += response["token_info"]["completion_tokens"]
                cumulative_tokens["total_tokens"] += response["token_info"]["total_tokens"]

                # Log token usage and other metrics
                logger.info(f"""
                    Attempt {attempt + 1} metrics:
                    - Tokens: {response['token_info']}
                    - Duration: {response['other_info']['eval_duration']:.2f}s
                    - Model: {response['other_info']['model']}
                """)

                # Parse and validate triplets in one pass
                triplets = []
                for line in response["answer"].strip().split("\n"):
                    triplet = self._parse_triplet(line)
                    if triplet and self._is_valid_neo4j_relationship_type(triplet[1].upper()):
                        logger.debug(f"Extracted triplet: {triplet}")
                        triplets.append(triplet)

                if triplets:  # If we have any valid triplets
                    return {
                        "triplets": triplets,
                        "response": response["answer"],
                        "token_usage": cumulative_tokens,
                        "metrics": response["other_info"]
                    }

                logger.warning(f"Attempt {attempt + 1} produced no valid triplets, retrying...")
                time.sleep(retry_delay)

            except Exception:
                logger.error(f"Attempt {attempt + 1} failed",exc_info=True)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        return {
            "triplets": None,
            "response": None,
            "token_usage": cumulative_tokens,
            "error": "Max retries exceeded"
        }



    def _determine_max_triplets(self, text: str) -> int:
        """Determine the number of triplets to extract based on SpaCy's dependency parsing.

        Args:
            text: Input text to analyze

        Returns:
            int: Number of triplets to extract (minimum 1, maximum 10)
        """
        doc = self.nlp(text)

        # Count relationships including passive voice and prepositional objects
        relationship_count = len([sent for sent in doc.sents
            if any(tok.dep_ in ("nsubj", "nsubjpass") for tok in sent)
            and any(tok.dep_ in ("dobj", "pobj") for tok in sent)])

        return min(relationship_count + 1, 10)  # Cap at 10 triplets

    def _insert_triplets_to_neo4j(self, node: TextNode, triplets: Iterator[Tuple[str, str, str]], session: Session) -> Generator[Tuple[str, str, str], None, None]:
        try:
        # Create TextNode
            session.run("""
                    MERGE (n:TextNode {
                        id: $id
                    })
                    ON CREATE SET
                        n.text = $text,
                        n.metadata = $metadata,
                        n.source = $source
                """,
                id=node.id_,
                text=node.text,
                metadata=json.dumps(node.metadata),
                source=node.metadata.get("source", "unknown")
                )

            for subject, predicate, obj in triplets:
                try:
                    # Convert predicate to valid relationship type
                    rel_type = predicate.upper().replace(" ", "_")

                    # Build query with relationship type directly in string
                    query = f"""
                        // Match our current TextNode
                        MATCH (n:TextNode {{id: $node_id}})

                        // Create or match our entities, tracking their first source
                        MERGE (s:Entity {{name: $subject}})
                        ON CREATE SET s.first_source = $source

                        MERGE (o:Entity {{name: $object}})
                        ON CREATE SET o.first_source = $source

                        // Create the relationship between entities
                        MERGE (s)-[r:{rel_type}]->(o)

                        // Connect entities back to their source TextNode
                        MERGE (s)-[:MENTIONED_IN]->(n)
                        MERGE (o)-[:MENTIONED_IN]->(n)
                    """

                    session.run(query,
                        node_id=node.id_,
                        subject=subject,
                        object=obj,
                        source=node.metadata.get("source", "unknown")
                    )

                    yield subject, predicate, obj

                except Exception:
                    logger.error(f"Error inserting triplet ({subject}, {predicate}, {obj}", exc_info=True)
                    continue

        except Exception:
            logger.error(f"Error creating TextNode {node.id_}",exc_info=True)
            raise

    def build_graph_index(self, text_nodes: list[TextNode]=None, database_name: str = None, llm_model_name: str = None) -> None:
        # Input is intentional because they have a major impact on the outcome and we sholdn't assume defaults.
        if not database_name or not llm_model_name or not text_nodes:
            error_message = "All parameters are required: database_name: {}, llm_model_name: {}, and text_nodes: {}".format(database_name, llm_model_name, text_nodes)
            raise ValueError(error_message)

        password = os.getenv("NEO4J_PASSWORD")
        if not password:
            error_message = "NEO4J_PASSWORD environment variable not set"
            raise ValueError(error_message)

        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", password))
        self._clear_neo4j_database(database=database_name)
        try:
            if not text_nodes:
                error_message = "No text nodes to process"
                raise ValueError(error_message)

            with driver.session(database=database_name) as session:
                logger.info(f"Total nodes to process: {len(text_nodes)}")

                for i, node in enumerate(text_nodes):
                    try:
                        logger.info(f"\nProcessing node {i+1}/{len(text_nodes)}")
                        logger.debug(f"Source file: {node.metadata.get('source', 'Untitled')}")
                        logger.debug(f"Node ID: {node.id_}")
                        logger.debug(f"Text preview: {node.text[:100]}...")

                        max_triplets = self._determine_max_triplets(node.text)
                        logger.debug(f"Max triplets: {max_triplets}")

                        result = self._extract_triplets_with_retries(
                            node.text,
                            max_triplets=max_triplets,
                            model_name=llm_model_name
                            )

                        if not result.get("triplets"):
                            logger.warning(f"No valid triplets extracted for node {i+1}")
                            continue

                        for subject, predicate, obj in self._insert_triplets_to_neo4j(
                            node=node,
                            triplets=result["triplets"],
                            session=session
                        ):
                            logger.debug(f"Inserted: ({subject}) --[{predicate}]--> ({obj})")

                    except Exception:
                        logger.error(f"Error processing node {i+1}",exc_info=True)
                        # Continue with next node instead of failing entire process
                        continue

        finally:
            driver.close()
            logger.info("Database connection closed")

class RetrieveGraphNodes:
    def __init__(self, embed_model_name: str = 'multi-qa-mpnet-base-cos-v1'):
        self.embed_model = SentenceTransformer(embed_model_name)

    def retrieve(self, query: str, database_name: str, k: int = 5) -> list[NodeWithScore]:
        """
        Retrieve nodes using sentence transformer embeddings.
        """
        driver = None
        try:
            driver = GraphDatabase.driver("bolt://localhost:7687",
                                    auth=("neo4j", os.getenv("NEO4J_PASSWORD")))

            with driver.session(database=database_name) as session:
                query_embedding = self.embed_model.encode(query, convert_to_tensor=True)

                text_matches = session.run("""
                    MATCH (e:Entity)-[:MENTIONED_IN]->(t:TextNode)
                    RETURN DISTINCT
                        t.text as text,
                        collect(e.name) as entities
                    LIMIT $k
                """, k=k)

                scored_entities = []
                for record in text_matches:
                    text_embedding = self.embed_model.encode(record["text"], convert_to_tensor=True)
                    score = float(util.cos_sim(query_embedding, text_embedding)[0][0])
                    scored_entities.append((record["text"], record["entities"], score))

                top_entities = sorted(scored_entities, key=lambda x: x[2], reverse=True)[:k]
                return self._get_nodes_with_scores(session, top_entities)

        except Exception as e:
            logger.error(f"Error retrieving nodes: {str(e)}", exc_info=True)
            return []

        finally:
            if driver:
                driver.close()
