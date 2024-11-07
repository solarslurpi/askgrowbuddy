import json
import logging
import os
import pickle
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import anthropic
from llama_index.core.schema import TextNode
from neo4j import GraphDatabase

from src.ask_question import AskQuestionClaude
from src.knowledge_graph import BuildGraphIndex
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class BuildGraphIndexBatch(BuildGraphIndex):
    """Extends BuildGraphIndex to use Anthropic's batch API for faster processing.

    This class inherits all core functionality from BuildGraphIndex (triplet parsing,
    Neo4j operations, validation, etc.) and only implements batch-specific processing.
    It processes up to 10,000 nodes in parallel while maintaining all validation
    and safety checks from the base class.
    """

    def __init__(self):
        """Initialize with Anthropic client for batch processing.

        Args:
            anthropic_api_key: API key for Anthropic
        """
        super().__init__()
        self.llm = AskQuestionClaude(model_name="claude-3-5-sonnet-20241022")
        self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    def _extract_triplets_batch(self,
                              batch_nodes: List[TextNode],
                                ) -> Dict:
        try:
            # Create batch requests
            batch_requests = [
                {
                    "custom_id": node.id_,
                    "question": self._get_triplet_extraction_prompt(self._determine_max_triplets(node.text)).format(text=node.text)
                }
                for node in batch_nodes
            ]

            # Use the batch ask method
            batch = self.llm.ask_batch(batch_requests)
            return batch
        except Exception:
            logger.error("Error processing batch",exc_info=True)
            raise


    def create_batch(self,text_nodes: List[TextNode], batch_size: int = 10000) -> dict:
        if not text_nodes:
            raise ValueError("No text nodes to process")

        if batch_size > 10000:
            raise ValueError("Batch size cannot exceed 10000 requests")

        logger.info(f"Starting LLM batch processing of {len(text_nodes)} nodes")

        try:
            # Process in batches
            for i in range(0, len(text_nodes), batch_size):
                batch_nodes = text_nodes[i:i + batch_size]
                logger.info(f"\nProcessing batch {i//batch_size + 1}/{(len(text_nodes) + batch_size - 1)//batch_size}")
                logger.info(f"Nodes {i+1}-{i+len(batch_nodes)} of {len(text_nodes)}")

                # Process batch
                batch_ticket = self._extract_triplets_batch(
                    batch_nodes=batch_nodes
                )
                self.save_batch_id(batch_ticket.id)
                return batch_ticket
        except Exception:
            logger.error("Error processing batch",exc_info=True)
            raise
    def save_batch_id(self, batch_id: str, filepath: str = "batch_status.json") -> None:
        """Save the batch ID to a JSON file"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"batch_id": batch_id}, f)

    def load_batch_id(self, filepath: str = "batch_status.json") -> str:
        """Load the batch ID from a JSON file"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data["batch_id"]
        except FileNotFoundError:
            return None

    def check_batch_status(self, batch_id: str) -> dict[str, Any]:
        try:
            batch = self.client.beta.messages.batches.retrieve(batch_id)

            # Convert batch object to dict for JSON serialization
            batch_dict = {
                "id": batch.id,
                "type": batch.type,
                "processing_status": batch.processing_status,
                "request_counts": {
                "processing": batch.request_counts.processing,
                "succeeded": batch.request_counts.succeeded,
                "errored": batch.request_counts.errored,
                "canceled": batch.request_counts.canceled,
                "expired": batch.request_counts.expired
                },
                "created_at": self._convert_to_pst(batch.created_at),
                "ended_at": self._convert_to_pst(batch.ended_at),
                "expires_at": self._convert_to_pst(batch.expires_at),
                "archived_at": self._convert_to_pst(batch.archived_at),
                "cancel_initiated_at": self._convert_to_pst(batch.cancel_initiated_at),
                "results_url": batch.results_url
            }

            # Calculate and add time remaining
            if batch.expires_at:
                time_remaining = batch.expires_at - datetime.now(timezone.utc)
                hours = int(time_remaining.total_seconds() // 3600)
                minutes = int((time_remaining.total_seconds() % 3600) // 60)
                seconds = int(time_remaining.total_seconds() % 60)
                batch_dict["time_remaining"] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            logger.info(json.dumps(batch_dict, indent=2))
            return batch_dict
        except anthropic.NotFoundError:
            logger.error("Batch ID not found",exc_info=True)
            raise

    def _convert_to_pst(self, utc_time) -> str:
        """Convert UTC datetime to PST string"""
        if not utc_time:
            return None
        pst = utc_time.astimezone(timezone(timedelta(hours=-8)))
        return pst.strftime('%Y-%m-%d %I:%M:%S %p PST')

    def list_batches(self) -> list[dict[str, Any]]:
        return self.client.beta.messages.batches.list()

    def process_batch_results(self, text_nodes: List[TextNode], results_decoder, database_name: str) -> None:
        """Process batch results from Claude's JSONL decoder"""

        if not database_name:
            raise ValueError("database_name is required")

        password = os.getenv("NEO4J_PASSWORD")
        if not password:
            raise ValueError("NEO4J_PASSWORD environment variable not set")

        driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", password))
        self._clear_neo4j_database(database=database_name)

        try:
            with driver.session(database=database_name) as session:
                for i, response in enumerate(results_decoder):
                    try:
                        logger.info(f"\nProcessing response {response.custom_id}")

                        # Use TokenTracker instead of SQLite
                        self.token_tracker.log_usage(response)

                        # Parse triplets from response content
                        content = response.result.message.content[0].text
                        triplets = []
                        for line in content.strip().split('\n'):
                            triplet = self._parse_triplet(line)
                            if triplet:
                                triplets.append(triplet)

                        if not triplets:
                            logger.warning(f"No valid triplets extracted for response {i}")
                            continue

                        # Insert triplets into Neo4j
                        for subject, predicate, obj in self._insert_triplets_to_neo4j(
                            node=text_nodes[i],
                            triplets=triplets,
                            session=session
                        ):
                            logger.debug(f"Inserted: ({subject}) --[{predicate}]--> ({obj})")

                    except Exception:
                        logger.error("Error processing response.", exc_info=True)
                        continue

        finally:
            driver.close()
            logger.info("Database connection closed")

    def save_batch_results(self, results, filepath="batch_results.pkl"):
        """Save batch results using pickle"""
        results_list = list(results)  # Convert iterator to list
        with open(filepath, 'wb') as f:
            pickle.dump(results_list, f)
        logger.info(f"Saved {len(results_list)} batch results to {filepath}")

    def load_batch_results(self, filepath="batch_results.pkl"):
        """Load batch results from pickle file"""
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        logger.info(f"Loaded {len(results)} batch results from {filepath}")
        return results
