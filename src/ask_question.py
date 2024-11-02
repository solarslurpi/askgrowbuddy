"""Ask questions to LLM providers."""

from __future__ import annotations

import logging
import sqlite3
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TypedDict

import anthropic
import pandas as pd
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.ollama import Ollama

from src.logging_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)


class TokenInfo(TypedDict):
    """Type definition for token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OtherInfo(TypedDict):
    """Type definition for model execution information."""

    model: str
    eval_duration: float
    load_duration: float
    total_duration: float


class QueryResult(TypedDict):
    """Type definition for query results containing run information and metrics."""

    run_id: str
    timestamp: str
    cost: float | None
    query: str
    answer: str
    token_info: TokenInfo
    other_info: OtherInfo


class AskQuestion(ABC):
    """Abstract base class for asking questions to different LLM providers."""

    def __init__(self, *, clear_db: bool = False) -> None:
        """Initialize with database path and ensure directory exists.

        Args:
            clear_db (bool): If True, deletes all records from the database. Defaults to False.

        """
        self.db_path = Path("data/token_usage.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if clear_db and self.db_path.exists():
            # Delete the existing database file
            self.db_path.unlink()

        self._init_db()

    def _calculate_cost(self, token_info: TokenInfo) -> float | None:
        """Calculate cost based on token usage and model-specific rates.

        Override in implementations that have known costs.
        """
        return None

    def _init_db(self):
        """Initialize SQLite database for token usage tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS token_usage (
                    run_id TEXT,
                    model TEXT,
                    timestamp TEXT,
                    source TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    cost REAL,
                    eval_duration REAL,
                    load_duration REAL,
                    total_duration REAL,
                    query TEXT,
                    answer TEXT,
                    PRIMARY KEY (timestamp, model)
                )
            """)

    def log_usage(self, result: QueryResult):
        """Log query result to SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO token_usage (
                    run_id, model, timestamp, source,
                    prompt_tokens, completion_tokens, total_tokens,
                    cost, eval_duration, load_duration, total_duration,
                    query, answer
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    result.get("run_id", ""),
                    result["other_info"]["model"],
                    result["timestamp"],
                    result.get("source", ""),
                    result["token_info"]["prompt_tokens"],
                    result["token_info"]["completion_tokens"],
                    result["token_info"]["total_tokens"],
                    result["cost"],
                    result["other_info"]["eval_duration"],
                    result["other_info"]["load_duration"],
                    result["other_info"]["total_duration"],
                    result["query"],
                    result["answer"],
                ),
            )

    def get_detailed_history(self, run_id: Optional[str] = None) -> pd.DataFrame:
        """Retrieve detailed query history, optionally filtered by run_id."""
        query = """
        SELECT
            timestamp,
            model,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cost,
            source,
            run_id,
            query,
            answer,
            eval_duration
        FROM token_usage
        WHERE 1=1
        """
        params = []

        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)

    def get_usage_by_run(self) -> pd.DataFrame:
        """Get aggregated statistics grouped by run_id."""
        query = """
        SELECT
            run_id,
            MIN(timestamp) as start_time,
            MAX(timestamp) as end_time,
            COUNT(*) as total_queries,
            SUM(prompt_tokens) as total_prompt_tokens,
            SUM(completion_tokens) as total_completion_tokens,
            SUM(total_tokens) as total_tokens,
            ROUND(SUM(cost), 4) as total_cost,
            ROUND(AVG(eval_duration), 2) as avg_eval_duration
        FROM token_usage
        WHERE run_id != ''
        GROUP BY run_id
        ORDER BY start_time DESC
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)

    def _create_token_usage(self, result: QueryResult) -> QueryResult:
        """Create TokenUsage from QueryResult."""
        return QueryResult(
            timestamp=datetime.now(tz=datetime.timezone.utc).isoformat(),
            cost=self._calculate_cost(result["token_info"]),
            query_result=result,
        )

    def ask(self, query: str) -> QueryResult:
        """Ask a question and return the response with metrics."""
        # Call the implementation-specific method
        result = self._ask_implementation(query)

        # Set the timestamp
        result["timestamp"] = datetime.now(tz=timezone.utc).isoformat()

        # Calculate cost if applicable
        result["cost"] = self._calculate_cost(result["token_info"])

        # Log the usage
        self.log_usage(result)

        return result

    @abstractmethod
    def _ask_implementation(self, query: str) -> tuple[str, TokenInfo, OtherInfo]:
        """Implementation-specific ask method.

        Returns:
            tuple: (answer, token_info, other_info)

        """


class AskQuestionClaude(AskQuestion):
    """Implementation for Claude models using Anthropic API."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = anthropic.Anthropic()
        super().__init__()

    def _calculate_cost(self, token_info: TokenInfo) -> float:
        """Calculate cost based on Claude's pricing."""
        # Claude-3 Sonnet pricing (as of March 2024)
        input_cost_per_1k = 0.003
        output_cost_per_1k = 0.015

        input_cost = (token_info["prompt_tokens"] / 1000) * input_cost_per_1k
        output_cost = (token_info["completion_tokens"] / 1000) * output_cost_per_1k

        return round(input_cost + output_cost, 6)

    def _ask_implementation(self, query: str) -> QueryResult:
        """Claude-specific implementation for asking questions."""
        try:
            start_time = time.time()
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1000,
                temperature=0,
                messages=[{"role": "user", "content": query}],
            )
            duration = time.time() - start_time

            return QueryResult(
                timestamp="",  # Will be set by base class
                cost=None,  # Will be set by base class
                query=query,
                answer=response.content[0].text,
                token_info=TokenInfo(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens
                    + response.usage.output_tokens,
                ),
                other_info=OtherInfo(
                    model=self.model_name,
                    eval_duration=duration,
                    load_duration=0.0,  # Claude API doesn't provide this
                    total_duration=duration,
                ),
            )
        except Exception:
            logger.error("Error in AskQuestionClaude", exc_info=True)
            raise


class AskQuestionOllama(AskQuestion):
    """Implementation for local models using Ollama."""

    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name
        self.llm = Ollama(model=model_name, request_timeout=1000.0)
        super().__init__()

    def _ask_implementation(self, query: str) -> QueryResult:
        """Ollama-specific implementation for asking questions."""
        try:
            messages = [ChatMessage(role=MessageRole.USER, content=query)]
            response = self.llm.chat(messages)

            if response is None:
                return QueryResult(
                    timestamp="",  # Will be set by base class
                    cost=None,  # Will be set by base class
                    query=query,
                    answer="No response received",
                    token_info=TokenInfo(
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                    ),
                    other_info=OtherInfo(
                        model=self.model_name,
                        total_duration=0.0,
                        load_duration=0.0,
                        eval_duration=0.0,
                    ),
                )

            return QueryResult(
                timestamp="",  # Will be set by base class
                cost=None,  # Will be set by base class
                query=query,
                answer=response.message.content,
                token_info=TokenInfo(
                    prompt_tokens=response.raw.get("prompt_eval_count", 0),
                    completion_tokens=response.raw.get("eval_count", 0),
                    total_tokens=response.raw.get("prompt_eval_count", 0)
                    + response.raw.get("eval_count", 0),
                ),
                other_info=OtherInfo(
                    model=response.raw.get("model", ""),
                    total_duration=response.raw.get("total_duration", 0.0),
                    load_duration=response.raw.get("load_duration", 0.0),
                    eval_duration=response.raw.get("eval_duration", 0.0),
                ),
            )
        except Exception:
            logger.error("Error in AskQuestionOllama", exc_info=True)
            raise


def get_llm(model_name: str) -> AskQuestion:
    """Factory function to get the appropriate LLM implementation."""
    if model_name.startswith("claude"):
        return AskQuestionClaude(model_name)
    return AskQuestionOllama(model_name)
