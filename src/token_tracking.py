import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, TypedDict

import pandas as pd

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

class TokenTracker:
    def __init__(self, *, clear_db: bool = False) -> None:
        """Initialize the token tracker.

        Args:
            clear_db: If True, clear the existing database
        """
        self.db_path = Path("data/token_usage.db")
        self.db_path.parent.mkdir(exist_ok=True)

        # Only try to delete if it exists and clear_db is True
        if clear_db and self.db_path.exists():
            try:
                self.db_path.unlink()
            except PermissionError:
                logger.warning("Could not clear token database - file is in use")

        self._init_db()

        # Initialize the usage history list
        self.usage_history = []
        self.total_tokens = 0
        self.total_cost = 0.0


    def _init_db(self):
        """Initialize SQLite database for token usage tracking."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    timestamp TEXT,
                    model TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    duration REAL
                )
            """)
            conn.commit()
        finally:
            conn.close()


    def calculate_cost(self, tokens, model_name):
        """Calculate cost based on token usage and model"""
        # Cost per 1K tokens (example rates)
        rates = {
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.002,
            "claude-3-5-sonnet": 0.003,
            "mistral": 0.0000,
            # Add other model rates as needed
        }

        rate = rates.get(model_name, 0.0000)  # Default to lowest rate if unknown
        return (tokens / 1000) * rate


    def log_usage(self, result):
        """Log token usage from an LLM response"""
        try:
            # Handle BetaMessageBatchIndividualResponse structure
            usage = {
                "prompt_tokens": result.result.message.usage.input_tokens,
                "completion_tokens": result.result.message.usage.output_tokens,
                "total_tokens": result.result.message.usage.input_tokens + result.result.message.usage.output_tokens,
                "model": result.result.message.model,
                "run_id": result.custom_id,
                "timestamp": datetime.now().isoformat()
            }

            usage["cost"] = self.calculate_cost(usage["total_tokens"], usage["model"])

            self.usage_history.append(usage)
            self.total_tokens += usage["total_tokens"]
            self.total_cost += usage["cost"]
            return usage

        except Exception as e:
            logger.warning(f"Failed to log token usage: {e}", exc_info=True)  # Added exc_info for better debugging
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "model": "unknown",
                "run_id": str(int(time.time())),
                "timestamp": datetime.now().isoformat(),
                "cost": 0.0
            }


    def get_usage_summary(self):
        """Get summary of token usage"""
        return {
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 4),
            "requests": len(self.usage_history),
            "average_tokens": round(self.total_tokens / len(self.usage_history)) if self.usage_history else 0
        }


    def get_usage_by_model(self):
        """Get token usage breakdown by model"""
        model_usage = {}
        for usage in self.usage_history:
            model = usage.get("model", "unknown")
            if model not in model_usage:
                model_usage[model] = {
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    "requests": 0
                }

            model_usage[model]["total_tokens"] += usage["total_tokens"]
            model_usage[model]["total_cost"] += usage.get("cost", 0.0)
            model_usage[model]["requests"] += 1

        return model_usage


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
