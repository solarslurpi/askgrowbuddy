"""Ask questions to LLM providers."""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List

import anthropic
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.ollama import Ollama

from src.logging_config import setup_logging
from src.token_tracking import OtherInfo, QueryResult, TokenInfo, TokenTracker

setup_logging()

logger = logging.getLogger(__name__)

class AskQuestion(ABC):
    """Abstract base class for asking questions to different LLM providers."""

    def __init__(self) -> None:
        """Initialize with a single token tracker and run_id"""
        self.token_tracker = TokenTracker(clear_db=False)
        current_datetime = datetime.now()
        self.run_id = current_datetime.strftime('%Y%m%d-%H%M%S')

    def ask(self, query: str) -> QueryResult:
        """Ask a question and return the response with metrics."""
        # Call the implementation-specific method
        result = self._ask_implementation(query)

        # Set the timestamp
        result["timestamp"] = datetime.now(tz=timezone.utc).isoformat()

        # Calculate cost if applicable
        result["cost"] = self._calculate_cost(result["token_info"])

        # Log the usage - THIS IS THE PROBLEM AREA
        self.token_tracker.log_usage(result)

        return result

    def ask_batch(self, queries: List[Dict[str, str]]) -> List[QueryResult]:
        """Ask multiple questions and return responses with metrics."""
        # Call the implementation-specific batch method
        batch = self._ask_implementation_batch(queries)

        return batch

    @abstractmethod
    def _ask_implementation(self, query: str) -> QueryResult:
        """Implementation-specific ask method.

        Returns:
            tuple: (answer, token_info, other_info)

        """

    @abstractmethod
    def _ask_implementation_batch(self, queries: List[Dict[str, str]]) -> List[QueryResult]:
        """Implementation-specific batch ask method."""
        pass

    @abstractmethod
    def _calculate_cost(self, token_info: TokenInfo) -> float:
        """Calculate cost based on the LLM provider's pricing."""
        pass

    @abstractmethod
    def _calculate_cost_batch(self, token_info: TokenInfo) -> float:
        """Calculate cost based on the LLM provider's pricing for batch processing."""
        pass


class AskQuestionClaude(AskQuestion):
    """Implementation for Claude models using Anthropic API."""

    def __init__(self, model_name: str='claude-3-5-sonnet-20241022'):
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        # Set the API key directly on the client's default headers
        super().__init__()

    def _calculate_cost(self, token_info: TokenInfo) -> float:
        """Calculate cost based on Claude's pricing."""
        # Claude-3 Sonnet pricing (as of March 2024)
        input_cost_per_1k = 0.003
        output_cost_per_1k = 0.015

        input_cost = (token_info["prompt_tokens"] / 1000) * input_cost_per_1k
        output_cost = (token_info["completion_tokens"] / 1000) * output_cost_per_1k

        return round(input_cost + output_cost, 2)

    def _calculate_cost_batch(self, token_info: TokenInfo) -> float:
        """Calculate cost based on Claude's pricing for batch processing."""
        input_cost_per_1k = 0.0015
        output_cost_per_1k = 0.0075
        input_cost = (token_info["prompt_tokens"] / 1000) * input_cost_per_1k
        output_cost = (token_info["completion_tokens"] / 1000) * output_cost_per_1k
        return round(input_cost + output_cost, 2)

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

    def _ask_implementation_batch(self, queries: List[Dict[str, str]]) -> List[QueryResult]:
        """Claude-specific implementation for batch questions."""
        try:
            requests = [
                Request(
                    custom_id=query["custom_id"],
                    params=MessageCreateParamsNonStreaming(
                        model=self.model_name,
                        max_tokens=1000,
                        temperature=0,
                        messages=[{"role": "user", "content": query["question"]}]
                    )
                )
                for query in queries
            ]

            batch = self.client.beta.messages.batches.create(
                requests=requests
            )
            return batch
        except Exception:
            logger.error("Error in AskQuestionClaude batch processing", exc_info=True)
            raise


class AskQuestionOllama(AskQuestion):
    """Implementation for local models using Ollama."""

    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name
        self.llm = Ollama(model=model_name, request_timeout=1000.0)
        super().__init__()

    def _calculate_cost(self, token_info: TokenInfo) -> float:
        """Ollama is free."""
        return 0.0
    def _ask_implementation(self, query: str) -> QueryResult:
        """Ollama-specific implementation for asking questions."""
        try:
            messages = [ChatMessage(role=MessageRole.USER, content=query)]
            response = self.llm.chat(messages)

            # Convert nanoseconds to seconds
            total_duration = response.raw.get("total_duration", 0.0) / 1e9
            load_duration = response.raw.get("load_duration", 0.0) / 1e9
            eval_duration = response.raw.get("eval_duration", 0.0) / 1e9

            return QueryResult(
                run_id=self.run_id,
                timestamp="",  # Will be set by base class
                cost=0,  # Local models have no cost
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
                    total_duration=total_duration,  # Now in seconds
                    load_duration=load_duration,    # Now in seconds
                    eval_duration=eval_duration,    # Now in seconds
                ),
            )
        except Exception:
            logger.error("Error in AskQuestionOllama", exc_info=True)
            raise

    def _ask_implementation_batch(self, queries: List[Dict[str, str]]) -> List[QueryResult]:
        """Ollama-specific implementation for batch questions."""
        try:
            # Create batch requests
            requests = [
                Request(
                    custom_id=query["custom_id"],
                    params=MessageCreateParamsNonStreaming(
                        model=self.model_name,
                        max_tokens=1000,
                        temperature=0,
                        messages=[{"role": "user", "content": query["question"]}]
                    )
                )
                for query in queries
            ]

            # Send batch request
            batch = self.client.beta.messages.batches.create(requests=requests)
            return batch
        except Exception:
            logger.error("Error in AskQuestionOllama batch processing", exc_info=True)
            raise

    def _calculate_cost_batch(self, token_info: TokenInfo) -> float:
        """Ollama is free."""
        return 0.0

def get_llm(model_name: str) -> AskQuestion:
    """Factory function to get the appropriate LLM implementation."""
    if model_name.startswith("claude"):
        return AskQuestionClaude(model_name)
    return AskQuestionOllama(model_name)
