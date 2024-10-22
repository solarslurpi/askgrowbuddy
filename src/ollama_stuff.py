from typing import TypedDict, Dict, Any
from llama_index.llms.ollama import Ollama
from llama_index.core.base.llms.types import ChatMessage, MessageRole

class TokenInfo(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class OtherInfo(TypedDict):
    model: str
    total_duration: float
    load_duration: float
    eval_duration: float

class QueryResult(TypedDict):
    query: str
    answer: str
    token_info: TokenInfo
    other_info: OtherInfo

# Function to run a query with the retriever and custom prompt
def ask_question(query: str, model_name='mistral'):
    # Initialize the Ollama LLM
    ollama_llm = Ollama(model=model_name)

    # Create the message for Ollama's chat method
    # Note: The query might already include context from a previous step
    messages = [ChatMessage(role=MessageRole.USER, content=query)]

    # Get the response from Ollama
    ollama_response = ollama_llm.chat(messages)

    return create_query_result(query, ollama_response)

def create_query_result(query: str, ollama_response: Any) -> QueryResult:
    if ollama_response is None:
        return QueryResult(
            query=query,
            answer="No response received",
            token_info=TokenInfo(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            other_info=OtherInfo(model="", total_duration=0.0, load_duration=0.0, eval_duration=0.0)
        )

    # Your existing code here
    token_info = TokenInfo(
        prompt_tokens=ollama_response.raw.get('prompt_eval_count', 0),
        completion_tokens=ollama_response.raw.get('eval_count', 0),
        total_tokens=ollama_response.raw.get('prompt_eval_count', 0) + ollama_response.raw.get('eval_count', 0)
    )

    other_info = OtherInfo(
        model=ollama_response.raw.get('model', ''),
        total_duration=ollama_response.raw.get('total_duration', 0.0),
        load_duration=ollama_response.raw.get('load_duration', 0.0),
        eval_duration=ollama_response.raw.get('eval_duration', 0.0)
    )

    return QueryResult(
        query=query,
        answer=ollama_response.message.content,
        token_info=token_info,
        other_info=other_info
    )