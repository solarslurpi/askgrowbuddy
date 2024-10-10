
import re
# --->: Set up the local embedding model and LLM
# Set embedding model
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.embed_model = OllamaEmbedding(
    model_name='nomic-embed-text',
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)
# Choose your LLM...
Settings.llm = Ollama(model='mistral', request_timeout=1000.0)

class SimpleFaithfulness:
    def __init__(self, model_name):
        self.model_name = model_name

    def evaluate(self, question, answer, context):
        prompt = f"""
        Question: {question}
        Answer: {answer}
        Context: {context}

        Evaluate the faithfulness of the answer based on the given context. Consider the following:
        1. Does the answer contain information not present in the context?
        2. Does the answer contradict any information in the context?
        3. Is the answer a fair representation of the information in the context?

        Respond with:
        1. A score from 0 to 1, where 0 is completely unfaithful and 1 is completely faithful.
        2. A brief explanation of your scoring.
        3. Any hallucinations or discrepancies found, if any.

        Format your response exactly as follows:
        Score: [Your score here]
        Explanation: [Your explanation here]
        Hallucinations: [List any hallucinations or discrepancies, or 'None' if none found]
        """

        try:
            response = self.mo(prompt)
            return self._parse_response(response)
        except Exception as e:
            print(f"An error occurred during evaluation: {str(e)}")
            return {"faithfulness": 0, "explanation": "Error occurred", "hallucinations": "Unable to evaluate"}

    def _parse_response(self, response):
        score_match = re.search(r'Score:\s*([\d.]+)', response)
        explanation_match = re.search(r'Explanation:\s*(.+?)(?:\n|$)', response, re.DOTALL)
        hallucinations_match = re.search(r'Hallucinations:\s*(.+?)(?:\n|$)', response, re.DOTALL)

        score = float(score_match.group(1)) if score_match else 0
        explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided"
        hallucinations = hallucinations_match.group(1).strip() if hallucinations_match else "Unable to determine"

        return {
            "faithfulness": score,
            "explanation": explanation,
            "hallucinations": hallucinations
        }

# Usage example
ollama_llm = OllamaLLM(model="mistral")
simple_faithfulness = SimpleFaithfulness(ollama_llm)

def evaluate_single_response(question, answer, context):
    result = simple_faithfulness.evaluate(question, answer, context)
    print(f"Faithfulness score: {result['faithfulness']}")
    print(f"Explanation: {result['explanation']}")
    print(f"Hallucinations: {result['hallucinations']}")
    return result

# Example usage
question = "What is wollastonite?"
answer = "Wollastonite is a calcium silicate mineral used in agriculture for its liming capability and silicon content."
context = "Wollastonite is a calcium silicate mineral. It is used in agriculture for its liming capability, silicon content, and calcium content."

evaluate_single_response(question, answer, context)