
from src.ollama_stuff import ask_question

class ResultHandler:
    def __init__(self, ollama_model):
        self.ollama_model = ollama_model

    def generate_results(self, prompts):
        results = []
        for prompt in prompts:
            results.append(ask_question(prompt))
        return results
