from llama_index.core import SimpleDirectoryReader, Settings

from llama_index.core.prompts import Prompt
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# Set embedding model
ollama_embedding = OllamaEmbedding(
    model_name="all-minilm",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

# Configure Ollama LLM
llm = Ollama(model="mistral", temperature=0, request_timeout=1000.0)
#
# Configure Settings singleton
Settings.embed_model = ollama_embedding
Settings.llm = llm
# Load documents
path = "data"
node_parser = SimpleDirectoryReader(input_dir=path, required_exts=['.pdf'])
documents = node_parser.load_data()

from llama_index.core.llama_dataset.generator import  RagDatasetGenerator
 # generate questions
data_generator = RagDatasetGenerator.from_documents(
    documents,
    show_progress=True,
    num_questions_per_chunk=4,
    text_question_template=Prompt(
        "A sample from the LlamaIndex documentation is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Using the documentation sample, carefully follow the instructions below:\n"
        "{query_str}"
    ),
    question_gen_query=(
        "You are an evaluator for a search pipeline. Your task is to write a single question "
        "using the provided documentation sample above to test the search pipeline. The question should "
        "reference specific names, functions, and terms. Restrict the question to the "
        "context information provided.\n"
        "Question: "
    )
)
generated_questions = data_generator.generate_questions_from_nodes()

# randomly pick 40 questions from each dataset
# import random
# generated_questions = random.sample(generated_questions, min(40, len(generated_questions)))


print(f"Generated {len(generated_questions)} questions.")

# Print all generated questions
print("\nGenerated Questions:")
for i, question in enumerate(generated_questions, 1):
    print(f"{i}. {question.strip()}")

# Remove or comment out the random sampling part
# print(random.sample(question_dataset, 5))