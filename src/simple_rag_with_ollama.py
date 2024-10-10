# simple_rag_with_ollama.py

from typing import List
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,  # Singleton instance
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings  # Updated import
import os
import json
import chromadb
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
# Removed deprecated ServiceContext import
# from llama_index.core import  ServiceContext  # Removed ServiceContext import

# Set embedding model
ollama_embedding = OllamaEmbedding(
    model_name="all-minilm",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

# Configure Settings singleton
Settings.embed_model = ollama_embedding  # Updated line

# Optional: Configure other Settings attributes if needed
# Settings.llm = resolve_llm("default")  # Example
# Settings.callback_manager = CallbackManager()  # Example

# Chunk settings
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

# Read Docs
path = "data"
node_parser = SimpleDirectoryReader(input_dir=path, required_exts=['.pdf'])
documents = node_parser.load_data()
# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="test_db")
chroma_collection = chroma_client.get_or_create_collection("my_collection")

# Create ChromaVectorStore
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Text Cleaner Transformation
class TextCleaner(TransformComponent):
    def __call__(self, nodes, **kwargs) -> List[BaseNode]:
        for node in nodes:
            node.text = node.text.replace('\t', ' ')
            node.text = node.text.replace(' \n', ' ')
        return nodes

# Ingestion Pipeline
text_splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
pipeline = IngestionPipeline(
    transformations=[
        TextCleaner(),
        text_splitter,
    ],
    vector_store=vector_store,  # Use ChromaDB client for vector storage
)

# Run pipeline and get generated nodes from the process
nodes = pipeline.run(documents=documents)

# Create VectorStoreIndex without creating a new Settings instance
vector_store_index = VectorStoreIndex(nodes, settings=Settings)  # Updated line

retriever = vector_store_index.as_retriever(similarity_top_k=2)

# Test retriever
def show_context(context):
    for i, c in enumerate(context):
        print(f"Context {i+1}:")
        print(f"Score: {c.score}")
        print("-------------------")
        print(c.text)
        print("-------------------")
        print("\n")

test_query = "Why are fungi important?"
context = retriever.retrieve(test_query)
show_context(context)

# Evaluation


from deepeval.metrics import GEval, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

# Define evaluation metrics
correctness_metric = GEval(
    name="Correctness",
    criteria= "Determine whether the actual output is factually correct based on the expected output.",
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    verbose_mode=True
)
test_case = LLMTestCase(
    input="What categories are Mycorrhizal fungi divided into?",
    actual_output="Mycorrhizal fungi are divided into two categories: those whose root-like hyphae surround and occasionally penetrate root tissues (ectomycorrhizae) and those whose hyphae always enter the root cells (endomycorrhizae)."
)
correctness_metric.measure(test_case)
print(correctness_metric.score)
print(correctness_metric.reason)

answer_relevance_metric = AnswerRelevancyMetric(
    include_reason = True, verbose_mode=True
)
test_case = LLMTestCase(
    input="What mineral is Mycorrhizal fungi adept at extracting from the soil?",
    actual_output = "Mycorrhizal are able to solubilize (extract) phosphate in their immediate environment."
)
answer_relevance_metric.measure(test_case)
print(answer_relevance_metric.score)
print(answer_relevance_metric.reason)



# def evaluate_rag(query_engine, num_questions: int = 5) -> None:
#     q_a_file_name = "../data/q_a.json"
#     with open(q_a_file_name, "r", encoding="utf-8") as json_file:
#         q_a = json.load(json_file)

#     questions = [qa["question"] for qa in q_a][:num_questions]
#     ground_truth_answers = [qa["answer"] for qa in q_a][:num_questions]
#     generated_answers = []
#     retrieved_documents = []

#     for question in questions:
#         response = query_engine.query(question)
#         context = [doc.text for doc in response.source_nodes]
#         retrieved_documents.append(context)
#         generated_answers.append(response.response)

#     test_cases = create_deep_eval_test_cases(questions, ground_truth_answers, generated_answers, retrieved_documents)
#     evaluate(
#         test_cases=test_cases,
#         metrics=[correctness_metric, faithfulness_metric, relevance_metric]
#     )

# # Evaluate results
# query_engine = vector_store_index.as_query_engine(similarity_top_k=2)
# evaluate_rag(query_engine, num_questions=1)