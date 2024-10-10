import chromadb
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.core.embeddings.utils import resolve_embed_model
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever, VectorIndexAutoRetriever
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.core.storage.storage_context import StorageContext
from src.ingest_service import IngestService

# Initialize the ChromaDB client.  The database of collections is hardcoded to be in the same path as the code and named "vectorstore". Not very flexible.
db = chromadb.PersistentClient(path="vectorstore")

# Get a collection
collection = db.get_collection("soil_test_comments_better_embedder")
# The metadata is needed to know what embedder was used to create the embeddings.
embed_model_name = collection.metadata.get("embed_model", "all-minilm")
print(f"embed_model: {embed_model_name}")
print(f"Metadata: {collection.metadata}")

# I wanted to be able to pick and choose what collection to use, where each collection already had the embeddings in it (as well as documents and metadata).  I am using chromadb.  I like using chromadb's native apis when possible. I only want to us Llamaindex to try out more advanced rag techniques.
# If you have already computed embeddings and dumped them into an external vector store (e.g. Pinecone, Chroma)
# Source: https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_guide/
vector_store = ChromaVectorStore(chroma_collection=collection)
# Let llamaindex know what embedding model to use. Recall we got the name of the embedding model from the collection's metadata.
ollama_embedding = OllamaEmbedding(
    model_name=embed_model_name,
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)
# The default embedding model in llamaindex is openai. To unseat, we call resolve_embed_model with the embedding model we want to use as the default.
embed_model = resolve_embed_model(embed_model=ollama_embedding)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)


# result = llm.complete("hello world")
# print(result)
###
## Now embeddings

# Get the default query engine. This seemed to be the most basic way.  I find llamaindex documentation/code to
# This is for chat completions, not embeddings.
llm = Ollama(model='mistral', request_timeout=30.0)
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("What is the ideal pH for Cannabis plants?")
print(response)


# Retriever fetches the relevant docs based on the query.
# base_retriever = index.as_retriever(similarity_top_k=2)
print("******************/nSIMPLE RETRIEVER/n******************")
base_retriever = VectorIndexRetriever(index=index, similarity_top_k=2)
retrievals = base_retriever.retrieve("What is the ideal pH for Cannabis plants?")

for r in retrievals:

    print(f'*******\n{r}\n*******')
print("******************/AUTO MERGING RETRIEVER/n******************")
# https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/
storage_context = StorageContext.from_defaults(vector_store=vector_store)
retriever = AutoMergingRetriever(
    base_retriever, storage_context=storage_context,verbose=True)

retrievals = retriever.retrieve("What is the ideal pH for Cannabis plants?")

for r in retrievals:

    print(f'*******\n{r}\n*******')

# AUTORETRIEVAL from chromadb...
# https://docs.llamaindex.ai/en/stable/examples/retrievers/vectara_auto_retriever/


# recursive_retriever = RecursiveRetriever
# Define VectorStoreInfo with metadata filters
# vector_store_info = VectorStoreInfo(
#     content_info="Comments on the results of soil tests.",

# )

# auto_retriever = VectorIndexAutoRetriever(
#     index=index, vector_store_info=vector_store_info, similarity_top_k=2,llm=llm)
# retrievals = auto_retriever.retrieve("What is the ideal pH for Cannabis plants?")
# for r in retrievals:
#     print(f'*******\n{r}\n*******')