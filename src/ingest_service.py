import logging
import os
import re
from typing import List
import dotenv
import chromadb
import ollama

from langchain_community.document_loaders import ObsidianLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from llama_index.core import Document as LlamaIndexDocument
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core import PropertyGraphIndex
from llama_index.llms.ollama import Ollama
from src.ollama_embedding import OllamaEmbedding
from llama_index.core import Settings
from llama_index.core.schema import TextNode


from src.logging_config import setup_logging
from src.bm25_retriever_code import BM25Retriever

dotenv.load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

class IngestService:
    """
    A service for ingesting, processing, and storing text data within an Obsidian vault for efficient retrieval.

    This class handles the following operations:
    1. Loading documents from a directory or file (e.g., documents.py) using Langchain's ObsidianLoader.
    2. Chunking the loaded texts using Langchain's MarkdownTextSplitter.
    3. Vectorizing the text chunks using Ollama embeddings.
    4. Storing the vectorized embeddings in a ChromaDB instance.

    The class provides methods for each step of the process, allowing for flexible usage
    and easy integration with other components of a document processing pipeline.
    """
    def __init__(self,persisitent_directory:str="vectorstore"):
        self.persistent_directory = persisitent_directory
        self._client = None # Holds chromadb client when needed.
        logger.debug(f"Persistent directory for storing vectorized documents: {self.persistent_directory}")

        Settings.embed_model = OllamaEmbedding(
            model_name='nomic-embed-text',
            base_url="http://localhost:11434",
            ollama_additional_kwargs={"mirostat": 0},
        )
        # Choose your LLM...
        Settings.llm = Ollama(model='mistral', request_timeout=100.0)

    @property
    def client(self):
        # Instantiate when needed.
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.persistent_directory)
        return self._client

    def load_obsidian_notes(self, dir_or_list:str|list) -> list[LlamaIndexDocument]:
        '''Uses Langchain's Obsidian Loader to load in the text found within the directory (and sub directories) or a list of strings in Document objects. Returns a list of **Llamaindex** Documents because the RAG pipeline uses Llamaindex.'''
        logger.debug(f"Loading documents from: {dir_or_list}")
        docs = None
        if not dir_or_list:
            raise ValueError("A directory with text files or a list of strings must be passed in.")
        if isinstance(dir_or_list, list):
            docs = [
            LlamaIndexDocument(text=text, metadata={"source": f"string_{i}"})
            for i, text in enumerate(dir_or_list)
            ]
        else:
            if not os.path.exists(dir_or_list):
                raise ValueError(f"Directory does not exist: {dir_or_list}")
            loader = ObsidianLoader(path=dir_or_list, collect_metadata=True)
            langchain_docs = loader.load()
            docs = [
                LlamaIndexDocument(
                    text=self._remove_timestamp_blocks(doc.page_content),
                    metadata=doc.metadata
                )
                # LlamaIndexDocument.from_langchain_format(doc)
                for doc in langchain_docs
            ]

        return docs
    def _remove_timestamp_blocks(self, text: str) -> str:
        '''Many notes have a timestamp codeblock interspersed in the text to support YouTube playback. This function removes them.'''


        pattern = r'```timestamp(?:-url)?\n.*?\n```'
        return re.sub(pattern, '', text)

    def chunk_text(self, docs: list[LlamaIndexDocument]) -> list[TextNode]:
        headers_to_split_on = [
            ("#", "Header 1"),
            # ("##", "Header 2"),
            # ("###", "Header 3"),
        ]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on,strip_headers=False)
        nodes = []
        for doc in docs:
            note_text = doc.text
            note_metadata = doc.metadata
            langchain_nodes = splitter.split_text(note_text)

            # Convert Langchain nodes to LlamaIndex TextNodes
            nodes.extend([
                TextNode(
                    text=node.page_content,
                    metadata={**node.metadata, **note_metadata}
                )
                for node in langchain_nodes
            ])
        if not nodes:
            raise ValueError("No text chunks generated.")
        return nodes

    def build_vector_index(self, nodes: list[LlamaIndexDocument], embed_model_name: str = 'nomic-embed-text', collection_name: str = 'vectorstore', persist_dir: str = 'vectorstore'):
        '''Creates a vector index for the given document nodes.'''
        try:
            logger.info(f"Starting to build vector index with embedding model: {embed_model_name}")

            if not nodes:
                raise ValueError("No document nodes provided to build the vector index.")

            logger.debug(f"Number of documentnodes to process: {len(nodes)}")

            # Create Chromadb collection from the nodes
            our_collection = self.create_collection(docs=nodes, collection=collection_name, embedding_model_name= embed_model_name)
            logger.debug(f"Created collection '{collection_name}' with {our_collection.count()} document nodes")

            chroma_vector_store = ChromaVectorStore(chroma_collection=our_collection, persist_dir=persist_dir)
            logger.debug(f"Created ChromaVectorStore with persist directory: {persist_dir}")

            # Create a VectorStoreIndex using the ChromaVectorStore
            vector_index = VectorStoreIndex.from_vector_store(chroma_vector_store)
            logger.info(f"Successfully built vector index for collection '{collection_name}'")

            return vector_index

        except ValueError as ve:
            logger.error(f"ValueError in build_vector_index: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in build_vector_index: {str(e)}", exc_info=True)
            raise

    def get_vector_index(self, collection_name: str = 'vectorstore', persist_dir: str = 'vectorstore'):
        try:
            logger.info(f"Attempting to get vector index for collection '{collection_name}'")

            our_collection = self.get_collection(collection_name)
            logger.debug(f"Retrieved collection '{collection_name}'")

            chroma_vector_store = ChromaVectorStore(chroma_collection=our_collection, persist_dir=persist_dir)
            logger.debug(f"Retrieved ChromaVectorStore with persist directory: {persist_dir}")

            # Create a VectorStoreIndex using the ChromaVectorStore
            vector_index = VectorStoreIndex.from_vector_store(chroma_vector_store)
            logger.info(f"Successfully loaded vector index for collection '{collection_name}'")

            return vector_index
        except ValueError as ve:
            logger.error(f"ValueError in get_vector_index: {str(ve)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in get_vector_index: {str(e)}", exc_info=True)
            raise

    def build_knowledge_graph(self, documents: List[LlamaIndexDocument], user="neo4j", password=None, embed_model_name='nomic-embed-text', model_name: str = 'mistral') -> PropertyGraphIndex:
        try:
            logger.info(f"Starting to build knowledge graph with {len(documents)} documents")

            graph_store = self._get_graph_store(user, password)
            logger.debug("Graph store initialized")

            llm = Ollama(model=model_name, request_timeout=100.0)
            logger.debug(f"LLM initialized with model: {model_name}")

            embed_model = self._get_embed_model(embed_model_name)
            logger.debug(f"Embedding model initialized: {embed_model_name}")

            # Create a Knowledge Graph Index.
            kg_index = PropertyGraphIndex.from_documents(
                documents,
                embed_model=embed_model,
                llm=llm,
                property_graph_store=graph_store,
                show_progress=True,
            )
            logger.info("Knowledge graph index successfully created")

            return kg_index

        except ValueError as ve:
            logger.error(f"ValueError in build_knowledge_graph: {str(ve)}")
            raise
        except AttributeError as ae:
            logger.error(f"AttributeError in build_knowledge_graph: {str(ae)}. Ensure documents are of correct type.")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in build_knowledge_graph: {str(e)}", exc_info=True)
            raise

    def get_knowledge_graph(self, user="neo4j", password=None, embed_model_name='nomic-embed-text', model_name:str='mistral'):
        try:
            logger.info("Attempting to retrieve existing knowledge graph")

            graph_store = self._get_graph_store(user, password)
            logger.debug("Graph store initialized")

            llm = Ollama(model=model_name, request_timeout=100.0)
            logger.debug(f"LLM initialized with model: {model_name}")

            embed_model = self._get_embed_model(embed_model_name)
            logger.debug(f"Embedding model initialized: {embed_model_name}")

            kg_index = PropertyGraphIndex.from_existing(
                embed_model=embed_model,
                llm=llm,
                property_graph_store=graph_store,
                show_progress=True,
            )
            logger.info("Existing knowledge graph successfully retrieved")

            return kg_index

        except ValueError as ve:
            logger.error(f"ValueError in get_knowledge_graph: {str(ve)}")
            raise
        except ConnectionError as ce:
            logger.error(f"ConnectionError in get_knowledge_graph: {str(ce)}. Check your database connection.")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in get_knowledge_graph: {str(e)}", exc_info=True)
            raise

    def build_bm25_retriever(self, nodes: List, persist_dir: str = "bm25_index", similarity_top_k: int = 5) -> BM25Retriever:
        try:
            if not nodes:
                logger.error("The 'nodes' list is empty. Cannot build BM25 retriever without documents.", exc_info=True)
                raise ValueError("The 'nodes' list is empty. Please provide documents to build the retriever.")

            # Build the BM25 retriever from documents
            bm25_retriever = BM25Retriever(nodes=nodes, similarity_top_k=similarity_top_k)
            retriever = bm25_retriever.from_defaults(nodes=nodes, similarity_top_k=similarity_top_k)
            logger.debug("BM25 retriever successfully created from documents.")

            # Ensure the persistence directory exists
            os.makedirs(persist_dir, exist_ok=True)
            logger.debug(f"Persistence directory ensured at '{persist_dir}'.")

            # Persist the retriever to the specified directory
            retriever.persist(persist_dir)
            logger.debug(f"BM25 retriever successfully persisted to '{persist_dir}'.")

            return retriever

        except Exception:
            logger.error("An error occurred while building the BM25 retriever.", exc_info=True)
            raise

    def get_bm25_retriever(self, persist_dir: str = "bm25_index"):
        try:
            if not os.path.exists(persist_dir):
                logger.error(f"Persistence directory '{persist_dir}' does not exist.", exc_info=True)
                raise FileNotFoundError(f"BM25 retriever persistence directory '{persist_dir}' not found.")

            # Load the retriever from the persisted directory

            retriever = BM25Retriever.from_persist_dir(persist_dir)
            logger.info(f"BM25 retriever successfully loaded from '{persist_dir}'.")

            return retriever

        except FileNotFoundError as e:
            logger.error(f"BM25 retriever not found in '{persist_dir}'. You may need to build it first.", exc_info=True)
            raise e
        except Exception as e:
            logger.error("An error occurred while loading the BM25 retriever.", exc_info=True)
            raise e


    def _get_graph_store(self, user="neo4j", password=None, database:str='rag_database'):
        neo4j_uri = "bolt://localhost:7687"  # Update with your Neo4j instance details
        neo4j_user = user
        neo4j_password = password if password else os.getenv("NEO4J_PASSWORD") # type: ignore
        graph_store = Neo4jPropertyGraphStore(
            username=neo4j_user,
            password=neo4j_password, # type: ignore
            url=neo4j_uri,
            database=database,  # Use appropriate database name
        )
        return graph_store

    def _get_embed_model(self, embed_model_name:str):
        return OllamaEmbedding(
            model_name=embed_model_name,
            base_url="http://localhost:11434",
            ollama_additional_kwargs={"mirostat": 0},
        )

    def get_collection(self, collection:str) -> chromadb.Collection:
        '''Returns a collection object of a given name.'''
        try:
            return self.client.get_collection(name=collection)
        except ValueError as e:
            logger.error(f"Error getting collection '{collection}'",exc_info=True)
            raise e

    def create_collection(self,
                          docs:list[LlamaIndexDocument]=None, # type: ignore
                          collection: str=None,  # type: ignore
                          collection_metadata:dict=None, # type: ignore
                          embedding_model_name:str='all-minilm') -> chromadb.Collection:
        """
        Create a ChromaDB collection with document content and embeddings.

        Args:
            docs (list[LlamaIndexDocument]): Documents to store.
            collection (str): Name of the collection to create.
            collection_metadata (dict, optional): Metadata for the collection.
            embedding_model (str, optional): Model to use for generating embeddings. Defaults to 'all-minilm'.

        Returns:
            chromadb.Collection: The created ChromaDB collection.

        Raises:
            ValueError: If no documents are provided or no collection name is given.

        Note:
            If the collection already exists, it will be deleted and recreated.
        """
        logger.debug(f"Storing documents in ChromaDB collection: {collection}")
        if not docs:
            raise ValueError("No documents to store.")
        if not collection:
            raise ValueError("No collection name provided.")
        # First delete the collection if it already exists
        if any(name == collection for name, _ in self.collections_name_and_metadata()):
            self.delete_collection(collection)
        # Create a new collection
        collection = self.client.create_collection(name=collection, metadata=collection_metadata)
        # store each document
        for i, d in enumerate(docs):
            response = ollama.embeddings(model=embedding_model_name, prompt=d.text)
            embedding = response["embedding"]
            # Build the parameters dictionary
            params = {
                "ids": [str(i)],
                "embeddings": [embedding],
                "documents": [d.text]
            }
            # Conditionally add metadatas if it is not None and not empty
            if d.metadata:
                params["metadatas"] = [d.metadata]
            collection.add(**params)

        logger.info(f"Created collection '{collection}' with {len(docs)} documents.")
        return collection

    def collection_count(self,collection:str) -> int:
        '''Returns the number of documents in the collection.'''

        collection = self.client.get_collection(name=collection)
        return collection.count()

    # def add_documents(self, collection_name: str, docs: list[str], e_model: str = 'all-minilm'):
    #     collection = self.client.get_collection(name=collection)

    #     # Get the current count of documents in the collection
    #     current_count = collection.count()

    #     # Extract page content from each document
    #     doc_texts = [doc.page_content for doc in docs]
    #     metadatas = [doc.metadata for doc in docs]
    #     # Generate embeddings for each document
    #     embeddings = [self.generate_embedding(doc, e_model) for doc in doc_texts]


        # # Add new documents to the collection
        # collection.add(
        #     documents=doc_texts,
        #     ids=[f"doc_{i}" for i in range(current_count, current_count + len(docs))],
        #     metadatas=metadatas,
        #     embeddings=embeddings
        # )

        # print(f"Added {len(docs)} documents to collection '{collection_name}'. Total documents: {collection.count()}")

    def delete_collection(self, collection_name: str):
        try:
            self.client.delete_collection(name=collection_name)
            message = f"Collection '{collection_name}' has been deleted."
        except ValueError as e:
            logger.error(f"Error deleting collection '{collection_name}",exc_info=True)
            raise e
        logger.debug(message)
        return message

    def collections_name_and_metadata(self) -> list[str]:
        '''Returns a list of tuples where each tuple contains a collection name and its corresponding metadata.'''
        collections = self.client.list_collections()
        collection_names = [collection.name for collection in collections]
        collection_metadatas = [collection.metadata for collection in collections]
        return list(zip(collection_names, collection_metadatas))