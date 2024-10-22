import logging
import os
from typing import Any, Tuple
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.schema import NodeWithScore
from src.ingest_service import IngestService
from src.logging_config import setup_logging
from src.soil_test_processors import M3Processor, SPProcessor
from src.soil_test_models import M3Report, SPReport
from src.hybrid_retriever import HybridRetriever
from src.guidance import instruction, properties
from src.soil_test_models import soil_report_instance

setup_logging()
logger = logging.getLogger(__name__)


class SoilTestAnalyst:
    def __init__(self,m3_pdf_path:str, sp_pdf_path:str, model_name:str='mistral') -> None:
        try:
            m3_report, sp_report = self.load_reports(m3_pdf_path, sp_pdf_path)
            soil_report_instance.set_reports(m3_report, sp_report)
        except (FileNotFoundError, ValueError):
            raise
        except Exception:
            raise

        self.ingest_service = IngestService()
        self.setup_ollama()
        self.retriever = self.load_retriever()
        self.model_name = model_name
    def setup_ollama(self):
        Settings.embed_model = OllamaEmbedding(
            model_name='nomic-embed-text',
            base_url="http://localhost:11434",
            ollama_additional_kwargs={"mirostat": 0},
        )
        # Choose your LLM...
        Settings.llm = Ollama(model='mistral', request_timeout=1000.0)
    @classmethod
    def load_reports(cls, m3_pdf_filename: str, sp_pdf_filename: str) -> tuple[M3Report, SPReport]:
        '''
        Load Logan Labs formatted soil test reports from PDF files.

        Args:
            m3_pdf_filename (str): Path to the Mehlich-3 PDF report file.
            sp_pdf_filename (str): Path to the Saturated Paste PDF report file.

        Returns:
            tuple[M3Report, SPReport]: Processed M3 and SP reports.

        Raises:
            FileNotFoundError: If either of the PDF files doesn't exist.
            ValueError: If there's an issue processing the PDF files.
            Exception: For any other unexpected errors.
        '''
        logger.info("Starting to load soil test reports")
        logger.debug(f"M3 PDF filename: {m3_pdf_filename}")
        logger.debug(f"SP PDF filename: {sp_pdf_filename}")

        try:
            # Check if files exist
            if not os.path.exists(m3_pdf_filename):
                raise FileNotFoundError(f"M3 PDF file not found: {m3_pdf_filename}")
            if not os.path.exists(sp_pdf_filename):
                raise FileNotFoundError(f"SP PDF file not found: {sp_pdf_filename}")

            # Process M3 report
            logger.info("Processing Mehlich-3 report")
            m3 = M3Processor()
            m3_report = m3.process_pdf(m3_pdf_filename)
            logger.debug("Mehlich-3 report processed successfully")

            # Process SP report
            logger.info("Processing Saturated Paste report")
            sp = SPProcessor()
            sp_report = sp.process_pdf(sp_pdf_filename)
            logger.debug("Saturated Paste report processed successfully")

            logger.info("Soil test reports loaded successfully")
            return m3_report, sp_report

        except FileNotFoundError as e:
            logger.error(f"File not found error: {str(e)}")
            raise

        except ValueError as e:
            logger.error(f"Error processing PDF files: {str(e)}")
            raise ValueError(f"Error processing soil test reports: {str(e)}")

        except Exception as e:
            logger.error(f"Unexpected error in load_soil_test_reports: {str(e)}", exc_info=True)
            raise Exception(f"An unexpected error occurred while loading soil test reports: {str(e)}")

    def load_retriever(self, vector_similarity_top_k: int = 5, kg_similarity_top_k: int = 5, cohere_rerank_top_n: int = 5) -> HybridRetriever:
        """
        Load and configure a HybridRetriever with specified parameters.

        Args:
            vector_similarity_top_k (int): Number of top results to retrieve from vector index. Defaults to 5.
            kg_similarity_top_k (int): Number of top results to retrieve from knowledge graph. Defaults to 5.
            cohere_rerank_top_n (int): Number of top results to re-rank using Cohere. Defaults to 5.

            Note: There is bm25 code is different than the code for vector and knowledge graph indexing.  It is more focused on just retrieval instead of indexing and retrieval. Hence, stuff like similarty top k doesn't apply.

        Returns:
            HybridRetriever: A retriever ready to be used to query ollama llms that has the top n results from the cohere reranker.

        Raises:
            ValueError: If there's an issue loading any of the indexes or retrievers.
            Exception: For any other unexpected errors.
        """
        logger.info("Starting to load hybrid retriever")
        logger.debug(f"Parameters: vector_similarity_top_k={vector_similarity_top_k}, "
                     f"kg_similarity_top_k={kg_similarity_top_k}, cohere_rerank_top_n={cohere_rerank_top_n}")

        try:
            logger.info("Loading vector index")
            vector_index = self.ingest_service.get_vector_index('soil_test_knowledge')
            logger.debug("Vector index loaded successfully")

            logger.info("Loading knowledge graph")
            kg_index = self.ingest_service.get_knowledge_graph()
            logger.debug("Knowledge graph loaded successfully")

            logger.info("Loading BM25 retriever")
            bm25_retriever = self.ingest_service.get_bm25_retriever()
            logger.debug("BM25 retriever loaded successfully")

            logger.info("Initializing HybridRetriever")
            retriever = HybridRetriever(
                vector_index=vector_index,
                kg_index=kg_index,
                bm25_retriever=bm25_retriever,
                vector_similarity_top_k=vector_similarity_top_k,
                kg_similarity_top_k=kg_similarity_top_k,
                cohere_rerank_top_n=cohere_rerank_top_n,
            )
            logger.info("HybridRetriever initialized successfully")

            return retriever

        except ValueError as ve:
            logger.error(f"Error loading indexes or retrievers: {str(ve)}")
            raise ValueError(f"Failed to load hybrid retriever components: {str(ve)}") from ve

        except Exception as e:
            logger.error(f"Unexpected error in load_retriever: {str(e)}", exc_info=True)
            raise RuntimeError(f"An unexpected error occurred while loading the hybrid retriever: {str(e)}") from e

    def build_prompt(self, soil_property_name: str) -> Tuple[str, list[NodeWithScore]]:
        """
        Build a prompt for the given soil property name.

        Args:
            soil_property_name (str): The name of the soil property.

        Returns:
            Tuple[str, list[NodeWithScore]]: The prompt string and a list of nodes with scores.
        """
        # Get the property_dict from the property_name.
        try:
            property_dict = next((item for item in properties if item["name"].lower() == soil_property_name.lower()), None)
            if property_dict is None:
                raise ValueError(f"Property name {soil_property_name} not found in properties list.")
            optimal_range = property_dict["optimal_range"]
            value = self._get_property_value(soil_property_name)
            logger.debug(f"Property: {soil_property_name}, Value: {value}, Optimal Range: {optimal_range}")

            retriever_prompt = f"Cannabis cultivation guidance for {soil_property_name} at value {value}; optimal range is {optimal_range}. Includes actionable advice on adjustments if current value is outside optimal conditions."
            nodes_with_score = self.retriever.retrieve(retriever_prompt)
            context = "\n".join([node.get_content() for node in nodes_with_score])
            prompt_str = instruction.format(
                        context=context,
                        property=soil_property_name,
                        value=value,
                        optimal_range=optimal_range
                    )
        except ValueError as e:
            logger.error("ValueError Occured.",exc_info=True)
            raise ValueError(f"ValueError Occured: {e}")
        except AttributeError as e:
            logger.error("AttributeError Occured.",exc_info=True)
            raise AttributeError(f"AttributeError Occured: {e}")
        except KeyError as e:
            logger.error("KeyError Occured.",exc_info=True)
            raise KeyError(f"KeyError Occured: {e}")
        except Exception as e:
            logger.error("Exception Occured.",exc_info=True)
            raise Exception(f"Exception Occured: {e}")
        return prompt_str, nodes_with_score


    def generate_response(self, soil_property_name: str) -> dict[str, Any]:
        prompt_str, nodes_with_score = self.build_prompt(soil_property_name)
        response = self.ask_question(prompt_str, nodes_with_score)
        return response

    # Function to run a query with the retriever and custom prompt
    def ask_question(self,query: str, nodes_with_score: list[NodeWithScore]) -> dict[str, Any]:
        # Initialize the Ollama LLM
        # We are directly using the Ollama class in order to get to the tokens.
        ollama_llm = Ollama(model=self.model_name)
        # Use Ollama's chat method with the formatted prompt
        from llama_index.core.base.llms.types import ChatMessage, MessageRole

        messages = [ChatMessage(role=MessageRole.USER, content=query)]
        ollama_response = ollama_llm.chat(messages)

        return {
            "query": query,
            "answer": ollama_response.message.content,
            "contexts": [node.text for node in nodes_with_score],
            "token_info": {
                "prompt_tokens": ollama_response.raw.get('prompt_eval_count', 0) if ollama_response.raw else 0,
                "completion_tokens": ollama_response.raw.get('eval_count', 0) if ollama_response.raw else 0,
                "total_tokens": (ollama_response.raw.get('prompt_eval_count', 0) + ollama_response.raw.get('eval_count', 0)) if ollama_response.raw else 0
            },
            "other_info": {
                "model": ollama_response.raw.get('model') if ollama_response.raw else None,
                "total_duration": ollama_response.raw.get('total_duration') if ollama_response.raw else None,
                "load_duration": ollama_response.raw.get('load_duration') if ollama_response.raw else None,
                "eval_duration": ollama_response.raw.get('eval_duration') if ollama_response.raw else None
            }
        }

    def _get_property_value(self, property_name: str) -> float:
        m3_report = soil_report_instance.get_m3_report()
        sp_report = soil_report_instance.get_sp_report()
        if m3_report is None or sp_report is None:
            raise ValueError("M3 or SP report not loaded")
        property_name = property_name.lower()
        if hasattr(m3_report, property_name.lower()):
            return getattr(m3_report, property_name.lower())
        elif hasattr(sp_report, property_name.lower()):
            return getattr(sp_report, property_name.lower())
        else:
            raise AttributeError(f"Property {property_name} not found in either M3 or SP report")
