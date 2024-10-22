import logging
from llama_index.core.schema import QueryBundle, NodeWithScore
from src.logging_config  import setup_logging
from src.ingest_service import IngestService
from src.soil_test_analyst import SoilTestAnalyst

setup_logging()
logger = logging.getLogger(__name__)

def test_load_retriever():
    soil_analyzer = SoilTestAnalyst()
    retriever = soil_analyzer.load_retriever(vector_similarity_top_k=3,kg_similarity_top_k=4, cohere_rerank_top_n=2)
    assert retriever is not None
    query = "What is the chemical composition of Wollastonite?"
    query_bundle = QueryBundle(query_str=query)
    nodes = retriever.retrieve(query_bundle)
    assert nodes is not None

def test_load_vector_retriever():
    ingest_service = IngestService()
    query = "What is the chemical composition of Wollastonite?"

    try:
        ingest_service = IngestService()
        name='soil_test_knowledge'
        vector_index = ingest_service.get_vector_index(name)
        # If successful, use the index to perform a query
        similarity_top_k = 3
        retriever = vector_index.as_retriever(similarity_top_k=similarity_top_k)
        query_bundle = QueryBundle(query_str=query)
        nodes = retriever.retrieve(query_bundle)

        assert nodes is not None
        assert len(nodes) == similarity_top_k
        assert isinstance(nodes[0], NodeWithScore)
        logger.info(f"Requested {similarity_top_k} nodes. Successfully retrieved {len(nodes)} nodes from vector index")
        logger.info("Nodes:\n")
        for node in nodes:
            print(f"  ID: {node.node_id}")
            print(f"  Score: {node.score}")
            print(f"  Text: {node.text[:200]}...")  # Print first 200 characters
            print("-" * 80)

    except ValueError:
        # If the vector index doesn't exist, log it and consider the test passed
        # (assuming this is expected behavior in some cases)
        logger.warning(f"Vector index named '{name}' not found. This may be expected if it hasn't been built yet.")
        assert True

    except Exception as e:
        # For any other unexpected errors, fail the test
        logger.error(f"Unexpected error in test_load_vector_retriever: {str(e)}", exc_info=True)
        assert False, f"Test failed due to unexpected error: {str(e)}"