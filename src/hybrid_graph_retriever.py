from typing import List
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.cohere_rerank import CohereRerank
import os
from dotenv import load_dotenv

class HybridGraphRetriever(BaseRetriever):
    def __init__(
        self,
        vector_index: VectorStoreIndex,
        kg_index: KnowledgeGraphIndex,
        bm25_retriever: BM25Retriever,
        vector_similarity_top_k: int = 5,
        kg_similarity_top_k: int = 5,
        cohere_rerank_top_n: int = 5,
    ):
        self.bm25_retriever = bm25_retriever
        # Create Vector Index Retriever
        self.vector_retriever = vector_index.as_retriever(
            similarity_top_k=vector_similarity_top_k
        )

        # Use the provided Knowledge Graph Index
        self.kg_retriever = kg_index.as_retriever(similarity_top_k=kg_similarity_top_k)

        # Set up Cohere Rerank
        load_dotenv()
        api_key = os.environ["COHERE_API_KEY"]
        self.cohere_rerank = CohereRerank(api_key=api_key, top_n=cohere_rerank_top_n)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        def _add_source(node: NodeWithScore, source: str) -> NodeWithScore:
            node.node.metadata['retriever_source'] = source
            return node

        # Retrieve results and add source information
        vector_results = [_add_source(node, 'vector') for node in self.vector_retriever.retrieve(query_bundle)]
        bm25_results = [_add_source(node, 'bm25') for node in self.bm25_retriever.retrieve(query_bundle)]
        kg_results = [_add_source(node, 'kg') for node in self.kg_retriever.retrieve(query_bundle)]

        print(f"Vector results: {len(vector_results)}")
        print(f"BM25 results: {len(bm25_results)}")
        print(f"KG results: {len(kg_results)}")

        # Combine results
        all_results = vector_results + bm25_results + kg_results
        print(f"All results: {len(all_results)}")

        # Remove duplicates based on content.  The indices assign different ids to nodes with the same content.
        unique_results = {}
        for node in all_results:
            content = node.node.get_content()
            if content not in unique_results or node.score > unique_results[content].score:
                unique_results[content] = node

        unique_results_list = list(unique_results.values())
        print(f"Unique results: {len(unique_results_list)}")

        # Apply Cohere reranking
        reranked_results = self.cohere_rerank.postprocess_nodes(
            unique_results_list,
            query_str=query_bundle.query_str
        )
        print(f"Reranked results: {len(reranked_results)}")

        return reranked_results

    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._retrieve(query_bundle)
