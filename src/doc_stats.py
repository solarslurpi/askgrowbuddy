from typing import List, Dict, Any

from llama_index.core import Document as LlamaIndexDocument

class DocStats:
    @classmethod
    def print_llama_index_docs_summary_stats(cls, documents: List[LlamaIndexDocument]) -> Dict[str, Any]:
        """
        Generate summary statistics for the ingested documents.

        Args:
        documents (List[Document]): List of ingested documents.

        Returns:
        Dict[str, Any]: Summary statistics of the document set.
        """
        total_docs = len(documents)
        total_length = sum(len(doc.text) for doc in documents)
        all_metadata_fields = set()

        for doc in documents:
            all_metadata_fields.update(doc.metadata.keys())

        return {
            "total_documents": total_docs,
            "avg_content_length": f"{total_length / total_docs:.1f}" if total_docs > 0 else 0,
            "metadata_fields": str(list(all_metadata_fields)),
            "all_docs_have_content": all(len(doc.text) > 0 for doc in documents),
            "all_docs_have_metadata": all(len(doc.metadata) > 0 for doc in documents),
            "shortest_doc_length": min(len(doc.text) for doc in documents) if total_docs > 0 else 0,
            "longest_doc_length": max(len(doc.text) for doc in documents) if total_docs > 0 else 0,
        }
