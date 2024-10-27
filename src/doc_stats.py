from typing import List, Dict, Any
from rich import print
from rich.table import Table
from rich.console import Console
from llama_index.core import Document as LlamaIndexDocument

class DocStats:
    @classmethod
    def print_llama_index_docs_summary_stats(cls, documents: List[Any]) -> Dict[str, Any]:
        """
        Generate summary statistics for the ingested documents and print them in a table format.

        Args:
        documents (List[Document]): List of ingested documents.

        Returns:
        Dict[str, Any]: Summary statistics of the document set.
        """
        total_docs = len(documents)
        total_length = sum(len(doc.text) for doc in documents)
        all_metadata_fields = set()
        metadata_values = {}

        for doc in documents:
            all_metadata_fields.update(doc.metadata.keys())
            for key, value in doc.metadata.items():
                if key not in metadata_values:
                    metadata_values[key] = set()
                metadata_values[key].add(str(value))

        stats = {
            "Document Type": type(documents[0]).__name__,
            "Total Documents": total_docs,
            "Avg Content Length": f"{total_length / total_docs:.1f}" if total_docs > 0 else 0,
            "All Docs Have Content": all(len(doc.text) > 0 for doc in documents),
            "All Docs Have Metadata": all(len(doc.metadata) > 0 for doc in documents),
            "Shortest Doc Length": min(len(doc.text) for doc in documents) if total_docs > 0 else 0,
            "Longest Doc Length": max(len(doc.text) for doc in documents) if total_docs > 0 else 0,
        }

        # Create main table
        main_table = Table(title="Document Statistics")
        main_table.add_column("Statistic", style="cyan", no_wrap=True)
        main_table.add_column("Value", style="magenta")

        # Add rows to the main table
        for key, value in stats.items():
            main_table.add_row(key, str(value))

        # Create metadata table
        metadata_table = Table(title="Metadata Fields", show_header=True, header_style="bold magenta")
        metadata_table.add_column("Field", style="cyan")
        metadata_table.add_column("Unique Values", style="green")

        # Add rows to the metadata table
        for field in sorted(all_metadata_fields):
            unique_values = ", ".join(sorted(metadata_values[field]))
            if len(unique_values) > 50:  # Truncate long values
                unique_values = unique_values[:47] + "..."
            metadata_table.add_row(field, unique_values)

        # Add metadata table to main table
        main_table.add_row("Metadata Fields", metadata_table)

        # Print the table
        console = Console()
        console.print(main_table)

        return stats
