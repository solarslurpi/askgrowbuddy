import os
from datetime import datetime

from llama_index.core.schema import TextNode
from neo4j import GraphDatabase

from src.knowledge_graph import BuildGraphIndex


def create_test_nodes():
    """Create test TextNodes with sample content."""
    return [
        TextNode(
            text="Calcium is taken up by the xylem to the leaves.",
            metadata={
                "source": "test_doc_1",
                "created_at": datetime.now().isoformat()
            }
        ),
        TextNode(
            text="Plants that are deficient of phosphorus will tend to be lighter in color, short and many times will display a reddish coloration.",
            metadata={
                "source": "test_doc_2",
                "created_at": datetime.now().isoformat()
            }
        ),
        TextNode(
            text="The ideal pH for growing Cannabis is 6.8-6.9.",
            metadata={
                "source": "test_doc_3",
                "created_at": datetime.now().isoformat()
            }
        )
    ]

def verify_neo4j_contents(database_name: str):
    """Query Neo4j and print the current state of the graph."""
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        raise ValueError("NEO4J_PASSWORD environment variable not set")

    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", password))

    try:
        with driver.session(database=database_name) as session:
            # Count nodes by label
            node_counts = session.run("""
                MATCH (n)
                RETURN labels(n) as label, count(*) as count
                ORDER BY label
            """)

            print("\nNode Counts:")
            print("-" * 50)
            for record in node_counts:
                print(f"{record['label']}: {record['count']}")

            # Get all relationships
            relationships = session.run("""
                MATCH (s)-[r]->(o)
                RETURN type(r) as rel_type,
                       s.name as subject,
                       o.name as object
                ORDER BY rel_type
            """)

            print("\nRelationships Found:")
            print("-" * 50)
            for record in relationships:
                print(f"({record['subject']}) -[{record['rel_type']}]-> ({record['object']})")

    finally:
        driver.close()

def main():
    """Run the graph insertion tests."""
    try:
        TEST_DB = "test"

        print("\nStarting Knowledge Graph Insertion Tests")
        print("=" * 50)

        # Create test nodes
        test_nodes = create_test_nodes()
        print(f"\nCreated {len(test_nodes)} test nodes")

        # Initialize graph builder
        builder = BuildGraphIndex()

        # Build graph
        print("\nBuilding graph...")
        builder.build_graph_index(
            text_nodes=test_nodes,
            database_name=TEST_DB,
            llm_model_name="mistral_triplets"
        )

        # Verify contents
        print("\nVerifying Neo4j contents...")
        verify_neo4j_contents(TEST_DB)

    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
