import spacy

from src.knowledge_graph import BuildGraphIndex


def analyze_sentence_structure(text: str):
    """Analyze the dependency structure of a sentence using SpaCy."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    print("\nSpaCy Dependency Analysis:")
    print("-" * 60)
    print(f"{'Token':<15} {'Dep':<12} {'Head':<15} {'Children'}")
    print("-" * 60)
    for token in doc:
        children = [child.text for child in token.children]
        print(f"{token.text:<15} {token.dep_:<12} {token.head.text:<15} {children}")

    # Count relationships
    subjects = [t for t in doc if t.dep_ in ("nsubj", "nsubjpass")]
    objects = [t for t in doc if t.dep_ in ("dobj", "pobj")]

    print("\nRelationship Analysis:")
    print(f"Subjects found: {[t.text for t in subjects]}")
    print(f"Objects found: {[t.text for t in objects]}")
    relationship_count = len([sent for sent in doc.sents
        if any(tok.dep_ in ("nsubj", "nsubjpass") for tok in sent)
        and any(tok.dep_ in ("dobj", "pobj") for tok in sent)])
    print(f"Relationship count: {relationship_count}")

def test_triplet_extraction():
    # Test cases with expected outputs
    test_cases = [
        {
            "text": "Calcium is taken up by the xylem to the leaves.",
            "max_triplets": 1,
            "expected_triplets": [
                ("xylem", "TRANSPORTS", "calcium")
            ]
        },
        {
            "text": "Plants that are deficient of phosphorus will tend to be lighter in color, short and many times will display a reddish coloration from the accumulation of sugars in the plant.",
            "max_triplets": 2,
            "expected_triplets": [
                ("phosphorus_deficiency", "CAUSES", "light_color"),
                ("phosphorus_deficiency", "CAUSES", "short_plants")
            ]
        },
        {
            "text": "The ideal pH for growing Cannabis is between 6.8-6.9.",
            "max_triplets": 1,
            "expected_triplets": [
                ("Cannabis_pH", "OPTIMAL_VALUE_IS", "6.8-6.9")
            ]
        }
    ]

    builder = BuildGraphIndex()
    results = []

    for test_case in test_cases:
        print("\n" + "="*50)
        print(f"Testing text: {test_case['text']}")

        # Add SpaCy analysis
        analyze_sentence_structure(test_case['text'])

        print(f"\nMax triplets: {test_case['max_triplets']}")
        print("Expected triplets:")
        for t in test_case['expected_triplets']:
            print(f"  {t}")

        # Get actual results
        result = builder._extract_triplets_with_retries(
            text=test_case['text'],
            max_triplets=test_case['max_triplets'],
            model_name="claude-3-5-sonnet-20241022"
        )

        print("\nActual triplets:")
        if result['triplets']:
            for t in result['triplets']:
                print(f"  {t}")
        else:
            print("  No triplets returned")

        # Analysis
        analysis = {
            "text": test_case['text'],
            "max_triplets_requested": test_case['max_triplets'],
            "triplets_returned": len(result['triplets']) if result['triplets'] else 0,
            "respects_max_triplets": (len(result['triplets']) if result['triplets'] else 0) <= test_case['max_triplets'],
            "raw_response": result['response'],
            "token_usage": result['token_usage']
        }


        results.append(analysis)

        print("\nAnalysis:")
        print(f"Respects max triplets: {analysis['respects_max_triplets']}")
        print(f"Token usage: {analysis['token_usage']}")

    return results

def main():
    """
    Main function to run triplet extraction tests and analyze results.
    """
    try:
        print("\nStarting Knowledge Graph Triplet Extraction Tests")
        print("=" * 50)

        # Run tests
        results = test_triplet_extraction()

        # Summary report
        print("\nSUMMARY REPORT")
        print("=" * 50)
        print(f"Total test cases run: {len(results)}")

        # Compliance stats
        max_triplet_violations = sum(1 for r in results if not r['respects_max_triplets'])

        print("\nCompliance Metrics:")
        print(f"- Max triplet violations: {max_triplet_violations}/{len(results)}")

        # Token usage stats
        total_tokens = sum(r['token_usage']['total_tokens'] for r in results)
        avg_tokens = total_tokens / len(results)
        print("\nToken Usage:")
        print(f"- Total tokens: {total_tokens}")
        print(f"- Average tokens per test: {avg_tokens:.2f}")

        # Detailed results
        print("\nDETAILED RESULTS")
        print("=" * 50)
        for i, result in enumerate(results, 1):
            print(f"\nTest Case {i}:")
            print(f"Text: {result['text'][:100]}...")
            print(f"Requested triplets: {result['max_triplets_requested']}")
            print(f"Received triplets: {result['triplets_returned']}")
            print(f"Token usage: {result['token_usage']}")

    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()