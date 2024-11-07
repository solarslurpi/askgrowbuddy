import spacy

nlp = spacy.load("en_core_web_sm")

def print_detailed_analysis(text: str):
    doc = nlp(text)
    print(f"\nAnalyzing: {text}")
    print("-" * 60)
    print(f"{'Token':<15} {'Dep':<12} {'Head':<15} {'Children'}")
    print("-" * 60)
    for token in doc:
        children = [child.text for child in token.children]
        print(f"{token.text:<15} {token.dep_:<12} {token.head.text:<15} {children}")

test_sentences = [
    "Calcium is taken up by the xylem to the leaves.",
    "The ideal pH for growing Cannabis is between 6.8-6.9.",
    "Plants that are deficient of phosphorus will display a reddish coloration.",
    "High pH reduces iron uptake.",
    "Nitrogen moves from the soil into the roots."
]

print("DETAILED DEPENDENCY ANALYSIS")
print("=" * 60)

for text in test_sentences:
    print_detailed_analysis(text)
    # Count SVO relationships
    doc = nlp(text)
    svo_count = len([sent for sent in doc.sents
                    if any(tok.dep_ == "nsubj" for tok in sent)
                    and any(tok.dep_ == "dobj" for tok in sent)])

    # Additional analysis
    subjects = [t for t in doc if t.dep_ in ("nsubj", "nsubjpass")]
    objects = [t for t in doc if t.dep_ in ("dobj", "pobj")]

    print("\nAnalysis:")
    print(f"SVO Count: {svo_count}")
    print(f"Subjects found: {[t.text for t in subjects]}")
    print(f"Objects found: {[t.text for t in objects]}")
    print("=" * 60)