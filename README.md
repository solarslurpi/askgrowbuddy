i'm finishing up on ingest using chromadb collections.  I'm thinking I'll make the collection name be hardcoded based on properties of what was done to the documents that are being stored.  For example, a collection name could be:
<user tag>_<chunk size>_<chunk_overlap>_<chunk_method>_<embed model>

alternatively, the name of the collection can be anything.

alternatively, the name of the collection can be anything and then there are metadata fields that are used to filter.

col = client.get_or_create_collection("test", metadata={"key": "value"})

how cool is this, when a collection is created, we can store the chunk size, etc.  we;ll add these as collection metdata

collection_metadata = {
    "chunk_size": 100,
    "chunk_overlap": 50,
    "chunk_method": "sliding",
    "embed_model": "bert-base-uncased"
}

col = client.get_or_create_collection("test", metadata=collection_metadata)

The challenge: Determine efficient ranking systems.

Bag of Words - How many times does each word in the question appear in each of the documents?  And if the word apprears many times in a short document versus the same amount of times in a long document, the short document has a higher score.  i.e.: The Bag of Words considers how often a word appears in the document and how long the document is. There might be other things, but this is the basic idea I got my heard around.

Llamaindex for retriever
- QueryFusionRetriever
- TreeSummarize
