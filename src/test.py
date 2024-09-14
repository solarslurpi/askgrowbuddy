import chromadb

# Assuming you have created a client and a collection
client = chromadb.PersistentClient(path="vectorstore")
collection = client.get_or_create_collection("vectorstore")

# Get the count of documents in the collection
document_count = collection.count()
print(f"Number of documents in the collection: {document_count}")