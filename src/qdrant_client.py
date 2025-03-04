
from qdrant_client import QdrantClient

def get_qdrant_client(host: str = 'localhost', port: int = 6333) -> QdrantClient:
    client = QdrantClient(host=host, port=port)
    return client

def search_vectors(client: QdrantClient, collection_name: str, query_vector, top: int = 5):
    """
    Search for documents similar to the given query_vector.
    The collection is assumed to have documents with a payload containing a 'text' field.
    """
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top
    )
    return results
