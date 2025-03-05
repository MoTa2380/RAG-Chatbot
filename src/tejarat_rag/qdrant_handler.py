from qdrant_client import QdrantClient

from .utils import ConfigLoader
import requests
from typing import List, Dict, Any, Optional


configs = ConfigLoader()


class SentenceEmbedder:
    """
    A class to interact with the sentence embedding model API.
    """

    def __init__(self, api_url: str, headers: Optional[Dict[str, str]] = None):
        """
        Initialize the SentenceEmbedder.

        :param api_url: The URL of the embedding API.
        :param headers: Optional headers, such as authentication tokens.
        """
        self.api_url = api_url
        self.headers = headers or {"Content-Type": "application/json"}

    def get_embeddings(self, texts: List[str]) -> Optional[Dict[str, Any]]:
        """
        Fetch embeddings for a list of input texts.

        :param texts: A list of sentences to be embedded.
        :return: A dictionary containing embeddings if successful, None otherwise.
        """
        if not texts:
            raise ValueError("The texts list cannot be empty.")

        payload = {"texts": texts}

        try:
            response = requests.post(self.api_url, json=payload, headers=self.headers)
            response.raise_for_status()  # Raise an error for HTTP errors
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching embeddings: {e}")
            return None


class RAGRetriever:
    def __init__(self, embedder, qdrant_url: str, collection_name: str) -> None:
        """
        Initializes the retriever with an embedder and Qdrant client.
        
        :param embedder: Sentence embedding model instance.
        :param qdrant_url: URL of the Qdrant server.
        :param collection_name: Name of the Qdrant collection.
        """
        self.embedder = embedder
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name

    def get_embeddings(self, sentences: List[str]) -> Dict[str, Any]:
        """
        Generates embeddings for a list of sentences.
        
        :param sentences: List of text queries.
        :return: Dictionary containing embeddings.
        """
        return self.embedder.get_embeddings(sentences)

    def retrieve_context(self, query: str, top_k: int = 5) -> List[Any]:
        """
        Retrieves relevant context from Qdrant for a given query.
        
        :param query: User query string.
        :param top_k: Number of top results to retrieve.
        :return: List of retrieved contexts.
        """
        embeddings = self.get_embeddings([query])
        query_vector = embeddings.get("embeddings", [None])[0]
        
        if query_vector is None:
            raise ValueError("Embedding generation failed for the given query.")

        response = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            with_payload=True,
            limit=top_k
        )
        
        
        context = ""
        for point in response.points:
            payload = point.payload
            context += f"question: {payload.get('question', 'N/A')}\nanswer: {payload.get('answer', 'N/A')}\n\n"
        
        return context.strip()
