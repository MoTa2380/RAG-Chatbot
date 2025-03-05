import dspy
from .qdrant_handler import SentenceEmbedder, RAGRetriever
from .mlflow_tracker import MLFlowManager
from .utils import ConfigLoader


CONFIGS = ConfigLoader()
PROXIES = {"https": CONFIGS.get("OPENAI_PROXY"), "http": CONFIGS.get("OPENAI_PROXY")}


mlflow_manager = MLFlowManager()
mlflow_manager.setup_mlflow()


class GenerateAnswer(dspy.Signature):
    """Answer the user query Considering the Context in Persian language."""

    question: str = dspy.InputField()
    context: str = dspy.InputField(desc="Answer may be in context.")
    answer: str = dspy.OutputField()


class RAG(dspy.Module):
    def __init__(self, retriever):
        self.retriever = retriever
        self.response = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = self.retriever.retrieve_context(question)

        return self.response(context=context, question=question)







class RAGHandler:
    def __init__(self):
        """
        Initializes the RAG pipeline, including the embedder, retriever, and language model.
        """
        self.embedder = SentenceEmbedder(api_url=CONFIGS.get("SENTENCE_EMBEDDER_E5"))
        self.retriever = RAGRetriever(
            embedder=self.embedder,
            qdrant_url=CONFIGS.get("QDRTANT_URL"),
            collection_name=CONFIGS.get("QDRANT_COLLECTION_NAME"),
        )

        self.lm = dspy.LM(
            model="openai/gpt-4o-mini",
            api_key=CONFIGS.get("OPENAI_API_KEY"),
            openai_config={"proxies": PROXIES},
        )
        dspy.settings.configure(lm=self.lm)

        self.rag = RAG(retriever=self.retriever)

    def query(self, question: str) -> str:
        """
        Queries the RAG pipeline with a given question.

        :param question: The question string to retrieve relevant documents and generate an answer.
        :return: The generated answer.
        """
        if not question:
            raise ValueError("Question cannot be empty.")

        return self.rag(question=question)


# embedder = SentenceEmbedder(api_url=CONFIGS.get("SENTENCE_EMBEDDER_E5"))
# retriever = RAGRetriever(
#     embedder=embedder,
#     qdrant_url=CONFIGS.get("QDRTANT_URL"),
#     collection_name=CONFIGS.get("QDRANT_COLLECTION_NAME"),
# )


# lm = dspy.LM(
#     model="openai/gpt-4o-mini",
#     api_key=CONFIGS.get("OPENAI_API_KEY"),
#     openai_config={"proxies": PROXIES},
# )
# dspy.settings.configure(lm=lm)


# rag = RAG(retriever=retriever)
# query = "یکم درمورد بانک تجارت توضیح میدی"


# output = rag(question=query)
# print(output)
