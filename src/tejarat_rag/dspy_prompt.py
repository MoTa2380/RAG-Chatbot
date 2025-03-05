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
    def __init__(self, is_local_llm: bool = False):
        """
        Initializes the RAG pipeline, including the embedder, retriever, and language model.
        """
        self.embedder = SentenceEmbedder(api_url=CONFIGS.get("SENTENCE_EMBEDDER_E5"))
        self.retriever = RAGRetriever(
            embedder=self.embedder,
            qdrant_url=CONFIGS.get("QDRTANT_URL"),
            collection_name=CONFIGS.get("QDRANT_COLLECTION_NAME"),
        )

        self.is_local_llm = is_local_llm
        
        self._setup_lm()
        self.rag = RAG(retriever=self.retriever)

    def _setup_lm(self):
        """
        Configures the language model (local or OpenAI-based) and applies the settings globally.
        """
        model_name = CONFIGS.get("LOCAL_MODEL_NAME") if self.is_local_llm else CONFIGS.get("OPENAI_MODEL_NAME")
        api_key = CONFIGS.get("LOCAL_LLM_API_KEY") if self.is_local_llm else CONFIGS.get("OPENAI_API_KEY")
        api_base = CONFIGS.get("LOCAL_LLM_API") if self.is_local_llm else None
        openai_config = {"proxies": PROXIES} if not self.is_local_llm else None

        self.lm = dspy.LM(
            model=model_name,
            api_key=api_key,
            api_base=api_base,
            openai_config=openai_config,
        )

        dspy.settings.configure(lm=self.lm)

    def query(self, question: str) -> str:
        """
        Queries the RAG pipeline with a given question.

        :param question: The question string to retrieve relevant documents and generate an answer.
        :return: The generated answer.
        """
        if not question:
            raise ValueError("Question cannot be empty.")

        return self.rag(question=question)



## TODO: Implement Optimizer for Prompt and Fewshot
## TODO: Get or Create train and validation set
