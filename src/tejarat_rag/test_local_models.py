import dspy
from .utils import ConfigLoader
from .dspy_prompt import RAGHandler

CONFIGS = ConfigLoader()


if __name__ == "__main__":

    rag_pipeline = RAGHandler()

    rag_pipeline.lm = dspy.LM(
        model=CONFIGS.get("MODEL_NAME"),
        api_key="Mohammad",
        api_base=CONFIGS.get("LOCAL_LLM_API"),
    )
    dspy.settings.configure(lm=rag_pipeline.lm)


    user_query = "یکم درمورد بانک تجارت توضیح میدی"
    answer = rag_pipeline.query(question=user_query)
    print(answer)
