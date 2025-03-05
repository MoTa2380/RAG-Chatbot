from .dspy_prompt import RAGHandler

if __name__ == "__main__":
    rag_pipeline = RAGHandler()
    
    user_query = "یکم درمورد بانک تجارت توضیح میدی"
    answer = rag_pipeline.query(question=user_query)
    print(answer)