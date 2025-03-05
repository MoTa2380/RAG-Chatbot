import dspy
import mlflow
import os
from .utils import ConfigLoader


configs = ConfigLoader()

# Set Proxy
# os.environ["HTTPS_PROXY"] = configs.get("OPENAI_PROXY")



# Mlflow Tracking
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy")
mlflow.dspy.autolog()


# class GenerateAnswer(dspy.Signature):
    

# class RAG(dspy.Module):
#     def __init__(self):
#         self.respond = dspy.ChainOfThought('context, question -> response')

#     def forward(self, question):
#         context = search(question).passages
#         return self.respond(context=context, question=question)


lm = dspy.LM(model="openai/gita", api_key="Mohammad", api_base="https://llm-wrapper.dv.mci.dev/")
dspy.settings.configure(lm=lm)


# Define a Chain of Thought module
class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.prog(question=question)


dspy_model = CoT()



output = dspy_model("تو کی هستی؟")
print(output)

