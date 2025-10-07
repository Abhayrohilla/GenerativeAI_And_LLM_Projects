# 👉 Q: "Ek LLM chain banao jo kisi topic pe poem likhe."

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2:1b", temperature=0.7)

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a short poem about {topic}"
)

chain = LLMChain(llm=llm, prompt=prompt)
print(chain.run("friendship"))
