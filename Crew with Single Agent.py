from crewai import Crew, Agent, Task
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2:1b", temperature=0.7)

researcher = Agent(
    role="Researcher",
    llm="ollama/llama3.2:1b" ,
    goal="Find the latest AI news",
    backstory="You are an expert at searching and summarizing recent AI developments.",
    memory=True
)

task = Task(
    description="Summarize the top 3 AI news articles from today",
    expected_output="A short summary of 3 AI news items",
    agent=researcher
)

crew = Crew(
    agents=[researcher],
    tasks=[task]
)

result = crew.kickoff()
print(result)
