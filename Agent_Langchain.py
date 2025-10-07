from langchain.agents import initialize_agent, load_tools
from langchain_ollama import OllamaLLM

# LLM init
llms = OllamaLLM(model="llama3.2:1b", temperature=0.0)

# Tools load (calculator)
tools = load_tools(["llm-math"], llm=llms)

# Agent init
agent = initialize_agent(
    tools=tools,
    llm=llms,
    agent="zero-shot-react-description",
    verbose=True
)

# Run
print(agent.run("What is 25 * 5?"))
