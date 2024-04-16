from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from prompts import context
from readFiles import fileLoader

toolList = fileLoader()

tools = []

tools.extend(toolList)

Settings.llm = Ollama(model="mistral")

agent = ReActAgent.from_tools(tools, llm=Settings.llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)
