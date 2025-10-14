import dotenv
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_tavily import TavilySearch

dotenv.load_dotenv()

tools = [TavilySearch()]

llm = ChatOpenAI(model="gpt-4o-mini")
react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor



def main():
    result = chain.invoke(
        input = {
            "input": "search for the top 3 Langchain job in Pune and provide the job title, company name and job link",
        }
    )



if __name__ == "__main__":
    main()
