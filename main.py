import dotenv
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.runnables import RunnableLambda

from schema import AgentResponse
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS

dotenv.load_dotenv()

tools = [TavilySearch()]

llm = ChatOpenAI(model="gpt-4o-mini")
structured_llm = llm.with_structured_output(AgentResponse)
react_prompt = hub.pull("hwchase17/react")

react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
).partial(format_instructions="")

agent = create_react_agent(llm, tools, prompt=react_prompt_with_format_instructions)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
extract_output = RunnableLambda(lambda x : x["output"])
chain = agent_executor | extract_output | structured_llm


def main():
    result = chain.invoke(
        input={
            "input": "search for the top 3 active Langchain job in Pune on linkedin and provide the job title, company name and job link",
        }
    )
    print(result)


if __name__ == "__main__":
    main()
