import os
from config import *
from env import groq_api_key, langsmith_api_key
from typing_extensions import TypedDict, Annotated
from typing import Sequence
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph.message import add_messages
from langchain_core.messages import *
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_groq import ChatGroq
from langsmith import utils

# Configuration API
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = groq_api_key
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "chat bot"

# Check if tracing works
utils.tracing_is_enabled()

# Initialisation Model
model = ChatGroq(model=MODEL_NAME)


# Define State ( of the conversation)
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


# Define Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You talk like a motorbike specialist. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define trimmer
trimmer = trim_messages(
    max_tokens=MAX_TOKENS,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Messages' History
messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]


# add a message to the history

def add_message(history: Sequence[BaseMessage], content: str, role: str = "human"):
    if role == "human":
        return history + [HumanMessage(content=content)]
    elif role == "ai":
        return history + [AIMessage(content=content)]
    else:
        return history


# Define function that calls the model
def call_model(state: State):
    chain = prompt | model
    trimmed_messages = trimmer.invoke(state["messages"])
    response = chain.invoke({"messages": trimmed_messages, "language": state["language"]})
    return {"messages": response}


# Make graph
def create_graph():
    graph = StateGraph(state_schema=State)
    graph.add_edge(START, "model")
    graph.add_node("model", call_model)
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


if __name__ == "__main__":
    app = create_graph()

    query = ""
    while query != "stop":
        query = input()
        input_message = add_message(messages, query)
        config = {"configurable": {"thread_id": THREAD_ID}}
        data = {
            "messages": input_message,
            "language": DEFAULT_LANGUAGE,
        }

        output = app.invoke(
            data, config,
        )
        output["messages"][-1].pretty_print()
        print('\n')
