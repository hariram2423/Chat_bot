from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. Initialize the LLM and Parser
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

# 2. Build Prompt Template with a MessagesPlaceholder for history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and concise AI assistant."),
    MessagesPlaceholder(variable_name="history"),  # Slot where prior messages are injected
    ("human", "{input}")                           # The current user message
])

# 3. Base LCEL Chain
base_chain = prompt | llm | parser

# 4. Set up an in-memory session store
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Retrieve existing chat history or initialize a new one for a session."""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 5. Wrap the base chain with message history management
chain_with_history = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 6. Interactive CLI loop with real-time streaming
def start_chat():
    session_config = {"configurable": {"session_id": "user_session_1"}}
    print("=" * 55)
    print("🤖 AI Chatbot ready! Type 'exit' or 'quit' to end.")
    print("=" * 55)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            # Exit condition
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            print("AI: ", end="", flush=True)

            # Real-time token streaming using .stream()
            for chunk in chain_with_history.stream(
                {"input": user_input},
                config=session_config
            ):
                print(chunk, end="", flush=True)
            print()  # newline after completion

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break

if __name__ == "__main__":
    start_chat()