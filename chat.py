import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# --- Page Setup ---
st.set_page_config(page_title="LangChain Chatbot", page_icon="🤖")
st.title("🤖 LangChain Chatbot")
st.caption("Powered by gpt-4o-mini & LCEL streaming")

# --- Initialize Chain ---
@st.cache_resource
def get_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful and concise AI assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()
    return prompt | llm | parser

chain = get_chain()

# --- Session State for History ---
# Streamlit re-runs the entire script on every user interaction,
# so st.session_state is used to persist chat messages.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# --- User Input & Streaming ---
user_query = st.chat_input("Type your message here...")

if user_query:
    # 1. Display and record user message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append(HumanMessage(content=user_query))

    # 2. Stream AI response into the UI
    with st.chat_message("assistant"):
        # st.write_stream consumes a generator and displays tokens live
        response_stream = chain.stream({
            "history": st.session_state.messages[:-1],  # prior turns
            "input": user_query
        })
        full_response = st.write_stream(response_stream)

    # 3. Record AI message in history
    st.session_state.messages.append(AIMessage(content=full_response))