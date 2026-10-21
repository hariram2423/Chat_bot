import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# --- Page Configuration ---
st.set_page_config(page_title="AI Chat Assistant", page_icon="⚡", layout="centered")

# --- Custom Styling: Neon Orange & Porcelain Theme ---
st.markdown("""
<style>
    /* Main Background: Porcelain */
    .stApp {
        background-color: #F8F9FA;
        color: #1F2421;
    }
    
    /* Header / Titles */
    h1 {
        color: #FF5E00 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* Subheader & Captions */
    .stCaption {
        color: #555555 !important;
    }

    /* Chat Input Focus Ring & Styling */
    .stChatInputContainer {
        border-color: #E2E4E8 !important;
    }
    .stChatInputContainer:focus-within {
        border-color: #FF5E00 !important;
        box-shadow: 0 0 10px rgba(255, 94, 0, 0.45) !important;
    }

    /* User Message Bubble: Neon Orange Accent */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background-color: #FFF2EB;
        border-left: 4px solid #FF5E00;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 12px;
    }

    /* Assistant Message Bubble: Crisp Porcelain Card */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #FFFFFF;
        border: 1px solid #EAECEF;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ LangChain Chat")
st.caption("Powered by LangChain LCEL & OpenAI • Styled in Neon Orange & Porcelain")

# --- Initialize Session State for Chat Memory ---
# In Streamlit, every interaction re-runs the script, so session_state preserves history
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Render Existing Message History ---
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# --- Set Up the LangChain Pipeline ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an intelligent, friendly AI assistant. Give concise and well-structured answers."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()
chain = prompt | llm | parser

# --- Handle User Input & Real-Time Streaming ---
user_query = st.chat_input("Type your message here...")

if user_query:
    # 1. Display and record user message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append(HumanMessage(content=user_query))

    # 2. Stream AI response directly into the UI container
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Stream chunks using chain.stream()
        for chunk in chain.stream({
            "history": st.session_state.messages[:-1],  # prior history
            "input": user_query
        }):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")  # Cursor effect while typing

        # Final display without the cursor
        response_placeholder.markdown(full_response)

    # 3. Record AI response into memory
    st.session_state.messages.append(AIMessage(content=full_response))