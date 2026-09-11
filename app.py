import streamlit as st
import os
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from src.prompt import system_prompt

# 1. Page Configuration & Custom CSS Injection for Premium Styling
st.set_page_config(
    page_title="Dr. Query - Conversational Medical RAG Assistant",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Premium dark dashboard theme */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0e1322 !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Brand Header Box with a clean solid accent background */
    .brand-container {
        background-color: #111827;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 24px;
        text-align: center;
        border: 1px solid #1f2937;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    
    .brand-title {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
        margin: 0 !important;
        padding: 0 !important;
        letter-spacing: -0.5px;
    }
    
    .brand-subtitle {
        font-size: 1.0rem !important;
        color: #38bdf8 !important; /* Cyan accent color */
        margin-top: 8px !important;
        font-weight: 500 !important;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* Style the Chat Messages to look premium and distinct */
    div[data-testid="stChatMessage"] {
        background-color: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 12px !important;
        padding: 14px !important;
        margin-bottom: 12px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    }

    /* Suggested Question Buttons (Card Style) */
    div.stButton > button {
        background-color: #111827 !important;
        color: #38bdf8 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease-in-out !important;
        text-align: center !important;
        width: 100%;
        display: block;
    }
    div.stButton > button:hover {
        background-color: #38bdf8 !important;
        color: #0b0f19 !important;
        border-color: #38bdf8 !important;
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.2) !important;
    }

    /* Sidebar Reset Button Custom Red Styling */
    section[data-testid="stSidebar"] div.stButton > button {
        background-color: #1f2937 !important;
        color: #f1f5f9 !important;
        border: 1px solid #374151 !important;
        padding: 8px 12px !important;
    }
    section[data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #ef4444 !important;
        color: #ffffff !important;
        border-color: #ef4444 !important;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# 2. Load Environment Variables / Secrets
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    try:
        PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    except Exception:
        pass

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
else:
    st.error("`PINECONE_API_KEY` is missing. Please add it to your environment variables or Streamlit secrets.")
    st.stop()

# 3. Cached Resource Loader (Loads embeddings & vector store once)
@st.cache_resource
def get_retriever():
    embeddings = download_hugging_face_embeddings()
    index_name = "medical-chatbot"
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    return docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 4. Session State Initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 5. Sidebar Layout
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-top: 15px; margin-bottom: 25px;">
        <span style="font-size: 1.4rem; font-weight: 800; color: #38bdf8; border: 2px solid #38bdf8; padding: 6px 14px; border-radius: 8px; font-family: sans-serif; letter-spacing: 0.5px;">DR. QUERY</span>
    </div>
    """, unsafe_allow_html=True)
    
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="Enter your sk-...",
        help="Provide your personal OpenAI API Key to authenticate. Keys are never saved or stored."
    ).strip()
    
    st.markdown("""
    <div style="background-color: #111827; border-left: 3px solid #38bdf8; padding: 12px 15px; border-radius: 4px; margin-top: 15px; margin-bottom: 15px; border-top: 1px solid #1f2937; border-right: 1px solid #1f2937; border-bottom: 1px solid #1f2937;">
        <strong style="color: #ffffff; font-size: 0.85rem; display: block; margin-bottom: 4px;">Security Pledge</strong>
        <span style="color: #94a3b8; font-size: 0.75rem; line-height: 1.4; display: block;">
            Your OpenAI key is processed in volatile memory only and is never saved, logged, or written to disk.
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# 6. Main Brand Header
st.markdown("""
<div class="brand-container">
    <h1 class="brand-title">Dr. Query</h1>
    <p class="brand-subtitle">Conversational Medical RAG Assistant</p>
</div>
""", unsafe_allow_html=True)

# 7. Render Chat History
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# 8. Empty State & Suggested Prompts
clicked_prompt = None
if len(st.session_state.chat_history) == 0:
    st.markdown("""
    <div style="background-color: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 24px; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <h3 style="margin-top: 0; color: #38bdf8; font-size: 1.25rem; font-weight: 700;">Welcome to Dr. Query</h3>
        <p style="margin-bottom: 0; color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">
            I am a clinical assistant designed to answer queries about medical conditions, symptoms, and treatments using curated guidelines.
            <br><br>
            To begin, enter your OpenAI API key in the sidebar, and type a question below or select one of the suggested prompts.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h4 style='font-size: 1.0rem; color: #94a3b8; font-weight: 600; margin-bottom: 12px;'>Suggested Questions</h4>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("What is Diabetes?", use_container_width=True):
            clicked_prompt = "What is Diabetes?"
    with col2:
        if st.button("How are Allergies treated?", use_container_width=True):
            clicked_prompt = "How are Allergies treated?"
    with col3:
        if st.button("What is Acne?", use_container_width=True):
            clicked_prompt = "What is Acne?"

# 9. Main User Chat Input
prompt_input = st.chat_input("Ask a medical question...")
if clicked_prompt:
    prompt_input = clicked_prompt

# 10. Query Processing
if prompt_input:
    if not openai_api_key:
        st.warning("Please enter your OpenAI API Key in the sidebar to start chatting.")
    else:
        # Display user message in chat container
        with st.chat_message("user"):
            st.markdown(prompt_input)
        
        # Query RAG Chain
        with st.chat_message("assistant"):
            with st.spinner("Analyzing context & generating response..."):
                try:
                    chatModel = ChatOpenAI(
                        model="gpt-4o",
                        api_key=openai_api_key
                    )
                    
                    qa_prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", system_prompt),
                            MessagesPlaceholder("chat_history"),
                            ("human", "{input}"),
                        ]
                    )
                    
                    contextualize_q_system_prompt = (
                        "Given a chat history and the latest user question "
                        "which might reference context in the chat history, "
                        "formulate a standalone question which can be understood "
                        "without the chat history. Do NOT answer the question, "
                        "just reformulate it if needed and otherwise return it as is."
                    )
                    contextualize_q_prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", contextualize_q_system_prompt),
                            MessagesPlaceholder("chat_history"),
                            ("human", "{input}"),
                        ]
                    )
                    
                    retriever = get_retriever()
                    history_aware_retriever = create_history_aware_retriever(
                        chatModel, retriever, contextualize_q_prompt
                    )
                    
                    question_answer_chain = create_stuff_documents_chain(chatModel, qa_prompt)
                    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
                    
                    response = rag_chain.invoke({
                        "input": prompt_input,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    answer = response["answer"]
                    st.markdown(answer)
                    
                    # Store in chat history
                    st.session_state.chat_history.append(HumanMessage(content=prompt_input))
                    st.session_state.chat_history.append(AIMessage(content=answer))
                    
                except Exception as e:
                    error_msg = str(e)
                    if "AuthenticationError" in error_msg or "invalid_api_key" in error_msg or "Incorrect API key" in error_msg:
                        st.error("Authentication Error: The provided OpenAI API key is invalid or could not be verified.")
                    else:
                        st.error("An error occurred while processing your request with OpenAI.")
