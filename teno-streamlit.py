import streamlit as st
from langchain.prompts import PromptTemplate
from langsmith import traceable
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import torch
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langsmith import traceable
from langgraph.checkpoint.memory import MemorySaver
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
import uuid
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Teno - Konselor AI (Beta)",
    page_icon="🧠",
    layout="centered"
)

# Check for API key
def check_api_key():
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key is None:
        st.error("GEMINI_API_KEY belum diatur. Mohon periksa file .env Anda.")
        st.stop()
    return api_key

# Set LangSmith environment variables
os.environ["LANGSMITH_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

# System prompt: mendefinisikan "Teno" konselor profesional
system_prompt = SystemMessagePromptTemplate.from_template(
    """
Kamu adalah Teno, seorang konselor profesional yang penuh empati.
Tugasmu adalah merespons dengan cara singkat, manusiawi, dan berfokus langsung pada inti masalah pengguna.
Gunakan bahasa sederhana, suportif, dan hindari kesan menggurui.

Selalu perhatikan percakapan sebelumnya (history) agar jawabanmu konsisten dan relevan dengan konteks yang sedang berjalan.
""".strip()
)

# Template lengkap untuk digunakan di app
prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    MessagesPlaceholder(variable_name="history"),   # Placeholder untuk history chat
    HumanMessagePromptTemplate.from_template(
        """
(Perhatikan konteks berikut bila relevan.)
{context}

Pertanyaan dari pengguna:
{question}

Berikan pemahaman atau dukungan yang hangat, lalu akhiri dengan kalimat yang membuka ruang agar pengguna mau berbagi lebih banyak atau memperdalam pembicaraan.
        """.strip()
    )
])

# Load SBERT model dan pastikan menggunakan GPU jika tersedia
@st.cache_resource
def load_sbert_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('naufalihsan/indonesian-sbert-large')
    model = model.to(device)
    return model, device

# Custom embeddings class for SBERT
class SBERTEmbeddings(Embeddings):
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Menggunakan model SBERT untuk menghasilkan embeddings
        embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
        embeddings = embeddings.to(self.device)  # Pindahkan embeddings ke GPU (jika ada)
        return embeddings.cpu().numpy().tolist()  # Pindahkan kembali ke CPU untuk konversi

    def embed_query(self, query: str) -> list[float]:
        # Menghasilkan embedding untuk query
        embedding = self.model.encode(query, convert_to_tensor=True)
        embedding = embedding.to(self.device)  # Pindahkan embedding ke GPU (jika ada)
        return embedding.cpu().numpy().tolist()  # Pindahkan kembali ke CPU untuk konversi

def serialize_context(context_list):
    serialized = ""
    for idx, doc in enumerate(context_list, start=1):
        serialized += f"[Dokumen {idx}]\nJudul: {doc['BAB']}\nIsi: {doc['isi']}\n\n"
    return serialized

@st.cache_resource
def load_vector_store(_sbert_embeddings):
    return FAISS.load_local("Embeddings_chonkie", _sbert_embeddings, allow_dangerous_deserialization=True)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    history: List[str]

def is_greeting(state: State) -> bool:
    greetings = ["hi", "hello", "hey", "halo", "hai"]
    return any(greet in state["question"].lower() for greet in greetings)

def retrieve(state: State):
    if is_greeting(state):
        return {"context": []}

    history = state.get("history", [])
    # Ambil 4 message terakhir (paling baru di awal)
    recent_context_texts = []
    for message in history[:4]:
        if isinstance(message, (HumanMessage, AIMessage)):
            recent_context_texts.append(message.content)

    recent_context = " ".join(recent_context_texts)

    enhanced_query = f"{recent_context} {state['question']}".strip()
    retrieved_docs = vector_store.similarity_search(enhanced_query, k=2)

    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = [
        {
            "BAB": doc.metadata.get("bab", "Tidak diketahui"),
            "isi": doc.page_content
        }
        for doc in state["context"]
    ]
    serialized_context = serialize_context(docs_content)

    history = state.get("history", [])

    # Invoke prompt, langsung passing history (tanpa join text)
    messages = prompt.invoke({
        "history": history,
        "question": state["question"],
        "context": serialized_context
    })

    response = llm.invoke(messages)

    # Salin history lama
    new_history = history.copy()

    # Tambahkan HumanMessage dan AIMessage ke awal
    new_history.insert(0, AIMessage(content=response.content))
    new_history.insert(0, HumanMessage(content=state["question"]))

    # Potong maksimal 3 pairs (6 items)
    MAX_HISTORY_PAIRS = 3
    if len(new_history) > MAX_HISTORY_PAIRS * 2:
        new_history = new_history[:MAX_HISTORY_PAIRS * 2]

    return {
        "answer": response.content,
        "history": new_history
    }

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

def main():
    # Title and description
    st.title("Teno - Empatic Chatbot 🧠")
    st.markdown("""
    Hai, Aku Teno
    Aku teman untukmu bercerita
    """)
    
    # Initialize session state
    initialize_session_state()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Ceritakan apa yang sedang Anda rasakan..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        
        # Show a spinner while processing
        with st.spinner("Teno sedang berpikir..."):
            # Create config for the graph
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Invoke the graph
            new_state = app.invoke({
                "question": user_input, 
                "history": st.session_state.history
            }, config=config)
            
            # Update history
            st.session_state.history = new_state['history']
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": new_state["answer"]})
            
        # Display assistant response
        with st.chat_message("assistant"):
            st.write(new_state["answer"])

if __name__ == "__main__":
    api_key = check_api_key()
    
    # Initialize LLM
    llm = init_chat_model(
        "gemini-2.0-flash",
        model_provider="google_genai",
        api_key=api_key
    )
    
    # Load models
    with st.spinner("Memuat model..."):
        sbert_model, device = load_sbert_model()
        sbert_embeddings = SBERTEmbeddings(sbert_model, device)
        vector_store = load_vector_store(sbert_embeddings)
    
    # Compile graph
    graph_builder = StateGraph(State)
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    graph_builder.add_edge("retrieve", "generate")

    memory = MemorySaver()
    app = graph_builder.compile(checkpointer=memory)
    
    main()