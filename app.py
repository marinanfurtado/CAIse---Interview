import streamlit as st
import os
import glob

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage


# ============================================================
# CONFIGURAÇÕES DE TEMA (CSS) — ESTILO VISAGIO
# ============================================================

st.set_page_config(page_title="CAIse Interview", page_icon="🤖")

st.markdown("""
<style>

body {
    font-family: 'Segoe UI', sans-serif;
}

/* Títulos */
.title {
    text-align: center;
    color: #FFFFFF;
    font-size: 40px;
    font-weight: 700;
    margin-bottom: -10px;
}

.subtitle {
    text-align: center;
    color: #EAEAEA;
    font-size: 20px;
    margin-bottom: 25px;
}

/* Bolha do BOT */
.bot-bubble {
    background-color: #1E5C4E;
    padding: 12px 18px;
    border-radius: 12px;
    color: white;
    max-width: 70%;
    margin: 10px 0;
}

/* Bolha do USUÁRIO */
.user-bubble {
    background-color: #F36C21;
    padding: 12px 18px;
    border-radius: 12px;
    color: white;
    max-width: 70%;
    margin: 10px 0 10px auto;
    text-align: right;
}

/* Caixa de boas-vindas */
.welcome-box {
    background-color: #1E5C4E;
    padding: 20px;
    border-radius: 12px;
    color: #FFFFFF;
    font-size: 16px;
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TÍTULOS
# ============================================================

st.markdown("<h1 class='title'>CAIse Interview</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>IA Entrevistadora de Cases</p>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="welcome-box">
        Bem-vindo ao <b>CAIse Interview</b>. 
        Aqui você pratica cases reais baseados nos PDFs da pasta <code>casebooks/</code>.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# API KEY
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ Falta definir a variável OPENAI_API_KEY.")
    st.stop()


# ============================================================
# FUNÇÃO PARA MONTAR RAG
# ============================================================

@st.cache_resource
def build_rag_chain(pdf_folder):

    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"Nenhum PDF encontrado na pasta: {pdf_folder}")

    # Carrega PDFs
    all_docs = []
    for file_path in pdf_files:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        all_docs.extend(docs)

    # Divide texto
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    # Embeddings baratos
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Banco vetorial
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./chroma_db"
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Modelo barato e bom
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

    # Prompts
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Reescreva a pergunta do usuário para ser independente do histórico."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_system_prompt = (
        "Você é o CAIse Interview, um entrevistador sênior de consultoria. "
        "Conduza somente cases de negócios. Use o CONTEXTO abaixo.\n\n"
        "CONTEXTO: {context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware, question_answer_chain)

    return rag_chain


# ============================================================
# CHAT
# ============================================================

PDF_FOLDER = "./casebooks"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Monta RAG
try:
    rag_chain = build_rag_chain(PDF_FOLDER)
except Exception as e:
    st.error(f"Erro ao carregar os PDFs: {e}")
    st.stop()

# Entrada do usuário
user_input = st.chat_input("Digite aqui...")

if user_input:
    # bolha do usuário
    st.markdown(f"<div class='user-bubble'>{user_input}</div>", unsafe_allow_html=True)

    # gera resposta
    response = rag_chain.invoke({
        "input": user_input,
        "chat_history": st.session_state.chat_history
    })

    answer = response["answer"]

    # bolha do bot
    st.markdown(f"<div class='bot-bubble'>{answer}</div>", unsafe_allow_html=True)

    # histórico
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=answer))


# Rodapé
st.markdown(
    """
    <hr style="margin-top:40px; border-color:#1E5C4E;">
    <p style="text-align:center; color:#EAEAEA; font-size:14px;">
        Desenvolvido em parceria com a Visagio · Powered by OpenAI & LangChain
    </p>
    """,
    unsafe_allow_html=True
)
