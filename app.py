import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate



load_dotenv()



st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI PDF Chatbot")
st.write("Upload a PDF book and ask questions from it.")


uploaded_file = st.file_uploader(
    "Upload your PDF",
    type="pdf"
)



if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



if uploaded_file is not None:

    with st.spinner("Processing PDF..."):

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_pdf_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()

        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        # Embedding model
        embedding_model = MistralAIEmbeddings()

        # Create Chroma DB
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory="Chroma-db"
        )

        st.session_state.vectorstore = vectorstore

    st.success("✅ PDF processed successfully!")


query = st.text_input("Ask a question from the PDF")

if query and st.session_state.vectorstore:

    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # LLM
    llm = ChatMistralAI(
        model="mistral-small-latest"
    )

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a helpful AI Assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),

        (
            "human",
            """
Context:
{context}

Question:
{question}
"""
        )
    ])

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    # Store chat history
    st.session_state.chat_history.append(
        ("You", query)
    )

    st.session_state.chat_history.append(
        ("AI", response.content)
    )



for sender, message in st.session_state.chat_history:

    if sender == "You":
        st.markdown(
            f"""
            <div style='
                background-color:#262730;
                padding:10px;
                border-radius:10px;
                margin-bottom:10px;
            '>
            <b> You:</b><br>{message}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f"""
            <div style='
                background-color:#1E3A5F;
                padding:10px;
                border-radius:10px;
                margin-bottom:10px;
            '>
            <b>🤖 AI:</b><br>{message}
            </div>
            """,
            unsafe_allow_html=True
        )