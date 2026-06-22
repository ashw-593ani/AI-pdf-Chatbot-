# AI PDF Chatbot

An intelligent Retrieval-Augmented Generation (RAG) based chatbot that allows users to upload PDF documents and ask natural language questions. The application extracts content from PDFs, generates embeddings, stores them in a vector database, and uses Large Language Models (LLMs) to provide context-aware answers.

## Live Demo

The application is hosted and can be accessed here:

[ResearchMind AI Agent](https://ashw-593ani-ai-pdf-chatbot--app-7bxpl6.streamlit.app/)

## Project Demo Video

Watch the complete project demonstration here:

[Project Demo Video](https://drive.google.com/drive/u/3/home)


## Features

* Upload and process PDF documents
* Extract and chunk document text
* Generate vector embeddings for semantic search
* Store embeddings using ChromaDB
* Context-aware question answering using RAG
* Powered by LangChain and Mistral AI
* Interactive Streamlit-based user interface
* Fast and accurate document retrieval

## Tech Stack

### Languages

* Python

### Frameworks & Libraries

* LangChain
* LangChain Community
* LangChain Core
* Streamlit
* FastAPI

### LLM & Embeddings

* Mistral AI
* OpenAI (Optional)
* Sentence Transformers

### Vector Database

* ChromaDB

### Document Processing

* PyPDF
* Unstructured
* BeautifulSoup4
* lxml

### Data Handling

* Pandas
* NumPy

## Project Architecture

1. User uploads a PDF document.
2. PDF content is extracted and cleaned.
3. Text is split into manageable chunks.
4. Embeddings are generated using Sentence Transformers.
5. Embeddings are stored in ChromaDB.
6. Relevant chunks are retrieved based on user queries.
7. Retrieved context is passed to the LLM.
8. The chatbot generates accurate and context-aware responses.


## Run the Application

### Streamlit

```bash
streamlit run app.py
```

## Project Structure

```text
AI-PDF-Chatbot/
│
├── app.py
├── main.py
├── requirements.txt
├── .env
├── chroma_db/
├── documents/
├── utils/
├── README.md
└── assets/
```

## Use Cases

* Research Assistance
* Academic Document Analysis
* Business Report Q&A
* Legal Document Review
* Knowledge Base Search
* Technical Documentation Support

## Future Improvements

* Multi-PDF Chat Support
* Conversation Memory
* Source Citation Display
* Authentication System
* Cloud Deployment
* Multi-LLM Support

