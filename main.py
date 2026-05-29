# from dotenv import load_dotenv
# from langchain_mistralai import ChatMistralAI
# # from langchain_community.document_loaders import TextLoader
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# load_dotenv()



# templates = ChatPromptTemplate.from_messages(
#     [("system","you are a AI that summarizes the text"),
#      ("human","{data}")
#     ]
# )

# model  = ChatMistralAI(model = "mistral-small-2586")



from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Embedding model
embedding_model = MistralAIEmbeddings()

# Load existing Chroma DB
vectorstore = Chroma(
    persist_directory="Chroma-db",
    embedding_function=embedding_model
)

# Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# LLM
llm = ChatMistralAI(model="mistral-small-latest")

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

print("RAG System Created")
print("Press 0 to exit")

# Chat loop
while True:

    query = input("You : ")

    if query == "0":
        break

    docs = retriever.invoke(query)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    print("AI Answer :", response.content)