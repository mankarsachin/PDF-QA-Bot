import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

# Embeddings & LLM
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0
)

# RAG Prompt Template (replaces RetrievalQA)
prompt = PromptTemplate.from_template("""
You are a helpful assistant. Answer the question based only on the following context:

CONTEXT: {context}

QUESTION: {question}

ANSWER:
""")

def process_document_to_faiss(file_name):
    """Process PDF and store in FAISS"""
    try:
        loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        vectordb = FAISS.from_documents(texts, embedding)
        vectordb.save_local(f"{working_dir}/faiss_index")
        print(f"✅ Document saved to {working_dir}/faiss_index")
        return vectordb
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None

def answer_question(user_question):
    """Modern LCEL RAG chain - NO RetrievalQA dependency"""
    try:
        # Load FAISS
        vectordb = FAISS.load_local(
            f"{working_dir}/faiss_index", 
            embedding,
            allow_dangerous_deserialization=True
        )
        
        # Create retriever
        retriever = vectordb.as_retriever(search_kwargs={"k": 4})
        
        # LCEL RAG chain (works on Python 3.14)
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain.invoke(user_question)
        
    except Exception as e:
        return f"Error: {str(e)}"
