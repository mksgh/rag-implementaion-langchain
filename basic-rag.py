import os
import bs4
import pprint
from langchain import hub
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


from dotenv import load_dotenv, find_dotenv

load_dotenv()

os.getenv("GROQ_API_KEY")


# retrieve document from git hub

DOC_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"
loader = WebBaseLoader(
    web_paths=(DOC_URL,),
    bs_kwargs=dict(
        # filter specific parts of the webpage, improving efficiency.
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)

## load document
docs = loader.load()


# load model from Groq
llm = ChatGroq(model="llama3-8b-8192")

# configuring and load hugging face embedding
model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

hf_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

# chunk document
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# storing chunked embedded data in vector database
vectorstore = FAISS.from_documents(documents=splits, embedding=hf_embeddings)

# retrive data from vector db
retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    print("-" *100 , "\n", rag_chain.invoke("What is Task Decomposition?"), "\n", "-" *100)
