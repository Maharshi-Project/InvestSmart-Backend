from fastapi import FastAPI
from pydantic import BaseModel
from langchain import OpenAI
import os
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from langchain import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

# OPENAI_API_KEY = 'sk-proj-vsJ4fg4tCUVlUVSYOTM5T3BlbkFJEgiifcEJfqFkco1mkWzZ'
# os.environ['GOOGLE_API_KEY'] = 'AIzaSyAmgeetTHhK8ZurIMUbHm03jieRqpeP_9A'

app = FastAPI()

urls = []
query = ""

gemini_llm = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.7, top_p=0.85)

# Initialise LLM with required params
gpt_llm = OpenAI(temperature=0.9, max_tokens=500) 

class request(BaseModel):
    urls : list[str]
    count : int
    query : str

class response(BaseModel):
    text : str
    sources : str

@app.get("/hello")
async def hello():
    return "Welcome"

@app.post("/getData")
async def getURLs(data : request):
    urls = data.urls
    query = data.query
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    print("\n")
    print(len(data))
    print("\n")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(data)
    print("\nDOcs types ")
    print(type(docs))
    print(len(docs))
    # Create the embeddings of the chunks using openAIEmbeddings
    embeddings = OpenAIEmbeddings()
    # Pass the documents and embeddings inorder to create FAISS vector index
    vectorindex_openai = FAISS.from_documents(docs, embeddings)
    print(vectorindex_openai.as_retriever)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=gpt_llm, retriever=vectorindex_openai.as_retriever())
    print(chain)
    result = chain({"question": query}, return_only_outputs=True)
    return result

@app.post("/getSummary")
async def getSummary(data : request):
    urls = data.urls
    query = data.query

    combined_docs = []
    loader = WebBaseLoader(urls)
    docs = loader.load()
    print(docs)

    doc_prompt = PromptTemplate.from_template("{page_content}")
    llm_prompt_template = """Write a detailed summary of the following:
    "{text}"
    DETAILED SUMMARY:"""
    llm_prompt = PromptTemplate.from_template(llm_prompt_template)
    print(llm_prompt)

    stuff_chain = (
        {
            "text": lambda docs: "\n\n".join(
                format_document(doc, doc_prompt) for doc in docs
            )
        }
        | llm_prompt
        | gemini_llm
        | StrOutputParser()
    )
    input_text = stuff_chain.invoke(docs)
    return input_text