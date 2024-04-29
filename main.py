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
from langchain.schema import StrOutputParser
from langchain.schema.prompt_template import format_document

from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from googletrans import Translator, LANGUAGES

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
app = FastAPI()

urls = []
vectorindex_openai = None
urlData = None
givenLanguage = None

os.environ['OPENAI_API_KEY'] ='sk-proj-xbuwz81wSU3LWQRuKEOpT3BlbkFJnMK3FnswKawmhzv5Wxxb'
os.environ['GOOGLE_API_KEY'] ='AIzaSyAmgeetTHhK8ZurIMUbHm03jieRqpeP_9A'

gemini_llm = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.7, top_p=0.85)

# Initialise LLM with required params
gpt_llm = OpenAI(temperature=0.9, max_tokens=500)

class requestURLs(BaseModel):
    urls : list[str]

class requestQuery(BaseModel):
    query : str


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with the appropriate origin or "*" for all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

def detect_language(text):
    global givenLanguage
    translator = Translator()
    # print("Detect Language")
    # print(translator)
    try:
        detected_lang = translator.detect(text).lang
        # print(detected_lang)
        givenLanguage = detected_lang
        return detected_lang
    except Exception as e:
        print("Language Detection Error:", e)
        raise HTTPException(status_code=400, detail="Language Detection Error")

def translate_to_other(query,srcLang,destLang):
    global givenLanguage
    if query is None:
        raise HTTPException(status_code=400, detail="Please provide Query.")
    translator = Translator()
    print("Translate to Given Language")
    try:
        translated_text = translator.translate(query, src=srcLang, dest=destLang).text
        # print(translated_text)
        return translated_text
    except Exception as e:
        print("Translation Error:", e)
        raise HTTPException(status_code=400, detail="Language Conversion Error")

@app.get("/hello")
async def hello():
    return "Welcome"

@app.post("/getURLs")
async def getURLs(data : requestURLs):
    urls = data.urls
    # load data
    # print(urls)
    loader = UnstructuredURLLoader(urls=urls)
    # print(loader)
    global urlData
    global vectorindex_openai
    urlData = loader.load()
    # print(len(urlData))
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(urlData)
    # print("Docs types ")
    # print(type(docs))
    # print(len(docs))
    # Create the embeddings of the chunks using openAIEmbeddings
    embeddings = OpenAIEmbeddings()
    # Pass the documents and embeddings inorder to create FAISS vector index
    vectorindex_openai = FAISS.from_documents(docs, embeddings)
    # print(vectorindex_openai)
    return "URLs Loaded!!"
    
@app.post("/getResponse")
async def getResponse(data : requestQuery):
    query = data.query
    global vectorindex_openai
    # print(vectorindex_openai)
    if vectorindex_openai is None:
        raise HTTPException(status_code=400, detail="Vector index not initialized. Please provide URLs.")
    detected_language = detect_language(query)
    if detected_language != 'en':
        new_query = translate_to_other(query,detected_language,'en')
        # print(vectorindex_openai.as_retriever)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=gpt_llm, retriever=vectorindex_openai.as_retriever())
        # print(chain)
        result = chain({"question": new_query}, return_only_outputs=True)
        result["answer"] = translate_to_other(result["answer"],'en',detected_language)
        return result
    else:
        print(vectorindex_openai.as_retriever)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=gpt_llm, retriever=vectorindex_openai.as_retriever())
        print(chain)
        result = chain({"question": query}, return_only_outputs=True)
        return result

@app.post("/getSummary")
async def getSummary():
    global urlData
    if urlData is None:
        raise HTTPException(status_code=400, detail="Please provide URLs.")

    doc_prompt = PromptTemplate.from_template("{page_content}")
    llm_prompt_template = """Write a detailed summary of the following:
    "{text}"
    DETAILED SUMMARY:"""
    llm_prompt = PromptTemplate.from_template(llm_prompt_template)
    print(llm_prompt)

    stuff_chain = (
        {
            "text": lambda urlData: "\n\n".join(
                format_document(doc, doc_prompt) for doc in urlData
            )
        }
        | llm_prompt
        | gemini_llm
        | StrOutputParser()
    )
    input_text = stuff_chain.invoke(urlData)
    input_text = translate_to_other(input_text,'en',givenLanguage)
    return input_text