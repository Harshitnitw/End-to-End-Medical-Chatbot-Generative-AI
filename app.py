from flask import Flask, render_template, jsonify, request
from langchain_pinecone import PineconeVectorStore
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from src.prompt import *
from src.helper import make_request
from dotenv import load_dotenv
import os
load_dotenv()

app=Flask(__name__)

os.environ["PINECONE_API_KEY"]=os.environ.get("PINECONE_API_KEY")
os.environ["MISTRAL_API_KEY"]=os.environ.get("MISTRAL_API_KEY")

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
)

index_name="medical"

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(index_name)      
docsearch = PineconeVectorStore(  
    index=index, embedding=embeddings  
)

retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.6,
    max_retries=3,
    max_tokens=500
)

prompt=ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}")
    ]
)

question_answer_chain=create_stuff_documents_chain(llm, prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET","POST"])
def chat():
    input=request.form["msg"]
    # response=rag_chain.invoke({"input":input})
    response=make_request(rag_chain=rag_chain,question=input)
    return str(response["answer"])

if __name__=='__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)