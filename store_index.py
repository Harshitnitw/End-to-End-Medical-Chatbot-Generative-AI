from src.helper import load_pdf_file, text_split
from pinecone import Pinecone, ServerlessSpec
from langchain_mistralai import MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore  

import os
from dotenv import load_dotenv

load_dotenv()
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
)

index_name="medical"

if pc.has_index(index_name): 
    index = pc.Index(index_name)      
    docsearch = PineconeVectorStore(  
        index=index, embedding=embeddings  
    )
    print("vectorstore already present with the following stats: ", index.describe_index_stats())
else:
    extracted_data=load_pdf_file(data="Data/")
    new_extracted_data=extracted_data[0:10]
    text_chunks=text_split(new_extracted_data)

    index = pc.Index(index_name)      
    pc.create_index(
        name=index,
        dimension=1024, # Replace with your model dimensions
        metric="cosine", # Replace with your model metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ) 
    )
    docsearch=PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=index_name,
        embedding=embeddings
    )
    print("New vectorstore created with the following stats: ", index.describe_index_stats())