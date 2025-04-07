from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

def load_pdf_file(data):
    loader=DirectoryLoader(data,
                           glob="*.pdf",
                           loader_cls=PyPDFLoader)    
    documents=loader.load()
    return documents

def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

def make_request(rag_chain,question, delay=1):
    while(True):
        try:
            answer = rag_chain.invoke({"input":question})

    # Check if the response has a status_code attribute (unlikely for LangChain outputs)
            if hasattr(answer, 'status_code') and answer.status_code == 429:
    #             # print(f"Rate limit hit! Retrying in {delay} seconds...")
                time.sleep(delay)
                # delay *= 2  # Exponential backoff
                continue  # Retry the request

            # print("Answer: ", answer)
            # print("----------")
            # with open("answers.txt", "a") as f:
            #     f.write(f"Question: {question}\n")
            #     f.write(f"Answer: {answer}\n")
            #     f.write("----------\n")
            return answer  # Return the answer if successful

        except Exception as e:
    #         print(f"Error occurred: {e}")
            time.sleep(delay)
            # delay *= 2  # Increase delay for next retry

    # print("Max retries reached. Skipping this question.")
    # return None