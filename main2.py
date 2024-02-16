# Had problems tokenizing documents so added a log and error handling as well as a processed files to avoid tokenizing again

from glob import glob

import langchain_openai
from langchain import text_splitter, chains
from langchain_community import document_loaders, vectorstores, chat_models

import config

import logging
import traceback

PDF_DIR = "./pdfs"


# Set up basic configuration for logging
logging.basicConfig(filename="error_log.txt", level=logging.ERROR,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def load_documents(filenames):
    splitter = text_splitter.RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=48)
    documents = []
    for filename in filenames:
        try:
            loader = document_loaders.PyPDFLoader(filename)
            documents.extend(loader.load_and_split(text_splitter=splitter))
        except Exception as e:
            logging.error(f"Error processing {filename}: {str(e)}")
            logging.error(traceback.format_exc())  # Log stack trace for deeper error context
            print(f"Error processing {filename}: {str(e)}")

    return documents


if __name__ == "__main__":
    # Load files
    filenames = glob(f"{PDF_DIR}/*.pdf")
    documents = load_documents(filenames)

    # Create vector DB
    embeddings_model = langchain_openai.OpenAIEmbeddings(openai_api_key=config.OPENAI_KEY,
                                                         openai_organization=config.OPENAI_ORG_ID)
    db = vectorstores.FAISS.from_documents(documents, embeddings_model)
    db.save_local("faiss_db/", "papers")

    # Initialise QA model
    qa_model = chains.RetrievalQA.from_llm(llm=langchain_openai.ChatOpenAI(temperature=0.1,
                                                                           openai_api_key=config.OPENAI_KEY,
                                                                           openai_organization=config.OPENAI_ORG_ID),
                                           retriever=db.as_retriever())

    # Run main QA loop
    while True:
        print("What's your question?\n")
        query = input()
        if query == "exit":
            break
        print(qa_model.invoke({"query": query})["result"])
