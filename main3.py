import glob
import logging
import os
import time
import traceback

import langchain_openai
from langchain import text_splitter, chains
from langchain_community import document_loaders, vectorstores, chat_models

import config

PDF_DIR = "./pdfs"
DB_FILE = "faiss_db/"

filenames = glob.glob(os.path.join(PDF_DIR, "*.pdf"))

logging.basicConfig(filename="log.txt", level=logging.INFO,  # Lower log level to INFO for more details
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
            logging.debug(traceback.format_exc())
            print(f"Error processing {filename}: {str(e)}")

    return documents


def main():
    print("Loading vector DB...")
    print("DB file exists:", os.path.exists(DB_FILE))
    if os.path.exists(DB_FILE):
        embeddings = langchain_openai.OpenAIEmbeddings(
            model=config.MODEL,
            openai_api_key=config.OPENAI_KEY,
            openai_organization=config.OPENAI_ORG_ID)
        db = vectorstores.FAISS.load_local(DB_FILE, embeddings)
        logging.info("Loaded vector DB from file")
    else:
        documents = load_documents(filenames)
        embeddings_model = langchain_openai.OpenAIEmbeddings(
            model=config.MODEL,
            openai_api_key=config.OPENAI_KEY,
            openai_organization=config.OPENAI_ORG_ID)
        db = vectorstores.FAISS.from_documents(documents, embeddings_model)
        db.save_local("faiss_db/")

    qa_model = chains.RetrievalQA.from_llm(
        llm=langchain_openai.ChatOpenAI(temperature=0.1,
                                        openai_api_key=config.OPENAI_KEY,
                                        openai_organization=config.OPENAI_ORG_ID,
                                        model_name=config.MODEL_NAME),
        retriever=db.as_retriever())

    while True:
        try:
            start_time = time.time()  # Measure response time
            print('\033[94m' + "What's your question?\n" + '\033[0m')
            query = input()
            if query == "exit":
                break
            print('\033[92m' + qa_model.invoke({"query": query})["result"] + '\033[0m')
            logging.info(f"Response time: {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logging.error(f"Error during QA interaction: {str(e)}")
            logging.debug(traceback.format_exc())
            print("An error occurred. Please try again.")


if __name__ == "__main__":
    main()
