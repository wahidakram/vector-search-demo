import os
from glob import glob
from langchain_community import document_loaders, vectorstores, chat_models
from langchain import text_splitter, chains
import langchain_openai
import config

PDF_DIR = "./pdfs"


def load_documents(filenames):
    splitter = text_splitter.RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=48)
    documents = []
    for filename in filenames:
        loader = document_loaders.PyPDFLoader(filename)
        documents.extend((loader.load_and_split(text_splitter=splitter)))

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

