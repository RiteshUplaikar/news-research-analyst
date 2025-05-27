import os
import streamlit as st
import pickle
import time
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')



from dotenv import load_dotenv
env_path = os.path.join(os.getcwd(), '.env')
load_dotenv(env_path)  # take environment variables from .env (especially openai api key)
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
import os
print("🔑 API KEY FROM ENV:", os.getenv("OPENAI_API_KEY"))


st.title(" News Research Tool 📈")
st.sidebar.title("News Article URLs")


urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9, max_tokens=500)


if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    print(docs)
    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(0.2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        # Load FAISS vectorstore
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        # Set up ChatOpenAI model
        from langchain.chat_models import ChatOpenAI
        from langchain.chains import RetrievalQA

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.9, max_tokens=500)

        # Build the RetrievalQA chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )

        # Run the query (correct input key is "question")
        output = chain({"query": query}, return_only_outputs=True)

        # Display answer and sources
        st.header("Answer")
        st.write(output["result"])

        if output.get("source_documents"):
            st.subheader("Sources")
            for source_doc in output["source_documents"]:
                st.write(source_doc.metadata.get("source", "No source info"))
