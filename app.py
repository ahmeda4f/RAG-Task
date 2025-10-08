import streamlit as st
import pandas as pd
import chardet
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

st.set_page_config(page_title="MovieRAG 🎥", page_icon="🎬")
st.title("🎬 MovieRAG: 2025 Popular Movies Q&A")
st.markdown("""
This mini **RAG app** answers questions about **popular 2025 movies**  
based on preloaded dataset.  
⚠️ *It only knows movies that exist in the CSV — not general ones!*
""")

csv_path = "movies.csv"

try:
    with open(csv_path, "rb") as f:
        result = chardet.detect(f.read())
    encoding = result["encoding"]
    df = pd.read_csv(csv_path, encoding=encoding)
except Exception as e:
    st.error(f"Error loading CSV: {e}")
    st.stop()

df["combined"] = df.apply(
    lambda r: f"Title: {r['Title']}\nGenre: {r['Genre']}\nDirector: {r['Director']}\nCast: {r['Cast']}\nBox Office: {r['Box Office in currency of  production country']}\nFact: {r['Fun Fact']}",
    axis=1
)

loader = DataFrameLoader(df, page_content_column="combined")
docs = loader.load()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./movie_db"
)

pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=512
)
llm = HuggingFacePipeline(pipeline=pipe)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})
model = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

st.success("RAG model is ready! Ask about 2025 movies 🎥")

query = st.text_input("Ask something about a 2025 movie (e.g., 'fun fact about Superman')")

if st.button("Ask"):
    if query.strip():
        with st.spinner("Thinking..."):
            result = model.invoke({"query": query})
            st.subheader("🎞️ Answer")
            st.write(result["result"])
    else:
        st.warning("Please type a question first!")

with st.expander("ℹ️ Model Limitations"):
    st.markdown("""
    - Trained only on the CSV data provided.
    - Answers are limited to 2025's popular movies.
    """)
