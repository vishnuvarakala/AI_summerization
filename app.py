# app.py

import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
import torch

# Load documents
documents = SimpleDirectoryReader("/content").load_data()

# System and query prompts
system_prompt = """
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""
query_wrapper_prompt = SimpleInputPrompt("{query_str}")

# Configure HuggingFace LLM
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True}
)

# Configure embedding model
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)

# Create service context
service_context = ServiceContext.from_defaults(
    chunk_size=1024, 
    llm=llm,
    embed_model=embed_model
)

# Create index and query engine
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()

# Streamlit UI
st.title("Document-based Q&A System")

# Input query
query = st.text_input("Enter your question:")

if query:
    response = query_engine.query(query)
    st.write("### Response:")
    st.write(response)
