import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
import pickle
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
import torch

# Set Streamlit configuration
st.set_page_config(page_title="Vector Store Index with Llama-2", layout="centered")

# Load documents from a directory
# documents = SimpleDirectoryReader("/content").load_data()

# System and query wrapper prompts for HuggingFaceLLM
system_prompt = """
You are a Question & Answer assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""
query_wrapper_prompt = SimpleInputPrompt("{query_str}")

# Initialize HuggingFaceLLM
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.5, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True}  # Disable quantization
)

# Initialize HuggingFaceEmbeddings for LangchainEmbedding
embed_model = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

# Create service context
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)

# Create and save vector store index
# index = VectorStoreIndex.from_documents(documents, service_context=service_context)
# with open('vector_store_index.pkl', 'wb') as f:
#     pickle.dump(index, f)



# Load vector store index
with open('/content/vector_store_index.pkl', 'rb') as f:
    index = pickle.load(f)

# Streamlit UI
st.title("Vector Store Index with Llama-2")

# User input prompt
prompt = st.text_input("Enter your query:", "")

# Button to trigger response generation and search
if st.button("Submit"):
    if prompt:
        with st.spinner("Generating response..."):
            # Query the vector store index
            query_engine = index.as_query_engine()
            response = query_engine.query(prompt)

            # Display the response
            if response:
                st.success("Response generated!")
                st.write(response)
            else:
                st.warning("No relevant information found in the vector store.")
    else:
        st.error("Please enter a query.")