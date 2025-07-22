import re
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

# --- Streamlit App Setup ---
st.set_page_config(page_title="University of Leeds Chemistry FAQ Chatbot")
st.title("Chemistry Chatbot")
st.write("Ask a question about University of Leeds Chemistry modules.")

print("Starting chatbot...")

# --- Load Example Questions ---
def load_examples(path = "examples.txt"):
    """ Reads examples from a text file and returns them as LangChain chat messages."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Use regular expressions to extract question-answer pairs
    qas = re.findall(r"Question:\s*(.*?)\s*Answer:\s*(.*?)(?=\nQuestion:|\Z)", text, re.DOTALL)

    # Create a list of alternating HumanMessage (question) and AIMessage (answer)
    messages = []
    for q, a in qas:
        messages.append(HumanMessage(content=q.strip()))
        messages.append(AIMessage(content=a.strip()))
    return messages

example_messages = load_examples()

# --- Prompt Template ---
system_prompt = SystemMessagePromptTemplate.from_template(f"""You are a helpful and precise assistant for answering questions about university module catalogue entries.
Instructions:
- Use the provided context to answer the question
- If asked to list or count modules, provide the exact number and a short list of their names and codes.
- If asked to summarise topics or content, give 2-4 concise bullet points or sentences that highlight the main points without repeating large sections of the context.
- Keep answers clear, readable, and free of unnecessary detail or long text dumps.
- If the answer is not explicitly states in the context, reply with: "I don't know."
""")

human_prompt = HumanMessagePromptTemplate.from_template("Context information is provided below. Use it to answer the question concisely.\n\n{context}\n\nQuestion: {question}")

# Combine system, example, and user prompts
prompt = ChatPromptTemplate.from_messages([
system_prompt,
*example_messages,
SystemMessagePromptTemplate.from_template("Now answer the next question using only the provided context."),
human_prompt
])

# --- Embeddings & Vector Store Setup ---

# Use Ollama to generate embeddings from chunked documents
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the FAISS index built in 'ingest.py' - contains the vector representations of the content
vectorstore = FAISS.load_local(
"faiss_index", # Path to the prebuilt FAISS index directory
embeddings,
allow_dangerous_deserialization=True # Required for FAISS legacy loading
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# --- LLM Setup Using Ollama ---

pipe = pipeline(
    "text-generation",
    model="deepseek-ai/deepseek-llm-7b-chat",
    device=0,
    torch_dtype=torch.float16,
    max_new_tokens=256,
    do_sample=False,
    temperature=0,
    return_full_text=False
)

# Load LLaMA 3 model using Ollama
llm = HuggingFacePipeline(pipeline=pipe)

# --- RAG Chain ---
rag_chain = (
prompt # Formatted prompt (with examples & context & question)
| llm # LLM for answering
| StrOutputParser() # Converts raw LLM output to a usable string
)

# --- Streamlit UI for User Interaction ---

# Text input field for the user to enter a question
user_question = st.text_input("Your question:", placeholder="e.g., What is the credit value of CHEM2112?")

# If the user has typed a question, run the chatbot logic
if user_question:
    with st.spinner("Retrieving information..."):
        # Step 1: Use the retriever to get relevant chunks of text
        docs = retriever.invoke(user_question)

        # Step 2: Join the chunks into one context string
        context = "\n".join([doc.page_content for doc in docs])

        # Step 3: Run the RAG chain to get an answer
        answer = rag_chain.invoke({"question": user_question, "context": context})

        # Step 4: Display the response
        st.markdown("### Answer:")
        st.write(answer.strip())

