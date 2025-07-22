from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import torch
import re
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
import streamlit as st

print("Starting chatbot...")
def load_examples(path = "examples.txt"):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    qas = re.findall(r"Question:\s*(.*?)\s*Answer:\s*(.*?)(?=\nQuestion:|\Z)", text, re.DOTALL)
    return [HumanMessage(content=q.strip()) if i & 2 == 0 else AIMessage(content=a.strip()) for i, (q, a) in enumerate(qas)]

example_messages = load_examples()

system_prompt = SystemMessagePromptTemplate.from_template(f"""You are a helpful and precise assistant for answering questions about university module catalogue entries.
Instructions:
    - Use the provided context to answer the question
    - If asked to list or count modules, provide the exact number and a short list of their nmaes and codes.
    - If asked to summarise topics or content, give 2-4 concise bullet points or sentences that highlight the main points without repeating large sections of the context.
    - Keep answers clear, readable, and free of unnecessary detail or long text dumps.
    - If the answer is not explicitly states in the context, reply with: "I don't know." 
""")

human_prompt = HumanMessagePromptTemplate.from_template("Context information is provided below. Use it to answer the question concisely.\n\n{context}\n\nQuestion: {question}")

prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    *example_messages,
    SystemMessagePromptTemplate.from_template("Now answer the next question using only the provided context."),
    human_prompt
])

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

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


# Step 2: Load the LLaMA 3 model using Ollama (the language model that will generate answers)
model_name = "deepseek-ai/deepseek-llm-7b-chat"

llm = HuggingFacePipeline(pipeline=pipe)

# Step 3: Create the RAG chain (prompt -> LLM -> string output)
rag_chain = (
    prompt 
    | llm 
    | StrOutputParser() # 'StrOutputParser' converts raw LLM output into a string
)

st.set_page_config(page_title="University of Leeds Chemistry FAQ Chatbot")
st.title("Chemistry Chatbot")
st.write("Ask a question about University of Leeds Chemistry modules.")

user_question = st.text_input("Your question:", placeholder="e.g., What is the credit value of CHEM2112?")

if user_question:
    with st.spinner("Retrieving information..."):
        docs = retriever.invoke(user_question)
        context = "\n".join([doc.page_content for doc in docs])
        answer = rag_chain.invoke({"question": user_question, "context": context})
        st.markdown("### Answer:")
        st.write(answer.strip())