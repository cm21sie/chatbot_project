# Import modules for vector storage, prompt templates, output parsing, etc.
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
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings

# Load example questions from example_questions.txt and structure them for use in prompt chaining
def load_examples(path = "examples.txt"):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Extract question-answer pairs using regex
    qas = re.findall(r"Question:\s*(.*?)\s*Answer:\s*(.*?)(?=\nQuestion:|\Z)", text, re.DOTALL)

    # Alternate between HumanMessage and AIMessage for each pair
    return [HumanMessage(content=q.strip()) if i & 2 == 0 else AIMessage(content=a.strip()) for i, (q, a) in enumerate(qas)]

# Load examples
example_messages = load_examples()

# Define system-level instructions to guide the assistant's tone and response style
system_prompt = SystemMessagePromptTemplate.from_template(f"""You are a helpful and precise assistant for answering questions about university module catalogue entries.
Instructions:
- Use the provided context to answer the question
- If asked to list or count modules, provide the exact number and a short list of their nmaes and codes.
- If asked to summarise topics or content, give 2-4 concise bullet points or sentences that highlight the main points without repeating large sections of the context.
- Keep answers clear, readable, and free of unnecessary detail or long text dumps.
- If the answer is not explicitly states in the context, reply with: "I don't know."
""")

# Prompt template that presents the question and retrieved context
human_prompt = HumanMessagePromptTemplate.from_template("Context information is provided below. Use it to answer the question concisely.\n\n{context}\n\nQuestion: {question}")

# Construct the full chat prompt from system & example questions & new human question
prompt = ChatPromptTemplate.from_messages([
system_prompt,
*example_messages,
SystemMessagePromptTemplate.from_template("Now answer the next question using only the provided context."),
human_prompt
])

# Use a HuggingFace embedding model to embed user questions and documents
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load pre-built FAISS vectorstore with the embedded documents
vectorstore = FAISS.load_local(
"faiss_index", # Local directory containing FAISS data
embeddings, # Embedding function used during index creation
allow_dangerous_deserialization=True # Required to load from disk safely
)

# Convert FAISS vector store into a retriever that fetches top-k relevant docs
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Create a text generation pipeline using HuggingFace Transformers
pipe = pipeline(
"text-generation",
model="deepseek-ai/deepseek-llm-7b-chat", # The LLM to generate answers
device=0, # Use GPU 0
torch_dtype=torch.float16, # Use float16 for performance
max_new_tokens=256, # Limit output length
do_sample=False, # Disable random sampling (deterministic output)
temperature=0, # Make output more focused and less diverse
return_full_text=False # Only return the generated part
)

model_name = "deepseek-ai/deepseek-llm-7b-chat"

# Wrap the HuggingFace pipelne in a LangChain-compatible LLM object
llm = Ollama(model="llama3", temperature=0) # Deterministic output

# Build the RAG chain: prompt -> model -> output
rag_chain = (
prompt
| llm
| StrOutputParser() # Converts raw LLM output into a string
)

# Questions to test the system
questions = [
"What is the credit value of CHEM2112?",
"How many second-year modules are available?",
"Are there any first-year modules assessed by 100% coursework?",
"Summarise what topics are covered across all first-year chemistry modules.",
"Are there any modules specifically about nanotechnology?",
"How do you know which modules are second-year modules?"
]

# Loop over each example question
for idx, question in enumerate(questions, start=1):
    print(f"\n Question {idx}: {question}")

    # Retrieve relevant documents from the vectorstore
    docs = retriever.invoke(question)

    # Join all the retrieved document content into a single context string
    context = "\n".join([doc.page_content for doc in docs])
    # Run the full RAG pipeline: provide prompt with context & question
    answer = rag_chain.invoke({"question": question, "context": context})

    # Print the model's answer
    print(answer.strip())