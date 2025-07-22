# Imports
from langchain_community.document_loaders import PlaywrightURLLoader, TextLoader # For loading content from URLs or local text files
from langchain.text_splitter import RecursiveCharacterTextSplitter # To split large documents into smaller chunks
from langchain_community.vectorstores import FAISS # Vector database for efficient similarity search
from langchain_ollama import OllamaEmbeddings # Embedding model from Ollama for text vectorisation
import re # Regex library for text cleaning
from langchain.schema import Document

# === CLEANING FUNCTIONS ===

# Removed common UI and navigation boilerplate from web pages
def remove_navigation_blocks(text):
    NAV_PHRASES = ["Home", "Undergraduates", "Postgraduates", "Module Catalogue", "Programme Catalogue", "Disclaimer", "Glossary", "View Timetable", "Footer navigation", "Quicklinks and contacts", "Site information", "Coronavirus", "Telephone", "Privacy and cookies"]
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        if line.strip() in NAV_PHRASES:
            continue # Skip known navigation phrases
        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    # Collapse multiple empty lines into a single one
    cleaned_text = re.sub(r"\n\s*\n+", "\n\n", cleaned_text)
    return cleaned_text.strip()

# Redacts email addresses and phone numbers using regex
def remove_contact_info(text):
    text = re.sub(r'\S+@\S+', ' ', text) # Remove emails
    text = re.sub(r'\+?\d[\d\s\-]{6,}', ' ', text) # Remove phone numbers
    return text

# Removes duplicate lines based on hashed content
def deduplicate_lines(text):
    seen = set()
    new_lines = []
    for line in text.splitlines():
        h = hash(line.strip())
        if h not in seen:
            seen.add(h)
            new_lines.append(line)
    return "\n".join(new_lines)

# Strips leading/trailing whitespace and collapses extra line breaks
def normalize_whitespace(text):
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    text = "\n".join([line.strip() for line in text.splitlines()])
    return text

# Main function to apply all cleaning steps to raw HTML or text content
def aggressive_clean(text):
    text = remove_navigation_blocks(text)
    # text = deduplicate_lines(text)
    text = remove_contact_info(text)
    text = normalize_whitespace(text)
    return text

# Cleans and sanitises URLs read from file
def clean_urls(raw_urls):
    cleaned = []
    for u in raw_urls:
        u = u.strip()
        u = u.encode("utf-8", "ignore").decode("utf-8") # Remove non-UTF-8 characters
        u = "".join(c for c in u if ord(c) > 31 and ord(c) < 127) # Keep only printable ASCII
        cleaned.append(u)
    return cleaned

# === INGEST PIPELINE ===

# Step 1: Load URLs from file and clean them
with open("moduleurls.txt", "r", encoding="utf-8") as f:
    raw_urls = [line.strip() for line in f if line.strip()]
urls = clean_urls(raw_urls)

docs_nested = []

# Step 2: Load and clean each webpage using Playwright
for url in urls:
    try:
        loader = PlaywrightURLLoader([url])
        docs = loader.load()
        for d in docs:
            cleaned = aggressive_clean(d.page_content)
            d.page_content = cleaned
        docs_nested.append(docs)
        print(f"Loaded and cleaned: {url}")
    except Exception as e:
            print(f"Failed to load {url}: {e}")

# Also load and clean a local summary file containing structured module information
try:
    summary_docs = TextLoader("moduleoverviews.txt").load()
    for d in summary_docs:
        d.page_content = aggressive_clean(d.page_content)
    docs_nested.append(summary_docs)
    print("Loaded and cleaned structured summary file.")
except Exception as e:
    print(f"Failed to load moduleoverviews.txt: {e}")

# Step 3: Flatten the list of lists into one list of documents, filtering out any empty ones
docs_list = [doc for sublist in docs_nested for doc in sublist if doc.page_content and isinstance(doc.page_content, str) and doc.page_content.strip()]
print(f"Total documents loaded (non-empty): {len(docs_list)}")

# Step 4: Split long documents into smaller overlapping chunks - helps with more accurate semantic search and retrieval
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
chunk_size=250, chunk_overlap=30, separators=["\n\n", "\n", ".", " "] # max characters per chunk, no overlap between chunks
)

chunks = text_splitter.split_documents(docs_list)

print(f"Total chunks created: {len(chunks)}")

# Error handling: stop if nothing was processed
if not chunks:
    raise ValueError("No text chunks available after splitting and cleaning. Check cleaning logic or input documents.")

# Step 5: Embed each chunk using the specified Ollama embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Step 6: Create the FAISS vectorstore from the embedded chunks
print("Creating vector store...")

vectorstore = FAISS.from_documents(chunks, embeddings)

# Step 7: Save the vectorstore locally for later use in the chatbot
vectorstore.save_local("faiss_index")

print("Vectorstore saved successfully.")