# --- Model selection ---
ROUTER_MODEL = "llama-3.1-8b-instant"          # fast, cheap — used for classifying queries
SYNTHESIS_MODEL = "llama-3.3-70b-versatile"     # stronger reasoning — used for the final answer
RERANK_MODEL = "llama-3.1-8b-instant"           # fast — used by the critique agent

# --- RAG pipeline settings ---
CORPUS_DIR = "data/corpus"
CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "fitness_research"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 50        # overlap between chunks so context isn't cut off mid-sentence
TOP_K = 5                 # how many chunks to retrieve per question