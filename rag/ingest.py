import os
import glob
import chromadb
from chromadb.utils import embedding_functions
import config


def load_documents(corpus_dir: str):
    docs = []
    paths = glob.glob(os.path.join(corpus_dir, "**/*.txt"), recursive=True) + \
            glob.glob(os.path.join(corpus_dir, "**/*.md"), recursive=True)
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append({"source": os.path.basename(path), "text": text})
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


def main():
    docs = load_documents(config.CORPUS_DIR)
    if not docs:
        print(f"No documents found in {config.CORPUS_DIR}. Add some .txt/.md files first.")
        return

    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME, embedding_function=embed_fn
    )

    ids, texts, metadatas = [], [], []
    chunk_counter = 0
    for doc in docs:
        for chunk in chunk_text(doc["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP):
            ids.append(f"chunk_{chunk_counter}")
            texts.append(chunk)
            metadatas.append({"source": doc["source"]})
            chunk_counter += 1

    collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
    print(f"Ingested {len(docs)} documents -> {chunk_counter} chunks into '{config.COLLECTION_NAME}'")


if __name__ == "__main__":
    main()