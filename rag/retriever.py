import chromadb
from chromadb.utils import embedding_functions
import config


class Retriever:
    def __init__(self):
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL
        )
        client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        self.collection = client.get_or_create_collection(
            name=config.COLLECTION_NAME, embedding_function=embed_fn
        )

    def query(self, text: str, top_k: int = None):
        top_k = top_k or config.TOP_K
        results = self.collection.query(query_texts=[text], n_results=top_k)

        out = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        for doc, meta in zip(docs, metas):
            out.append({"text": doc, "source": meta.get("source", "unknown")})
        return out