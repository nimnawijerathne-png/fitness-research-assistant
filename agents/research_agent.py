from groq import Groq
from rag.retriever import Retriever
from agents.messages import ResearchRequest, DraftAnswer
import config


SYNTHESIS_SYSTEM_PROMPT = """You are a fitness & lifestyle research
assistant. Answer the user's question using ONLY the provided research
context. If the context doesn't cover the question, say so honestly
instead of guessing. Cite which source each claim comes from using the
[source: ...] tags given in the context."""


class ResearchAgent:
    def __init__(self, client: Groq, retriever: Retriever):
        self.client = client
        self.retriever = retriever

    def handle(self, request: ResearchRequest) -> DraftAnswer:
        # --- Act: retrieve ---
        results = self.retriever.query(request.query, top_k=config.TOP_K)

        # --- Observe: if nothing useful came back, try a broader query once ---
        if not results:
            broadened = f"fitness lifestyle {request.query}"
            results = self.retriever.query(broadened, top_k=config.TOP_K)

        context_block = "\n\n".join(
            f"[source: {r['source']}]\n{r['text']}" for r in results
        )
        sources = sorted({r["source"] for r in results})

        # --- Act: synthesise ---
        response = self.client.chat.completions.create(
            model=config.SYNTHESIS_MODEL,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context_block}\n\nQuestion: {request.query}",
                },
            ],
            temperature=0.3,
            max_tokens=600,
        )
        answer = response.choices[0].message.content.strip()

        return DraftAnswer(
            query=request.query,
            answer=answer,
            sources=sources,
            context_block=context_block,
            trace_id=request.trace_id,
        )