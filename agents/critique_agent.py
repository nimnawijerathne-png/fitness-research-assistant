from groq import Groq
from agents.messages import DraftAnswer, FinalAnswer
import config


CRITIQUE_SYSTEM_PROMPT = """You are a strict fact-checking reviewer. You
will be shown a draft answer and the research context it was based on.
List any claims in the draft that are NOT actually supported by the
context (one short line per claim). If everything is well supported,
respond with exactly: NONE."""


class CritiqueAgent:
    def __init__(self, client: Groq):
        self.client = client

    def review(self, draft: DraftAnswer) -> FinalAnswer:
        response = self.client.chat.completions.create(
            model=config.RERANK_MODEL,
            messages=[
                {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{draft.context_block}\n\nDraft answer:\n{draft.answer}",
                },
            ],
            temperature=0,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        flags = [] if raw.upper() == "NONE" else [
            line.strip("- ").strip() for line in raw.splitlines() if line.strip()
        ]

        answer = draft.answer
        if flags:
            answer += (
                "\n\n_Note: the following claims could not be fully "
                "verified against the retrieved sources:_\n- " + "\n- ".join(flags)
            )

        return FinalAnswer(
            answer=answer,
            sources=draft.sources,
            flags=flags,
            trace_id=draft.trace_id,
        )