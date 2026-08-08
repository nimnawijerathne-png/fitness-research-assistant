from groq import Groq
from agents.messages import DraftAnswer, FinalAnswer
import config


CRITIQUE_SYSTEM_PROMPT = """You are a fact-checking reviewer for a fitness
research assistant. You will be shown a draft answer and the research
context it was based on.

Flag a claim ONLY if it introduces information, numbers, or conclusions
that are genuinely absent from or contradicted by the context — not if
it merely paraphrases, summarizes, or rewords something the context
already says. Minor rephrasing is NOT a violation.

List each genuinely unsupported claim as one short line. If everything
is adequately supported (including reasonable paraphrases), respond
with exactly: NONE.

Be conservative: when in doubt, do not flag it."""


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