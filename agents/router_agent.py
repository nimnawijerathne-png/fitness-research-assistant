from groq import Groq
import config


ROUTER_SYSTEM_PROMPT = """You are a query router for a fitness & lifestyle
research assistant. Classify the user's message into exactly one label:

- RESEARCH: a question about training, nutrition, sleep, recovery, or
  lifestyle that should be answered using research-backed sources.
- OUT_OF_SCOPE: anything unrelated to fitness/nutrition/lifestyle, or a
  request for individualised medical diagnosis/treatment.
- SMALL_TALK: greetings, thanks, chit-chat with no research question.

Respond with ONLY the label, nothing else."""


class RouterAgent:
    def __init__(self, client: Groq):
        self.client = client

    def route(self, user_query: str) -> str:
        response = self.client.chat.completions.create(
            model=config.ROUTER_MODEL,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": user_query},
            ],
            temperature=0,
            max_tokens=10,
        )
        label = response.choices[0].message.content.strip().upper()
        if label not in {"RESEARCH", "OUT_OF_SCOPE", "SMALL_TALK"}:
            label = "RESEARCH"  # safe default
        return label