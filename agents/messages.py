from dataclasses import dataclass, field
from typing import List
import uuid


@dataclass
class ResearchRequest:
    query: str
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class DraftAnswer:
    query: str
    answer: str
    sources: List[str]
    context_block: str
    trace_id: str


@dataclass
class FinalAnswer:
    answer: str
    sources: List[str]
    flags: List[str]
    trace_id: str