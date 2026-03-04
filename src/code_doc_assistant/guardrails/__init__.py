"""Content safety guardrails — no external service required.

Two-layer design
----------------
Layer 1 — fast regex pre-screen (zero latency, zero cost):
    Rejects well-known prompt-injection and jailbreak patterns before any
    LLM call is made.

Layer 2 — LLM safety classifier (one short GPT-4o call):
    Evaluates the user's message against the four Microsoft Content Safety
    harm categories (Hate, Sexual, Violence, Self-Harm) and detects indirect
    prompt-injection / off-topic abuse.  Returns a single-word verdict so the
    call adds minimal latency (~150–300 ms) and token cost.

Both layers return a :class:`GuardResult` that the FastAPI router inspects
before forwarding to the main assistant.
"""

from code_doc_assistant.guardrails.input_guard import GuardResult, InputGuard

__all__ = ["GuardResult", "InputGuard"]
