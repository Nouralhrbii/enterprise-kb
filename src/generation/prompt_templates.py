"""
prompt_templates.py
-------------------
Source-aware prompts that adapt based on which source types are in context.

Why source-aware prompts?
A changelog chunk needs a different framing than a support ticket.
Telling the model "the following are changelog entries" helps it
understand the format and cite versions rather than filenames.
"""

SYSTEM_PROMPT_BASE = """\
You are a helpful enterprise knowledge base assistant. Answer the user's
question using ONLY the context passages provided below.

Each passage is labelled with its source type and identifier:
  [doc]       — reference documentation
  [ticket]    — resolved support ticket
  [changelog] — product changelog entry

Rules:
1. Answer only from the provided context. Do not use outside knowledge.
2. If the context does not contain enough information, say:
   "I don't have enough information in the knowledge base to answer that."
3. Always cite your sources. After each claim add a parenthetical:
   (Source: <filename>, type: <source_type>)
4. For changelog questions, always include the version number in your answer.
5. For ticket questions, note if the ticket is resolved or still open.
6. Keep your answer concise and focused.
7. Never invent facts not present in the context.

Context:
{context}
"""

USER_PROMPT = "Question: {question}"


def build_context_block(chunks) -> str:
    """Format retrieved chunks with source type labels."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source_type = getattr(chunk, "source_type", "unknown")
        source = chunk.source

        # Add extra metadata label based on source type
        if source_type == "ticket":
            subject = chunk.metadata.get("subject", "")
            status = chunk.metadata.get("ticket_status", "")
            label = f"[ticket] {source} — {subject} ({status})"
        elif source_type == "changelog":
            version = chunk.metadata.get("version", "")
            date = chunk.metadata.get("release_date", "")
            label = f"[changelog] v{version} ({date})"
        else:
            section = chunk.metadata.get("section", "")
            label = f"[doc] {source}" + (f" — {section}" if section else "")

        parts.append(f"[{i}] {label}\n{chunk.content}")

    return "\n\n---\n\n".join(parts)


def build_messages(question: str, chunks) -> list[dict]:
    context = build_context_block(chunks)
    return [
        {"role": "system", "content": SYSTEM_PROMPT_BASE.format(context=context)},
        {"role": "user", "content": USER_PROMPT.format(question=question)},
    ]