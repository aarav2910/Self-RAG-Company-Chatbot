"""Context-grounded answer generation shared by Self-RAG and evaluations."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import traceable


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an internal company assistant that answers employee "
            "questions from company documents.\n\n"
            "Rules:\n"
            "1. Use ONLY information supported by the context. Never use "
            "outside knowledge or assumptions.\n"
            "2. Answer every part of the question that can be supported "
            "by the context. Lead with the direct answer and remain concise. "
            "Use as many sentences as necessary to cover the relevant details.\n"
            "3. Evaluate every passage independently for relevance to the "
            "question. Different passages may describe different aspects of "
            "the same question. Combine information from ALL relevant passages, "
            "including passages from different documents. Do not stop after "
            "finding one valid answer. For broad or ambiguous questions, cover "
            "the distinct relevant interpretations supported by the context.\n"
            "4. Preserve numbers, dates, amounts, durations, names, and "
            "conditions exactly as stated. Include relevant eligibility "
            "requirements, exceptions, limitations, and authorization "
            "requirements. Keep exceptions attached to the rules they modify.\n"
            "5. Before answering, identify the distinct parts of the question. "
            "Answer every part supported by the context. If only some parts "
            "can be answered, answer those parts and briefly identify what "
            "information is missing. Never discard a supported answer just "
            "because another part cannot be answered.\n"
            "6. If no relevant information is available to answer the "
            "question, reply exactly: No answer found\n"
            "7. If relevant passages contradict each other, clearly state "
            "the contradiction and present both versions without choosing one.\n"
            "8. Do not mention the context, passages, or documents in "
            "your answer."
        ),
        ("human", "Question:\n{question}\n\nContext:\n{context}"),
    ]
)


@traceable(run_type="chain", name="generator")
def generate(query: str, context: list[str] | str) -> str:
    """Generate an answer from supplied context without running retrieval."""
    if isinstance(context, str):
        context_text = context.strip()
    else:
        context_text = "\n\n---\n\n".join(context).strip()
    if not context_text:
        return "No answer found."

    response = llm.invoke(
        generation_prompt.format_messages(question=query, context=context_text)
    )
    return response.content
