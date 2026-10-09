from pathlib import Path

import chromadb
import ollama


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTOR_DIR = PROJECT_ROOT / "data" / "vector_store"

CHAT_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "embeddinggemma"
COLLECTION_NAME = "faq_nubank"

TOP_K = 3
MIN_SIMILARITY = 0.35


SYSTEM_PROMPT = """
Você é um assistente virtual que responde dúvidas sobre produtos e serviços
do Nubank.

Use somente as informações contidas no CONTEXTO fornecido.

Regras:
- Não invente taxas, prazos, regras, políticas ou procedimentos.
- Não use conhecimento externo.
- Não dê recomendações financeiras personalizadas.
- Se o contexto não tiver informação suficiente, diga claramente que não
  encontrou a resposta na base de FAQ e sugira consultar os canais oficiais
  do Nubank.
- Responda em português brasileiro, de forma clara, objetiva e cordial.
"""


def get_collection():
    if not VECTOR_DIR.exists():
        raise FileNotFoundError(
            "Índice vetorial não encontrado. Execute primeiro:\n"
            "python src/index_faq.py"
        )

    client = chromadb.PersistentClient(path=str(VECTOR_DIR))

    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )


def retrieve_faqs(collection, question: str) -> list[dict]:
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=question,
    )

    query_embedding = response["embeddings"][0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity = 1 - distance

        retrieved.append(
            {
                "document": document,
                "topic": metadata["topic"],
                "question": metadata["question"],
                "source": metadata["source"],
                "similarity": similarity,
            }
        )

    return retrieved


def build_context(retrieved: list[dict]) -> str:
    sections = []

    for index, item in enumerate(retrieved, start=1):
        sections.append(
            f"[FAQ {index}]\n"
            f"Fonte: {item['source']}\n"
            f"{item['document']}"
        )

    return "\n\n".join(sections)


def answer_question(collection, question: str) -> tuple[str, list[dict]]:
    retrieved = retrieve_faqs(collection, question)

    if not retrieved or retrieved[0]["similarity"] < MIN_SIMILARITY:
        return (
            "Não encontrei uma resposta confiável para essa pergunta na base "
            "de perguntas frequentes do Nubank. Consulte os canais oficiais "
            "de atendimento para obter orientação.",
            retrieved,
        )

    context = build_context(retrieved)

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"CONTEXTO:\n\n{context}\n\n"
                    f"PERGUNTA DO USUÁRIO:\n{question}"
                ),
            },
        ],
        options={
            "temperature": 0.1,
        },
    )

    return response["message"]["content"], retrieved


def print_sources(retrieved: list[dict]) -> None:
    print("\nFontes recuperadas:")

    for item in retrieved:
        print(
            f"- [{item['similarity']:.2f}] "
            f"{item['topic']} — {item['question']}"
        )


def main():
    collection = get_collection()

    print("Chatbot RAG do FAQ Nubank")
    print("Digite 'sair' para encerrar.\n")

    while True:
        question = input("Você: ").strip()

        if not question:
            continue

        if question.lower() in {"sair", "exit", "quit"}:
            print("Chatbot encerrado.")
            break

        answer, retrieved = answer_question(collection, question)

        print(f"\nAssistente: {answer}")
        print_sources(retrieved)
        print()


if __name__ == "__main__":
    main()