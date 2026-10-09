from collections import defaultdict
from pathlib import Path
import re
import shutil

import chromadb
import ollama


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FAQ_DIR = PROJECT_ROOT / "docs" / "faq_nubank_por_tema"
VECTOR_DIR = PROJECT_ROOT / "data" / "vector_store"

EMBEDDING_MODEL = "embeddinggemma"
COLLECTION_NAME = "faq_nubank"


def extract_faqs(file_path: Path) -> list[dict]:
    """Extrai cada bloco Pergunta/Resposta de um arquivo faq.md."""
    content = file_path.read_text(encoding="utf-8")

    pattern = re.compile(
        r"## Pergunta\s*\n+"
        r"(?P<question>.*?)\s*\n+"
        r"## Resposta\s*\n+"
        r"(?P<answer>.*?)(?=\n+---|\Z)",
        flags=re.DOTALL,
    )

    topic = file_path.parent.name
    records = []

    for index, match in enumerate(pattern.finditer(content), start=1):
        question = " ".join(match.group("question").split())
        answer = match.group("answer").strip()

        if not question or not answer:
            continue

        records.append(
            {
                "id": f"{topic}-{index:03d}",
                "topic": topic,
                "question": question,
                "answer": answer,
                "source": str(file_path.relative_to(PROJECT_ROOT)),
                "document": (
                    f"Tema: {topic}\n"
                    f"Pergunta: {question}\n"
                    f"Resposta: {answer}"
                ),
            }
        )

    return records


def load_faqs() -> list[dict]:
    records = []

    for file_path in sorted(FAQ_DIR.glob("*/faq.md")):
        records.extend(extract_faqs(file_path))

    return records


def create_index(records: list[dict]) -> None:
    if VECTOR_DIR.exists():
        shutil.rmtree(VECTOR_DIR)

    client = chromadb.PersistentClient(path=str(VECTOR_DIR))

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 64

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        texts = [record["document"] for record in batch]

        embedding_response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=texts,
        )

        collection.add(
            ids=[record["id"] for record in batch],
            documents=texts,
            embeddings=embedding_response["embeddings"],
            metadatas=[
                {
                    "topic": record["topic"],
                    "question": record["question"],
                    "source": record["source"],
                }
                for record in batch
            ],
        )

        print(
            f"Indexados {min(start + batch_size, len(records))}"
            f" de {len(records)} FAQs"
        )


def main():
    if not FAQ_DIR.exists():
        raise FileNotFoundError(f"Pasta de FAQ não encontrada: {FAQ_DIR}")

    records = load_faqs()

    if not records:
        raise RuntimeError("Nenhum par Pergunta/Resposta foi encontrado.")

    print(f"Foram encontrados {len(records)} pares de FAQ.")

    create_index(records)

    print("\nIndexação concluída.")
    print(f"Índice salvo em: {VECTOR_DIR}")


if __name__ == "__main__":
    main()