import sys
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.chat.chat import answer_question, get_collection


model = OllamaModel(
    model="llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0,
)

## TESTE SIMPLES PARA VALIDAR O COMPORTAMENTO DO MODELO COM UMA PERGUNTA DE LOGIN


@pytest.fixture(scope="module")
def login_test_case():
    question = "Não consigo entrar no app mesmo com senha correta"
    answer, retrieved_faqs = answer_question(get_collection(), question)

    return LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=[faq["document"] for faq in retrieved_faqs],
    )


def test_login_response_relevancy(login_test_case):
    metric = AnswerRelevancyMetric(threshold=0.5, model=model)
    
    print(f"\nPergunta:\n{login_test_case.input}")
    print(f"\nResposta gerada:\n{login_test_case.actual_output}")
    print("\nContexto recuperado:")
    for i, trecho in enumerate(login_test_case.retrieval_context or [], start=1):
        print(f"\n--- Trecho {i} ---\n{trecho}")
    
    assert_test(login_test_case, [metric], run_async=False)
