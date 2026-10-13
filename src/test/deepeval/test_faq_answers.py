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
from src.chat.index_faq import FAQ_DIR, extract_faqs


MODEL = OllamaModel(
    model="llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0,
    generation_kwargs={"num_ctx": 8192},
)

FAQ_SCENARIOS = [
    pytest.param(
        "02-login-no-app-nubank",
        "Minha senha e biometria estao corretas, mas ainda não consigo entrar "
        "no app. O que eu faço?",
        "Não consigo entrar no app mesmo com senha correta",
        id="login",
    ),
    pytest.param(
        "03-seguranca-no-nubank",
        "Como ativar o Modo Rua?",
        "Como ativo o Modo Rua para proteger minhas transferências fora de casa?",
        id="security-street-mode",
    ),
    pytest.param(
        "07-pix-no-nubank",
        "Como criar um Pix Agendado?",
        "Como configuro um Pix recorrente pelo aplicativo?",
        id="scheduled-pix",
    ),
    pytest.param(
        "08-cartao-de-credito-nubank",
        "Como negociar e pagar minha dívida do Cartão Nubank?",
        "Como faço para negociar uma dívida do meu cartão Nubank?",
        id="credit-card-debt",
    ),
    pytest.param(
        "09-emprestimos-nubank",
        "Em quanto tempo eu recebo o dinheiro do Empréstimo Consignado? "
        "Em qual conta o valor será depositado?",
        "Depois de contratar um empréstimo consignado, quando recebo o dinheiro?",
        id="consigned-loan",
    ),
    pytest.param(
        "01-o-nubank",
        "Como entro em contato com o Nubank?",
        "Como posso falar com o atendimento do Nubank?",
        id="customer-support",
    )
]


@pytest.fixture(scope="module")
def faq_collection():
    return get_collection()


@pytest.mark.parametrize(
    ("topic", "source_question", "user_question"),
    FAQ_SCENARIOS,
)
def test_chat_answers_faq_question(
    faq_collection,
    topic: str,
    source_question: str,
    user_question: str,
):
    source_path = FAQ_DIR / topic / "faq.md"
    source_faq = next(
        (
            faq
            for faq in extract_faqs(source_path)
            if faq["question"] == source_question
        ),
        None,
    )
    assert source_faq is not None, (
        f"FAQ de referência não encontrada: {topic} / {source_question}"
    )

    answer, retrieved_faqs = answer_question(faq_collection, user_question)
    retrieved_questions = {faq["question"] for faq in retrieved_faqs}
    assert source_faq["question"] in retrieved_questions, (
        f"A FAQ esperada não foi recuperada para: {user_question}"
    )

    test_case = LLMTestCase(
        input=user_question,
        actual_output=answer,
        expected_output=source_faq["answer"],
        retrieval_context=[faq["document"] for faq in retrieved_faqs],
    )
    metric = AnswerRelevancyMetric(
        threshold=0.8,
        model=MODEL,
        include_reason=True,
    )

    assert_test(test_case, [metric], run_async=False)
