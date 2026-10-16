# Local Bank AI RAG Chat with DeepEval

This project is a local **Retrieval-Augmented Generation (RAG)** experiment for a fictional banking FAQ assistant. It tests both retrieval of relevant FAQ content and the quality of the answer generated from that context, using:

- Python
- ChromaDB for vector search
- Ollama for local language and embedding models
- DeepEval for automated answer evaluation

The goal is to evaluate a banking support chatbot that answers questions about Nubank using a Portuguese FAQ dataset, without depending on external AI APIs.

## Overview

The project flow is:

1. Load FAQs from the `docs/faq_nubank_por_tema` directory
2. Generate local embeddings with Ollama
3. Index the documents in ChromaDB
4. Query the knowledge base through semantic retrieval
5. Generate answers with a local AI model
6. Evaluate retrieval and generated answers with DeepEval RAG metrics

The evaluation is focused on the RAG pipeline—not only on whether the final answer sounds relevant. It checks the retrieved context as well as whether the generated answer is supported by that context.

## Main structure

- `src/chat/index_faq.py` — indexes the FAQ in a local vector database
- `src/chat/chat.py` — runs the local RAG chatbot
- `src/test/deepeval/test_faq_answers.py` — RAG tests using DeepEval metrics
- `docs/faq_nubank_por_tema/` — FAQ corpus organized by topic
- `data/vector_store/` — local ChromaDB storage

## Requirements

- Python 3.10+ (recommended 3.11 or 3.12)
- Git
- Internet access to download Ollama and the models

## 1) Clone the project

```bash
cd ~/GitHub

git clone <repository-url>
cd bank-AI-assistant-testing
```

## 2) Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Install Ollama

On Linux, the easiest way is to use the official install script:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

You can also follow the official Ollama documentation for your operating system if needed.

## 4) Start Ollama locally

Open one terminal and run:

```bash
ollama serve
```

In another terminal, download the required models:

```bash
ollama pull llama3.2:3b
ollama pull embeddinggemma
```

Check that the models are installed:

```bash
ollama list
```

## 5) Prepare the FAQ vector index

Before chatting with the bot or running the tests, generate the local index:

```bash
cd ~/GitHub/bank-AI-assistant-testing
source .venv/bin/activate
python src/chat/index_faq.py
```

> This command creates the vector database in `data/vector_store`.

## 6) Run the chatbot locally

```bash
cd ~/GitHub/bank-AI-assistant-testing
source .venv/bin/activate
python src/chat/chat.py
```

The program starts an interactive terminal session. You can test questions like:

```text
How do I activate Street Mode?
How do I create a scheduled Pix transfer?
I can't log in even though my password is correct
```

## 7) Configure DeepEval to use Ollama

This project uses the `llama3.2:3b` model from Ollama as the evaluation model:

```bash
cd ~/GitHub/bank-AI-assistant-testing
source .venv/bin/activate
deepeval set-ollama --model=llama3.2:3b
```

It is also useful to verify the configuration:

```bash
deepeval diagnose
```

## 8) Run the project tests

The parameterized test scenarios evaluate each question-answer interaction with these DeepEval metrics:

- **Answer relevancy** — whether the generated answer addresses the user's question
- **Faithfulness** — whether claims in the answer are supported by the retrieved context
- **Contextual relevancy** — whether the retrieved FAQ context is relevant to the question
- **Contextual precision** — whether relevant context is ranked ahead of irrelevant context
- **Contextual recall** — whether the retrieved context contains the information needed by the reference answer

The tests also assert that the expected FAQ was retrieved. To run the full RAG evaluation suite:

```bash
cd ~/GitHub/bank-AI-assistant-testing
source .venv/bin/activate
deepeval test run src/test/deepeval/test_faq_answers.py
```

If you want to run only one specific scenario, use the pytest test identifier:

```bash
deepeval test run src/test/deepeval/test_faq_answers.py -k login
```

If DeepEval is slow or times out on a given case, increase the timeout per attempt:

```bash
cd ~/GitHub/bank-AI-assistant-testing
source .venv/bin/activate
DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=240 deepeval test run src/test/deepeval/test_faq_answers.py
```

## 9) Usage tips

- Keep Ollama running in a separate terminal with `ollama serve`
- If local execution fails because a model is missing, run:

```bash
ollama pull llama3.2:3b
ollama pull embeddinggemma
```

- To stop the Ollama service on Linux:

```bash
sudo systemctl stop ollama
```

## Notes

This project is an example of a local **RAG (Retrieval-Augmented Generation) system** and its evaluation. DeepEval checks both retrieval quality and answer quality against real questions from the FAQ corpus. In particular, contextual precision and recall help evaluate retrieval, while faithfulness checks grounding and answer relevancy checks whether the response addresses the question. All inference and evaluation use local Ollama models.

## Recommended execution flow

```bash
cd ~/GitHub/bank-AI-assistant-testing
source .venv/bin/activate

ollama serve
# in another terminal:
ollama pull llama3.2:3b
ollama pull embeddinggemma

python src/chat/index_faq.py
deepeval set-ollama --model=llama3.2:3b
deepeval test run src/test/deepeval/test_faq_answers.py
```
