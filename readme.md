# Local Bank AI Chat with DeepEval

This project implements a fictional banking FAQ assistant running locally, using:

- Python
- ChromaDB for vector search
- Ollama for local language and embedding models
- DeepEval for automated answer evaluation

The goal is to test a banking support chatbot that answers questions about Nubank using a Portuguese FAQ dataset without depending on external APIs.

## Overview

The project flow is:

1. Load FAQs from the `docs/faq_nubank_por_tema` directory
2. Generate local embeddings with Ollama
3. Index the documents in ChromaDB
4. Query the knowledge base through semantic retrieval
5. Generate answers with a local AI model
6. Validate answer quality with DeepEval tests

## Main structure

- `src/chat/index_faq.py` — indexes the FAQ in a local vector database
- `src/chat/chat.py` — runs the local RAG chatbot
- `src/test/deepeval/test_faq_answers.py` — automated tests using DeepEval
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

To run the full FAQ evaluation test suite:

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

This project is an example of a local RAG + evaluation workflow using AI in a development environment without depending on external providers. The focus is on validating the quality of assistant responses using DeepEval metrics against real FAQ questions.

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
