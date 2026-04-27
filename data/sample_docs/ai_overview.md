# Artificial Intelligence: An Overview

## What is Artificial Intelligence?

Artificial Intelligence (AI) refers to the simulation of human intelligence processes by computer systems. These processes include learning (the acquisition of information and rules for using the information), reasoning (using rules to reach approximate or definite conclusions), and self-correction.

## Machine Learning

Machine Learning (ML) is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. ML focuses on the development of computer programs that can access data and use it to learn for themselves.

Key ML approaches:
- **Supervised Learning**: Models trained on labeled datasets to predict outputs for new inputs.
- **Unsupervised Learning**: Models that find hidden patterns in data without pre-existing labels.
- **Reinforcement Learning**: Agents learn by interacting with an environment and receiving rewards or penalties.

## Deep Learning

Deep Learning is a subset of machine learning that uses neural networks with many layers (hence "deep"). These networks can automatically learn representations from raw data.

### Transformer Architecture

The Transformer architecture, introduced in the 2017 paper "Attention Is All You Need," revolutionized natural language processing. Key innovations:
- Self-attention mechanisms allow the model to weigh the relevance of different words in context.
- Parallel processing replaces sequential RNN computations.
- Positional encodings allow models to understand word order.

## Large Language Models (LLMs)

Large Language Models are transformer-based models trained on vast corpora of text. Notable examples:
- GPT series (OpenAI)
- Claude (Anthropic)
- Gemini (Google)
- LLaMA (Meta)

LLMs exhibit emergent capabilities such as few-shot learning, chain-of-thought reasoning, and instruction following.

## Retrieval-Augmented Generation (RAG)

RAG combines the parametric knowledge of LLMs with non-parametric retrieval from external knowledge bases. The process:
1. The query is encoded into a vector embedding.
2. Semantically similar documents are retrieved from a vector store.
3. Retrieved context is injected into the LLM prompt.
4. The LLM generates a grounded, context-aware response.

RAG significantly reduces hallucination compared to pure LLM generation.
