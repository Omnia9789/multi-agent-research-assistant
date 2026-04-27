# Hallucination in LLMs: Detection and Mitigation

## What is Hallucination?

In the context of Large Language Models, "hallucination" refers to the generation of content that is factually incorrect, fabricated, or not grounded in the provided context or training data. The term is borrowed by analogy from the psychological phenomenon.

## Types of Hallucination

### Intrinsic Hallucination
The generated output contradicts the source material provided in the prompt. This is particularly problematic in summarization tasks.

### Extrinsic Hallucination
The generated output introduces information that cannot be verified from the source material—it may be true or false, but it is not grounded.

### Factual Hallucination
The output states facts that are simply incorrect (e.g., wrong dates, wrong attributions, invented statistics).

## Why LLMs Hallucinate

1. **Training data noise**: The model learns from imperfect web data containing errors.
2. **Parametric knowledge limitations**: The model encodes world knowledge imperfectly in its weights.
3. **Overconfidence**: Models tend to generate fluent, confident text even when uncertain.
4. **Instruction following vs. accuracy trade-off**: Models optimized to be helpful may "fill in gaps."

## Detection Methods

### Model-Based Detection
Use a separate LLM to evaluate the faithfulness of generated text against source documents.

### Entailment-Based Methods
Check whether the generated claims are logically entailed by the source context using NLI (Natural Language Inference) models.

### Confidence Calibration
Measure token-level log probabilities as a proxy for model uncertainty.

### Self-Consistency
Generate multiple responses to the same query and measure variance. High variance correlates with hallucination risk.

## Mitigation Strategies

1. **Retrieval-Augmented Generation (RAG)**: Ground the model in retrieved evidence.
2. **Chain-of-Thought Prompting**: Encourage step-by-step reasoning before committing to an answer.
3. **Constitutional AI**: Train models with explicit honesty constraints.
4. **Human Feedback (RLHF)**: Reward factual, calibrated responses during training.
5. **Post-hoc Fact-Checking**: A separate verification pass after generation.

## Confidence Scoring

A practical approach for production systems:
- Score 0.85+: High confidence, answer is likely accurate.
- Score 0.60–0.84: Moderate confidence, recommend human review for high-stakes decisions.
- Score below 0.60: Low confidence, flag the answer and surface sources for user verification.
