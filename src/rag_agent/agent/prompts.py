"""
prompts.py
==========
All LLM prompt templates for the RAG interview preparation agent.

Prompts are defined here as module-level constants so they can be
imported by nodes.py and tested independently of the full agent.

The Prompt Engineer owns this file. Document every design decision —
you will be asked to defend these choices in Hour 3.

PEP 8 | Single Responsibility
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a deep learning interview preparation assistant.

You MUST follow these rules strictly:

1. Answer ONLY using the provided context.
2. DO NOT use any external knowledge.
3. If the context does not contain enough information, say:
   "I don't have enough information in the provided context."
4. ALWAYS include source citations in this format:
   [SOURCE: topic | filename]
5. Be concise, clear, and technical.
6. Do NOT hallucinate or guess.

Your goal is to help users prepare for technical interviews using ONLY the study material provided.
"""

# ---------------------------------------------------------------------------
# Query Rewriting Prompt
# ---------------------------------------------------------------------------

QUERY_REWRITE_PROMPT = """You are a search query optimizer for a deep learning \
knowledge base.

Rewrite the following natural language question into a short, keyword-dense \
search query that will produce better vector similarity matches.

Rules:
- Output only the rewritten query, nothing else
- Use technical terminology from deep learning
- Remove conversational filler words
- Expand abbreviations (e.g. "RNN" → "recurrent neural network RNN")
- Include related concepts that might appear in a relevant document
- Maximum 15 words

Original question: {original_query}

Rewritten query:"""

# ---------------------------------------------------------------------------
# Question Generation Prompt
# ---------------------------------------------------------------------------

QUESTION_GENERATION_PROMPT = """You are generating a technical interview \
question for a deep learning candidate.

Use the following source material to generate ONE interview question.

SOURCE MATERIAL:
{context}

DIFFICULTY LEVEL: {difficulty}

Generate a question that:
- Requires genuine understanding, not just recall
- Is open-ended (cannot be answered with yes/no)
- Connects at least two concepts from the source material if possible
- Is appropriate for the specified difficulty level

Respond with a JSON object in exactly this format:
{{
    "question": "the interview question",
    "difficulty": "{difficulty}",
    "topic": "primary topic tested",
    "model_answer": "a complete, accurate model answer drawn from the source material",
    "follow_up": "one follow-up question to probe deeper understanding",
    "source_citations": ["[SOURCE: topic | filename]"]
}}

Respond with the JSON object only. No preamble or explanation."""

# ---------------------------------------------------------------------------
# Answer Evaluation Prompt
# ---------------------------------------------------------------------------

ANSWER_EVALUATION_PROMPT = """You are evaluating a candidate's answer to a \
technical deep learning interview question.

QUESTION: {question}

CANDIDATE'S ANSWER: {candidate_answer}

SOURCE MATERIAL (ground truth):
{context}

Evaluate the candidate's answer against the source material.

Respond with a JSON object in exactly this format:
{{
    "score": <integer 0-10>,
    "what_was_correct": "specific aspects the candidate got right",
    "what_was_missing": "concepts or details that were absent or incorrect",
    "ideal_answer": "a complete model answer drawn strictly from the source material",
    "interview_verdict": "hire / consider / no hire based on this answer alone",
    "coaching_tip": "one specific thing the candidate should study before their interview"
}}

Scoring guide:
- 9-10: Complete, accurate, well-articulated. Ready for senior roles.
- 7-8: Mostly correct with minor gaps. Good junior to mid-level candidate.
- 5-6: Core concept understood but significant details missing.
- 3-4: Partial understanding, notable misconceptions present.
- 0-2: Fundamental misunderstanding or no relevant knowledge demonstrated.

Respond with the JSON object only. No preamble or explanation."""

# ---------------------------------------------------------------------------
# Hallucination Guard Message
# ---------------------------------------------------------------------------

NO_CONTEXT_RESPONSE = """I was unable to find relevant information in the \
study corpus for your query.

This may mean:
- The topic is not yet covered in the corpus (check if it is a bonus topic)
- Your query needs to be more specific (try including the exact topic name)
- The corpus needs more content on this area

Suggested next steps:
- Rephrase your query with specific deep learning terminology
- Check which topics are available using the corpus browser
- If you are the Corpus Architect, consider adding content on this topic

Topics currently available: ANN, CNN, RNN, LSTM, Seq2Seq, Autoencoder
Bonus topics (if ingested): SOM, Boltzmann Machines, GAN"""
