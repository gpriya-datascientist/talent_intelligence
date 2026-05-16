"""
src/grok_client.py
------------------
Grok API client — sends two sentences to grok-3 and asks it to:
1. Tokenize both sentences
2. Return contextual embeddings (hidden states) for the focus word
3. Explain what context is shifting the meaning
"""
from __future__ import annotations

import json
import os
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def _get_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("GROK_API_KEY", ""),
        base_url=os.getenv("GROK_BASE_URL", "https://api.x.ai/v1"),
    )


ANALYSIS_PROMPT = """
You are ContextLens, an expert in transformer NLP and contextual word embeddings.

Analyze how the word "{focus_word}" gets different meanings in these two sentences:

Sentence A: "{sentence_a}"
Sentence B: "{sentence_b}"

Return ONLY a valid JSON object (no markdown, no explanation outside JSON) with this exact structure:

{{
  "focus_word": "{focus_word}",
  "sentence_a": {{
    "text": "{sentence_a}",
    "tokens": ["list", "of", "tokens", "with", "##subwords"],
    "focus_token_index": 0,
    "context_keywords": ["word1", "word2"],
    "domain": "e.g. Education / ML / Finance",
    "meaning": "one sentence definition in this context",
    "hidden_state_mock": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "layer_analysis": {{
      "layers_1_5": "What syntactic role does the word play here?",
      "layers_6_8": "What semantic relationships form here?",
      "layers_9_11": "What domain/topic is encoded?",
      "layer_12": "Final contextual meaning"
    }}
  }},
  "sentence_b": {{
    "text": "{sentence_b}",
    "tokens": ["list", "of", "tokens", "with", "##subwords"],
    "focus_token_index": 0,
    "context_keywords": ["word1", "word2"],
    "domain": "e.g. Education / ML / Finance",
    "meaning": "one sentence definition in this context",
    "hidden_state_mock": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "layer_analysis": {{
      "layers_1_5": "What syntactic role does the word play here?",
      "layers_6_8": "What semantic relationships form here?",
      "layers_9_11": "What domain/topic is encoded?",
      "layer_12": "Final contextual meaning"
    }}
  }},
  "embedding_comparison": {{
    "cosine_similarity": 0.0,
    "shift_score": 0.0,
    "are_identical_at_embedding_layer": true,
    "divergence_starts_at_layer": "layers_6_8",
    "key_differentiating_words_a": ["words", "in", "A", "that", "shift", "meaning"],
    "key_differentiating_words_b": ["words", "in", "B", "that", "shift", "meaning"],
    "attention_heatmap_a": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
    "attention_heatmap_b": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
    "explanation": "2-3 sentence explanation of why the vectors differ"
  }},
  "fun_fact": "One surprising or interesting thing about this word's polysemy"
}}

For hidden_state_mock: generate 8 realistic float values that represent how this word's
embedding vector would look (use values between -2.0 and 2.0, different for each sentence
to show the contextual shift).

For attention_heatmap: generate one row of attention weights (floats 0–1, sum ≈ 1.0)
with one value per token in that sentence.

For cosine_similarity: estimate how similar the two contextual vectors would be (0=opposite, 1=identical).
For shift_score: 1 - cosine_similarity.
"""


def analyze_context(
    sentence_a: str,
    sentence_b: str,
    focus_word: str,
    model: Optional[str] = None,
) -> dict:
    """
    Call Grok API and return structured ContextLens analysis.
    """
    client = _get_client()
    grok_model = model or os.getenv("GROK_MODEL", "grok-3")

    prompt = ANALYSIS_PROMPT.format(
        focus_word=focus_word,
        sentence_a=sentence_a,
        sentence_b=sentence_b,
    )

    response = client.chat.completions.create(
        model=grok_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2000,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)
