"""
src/groq_client.py  (replaces grok_client.py)
----------------------------------------------
Groq API client for ContextLens — uses groq.com free API.
Groq is OpenAI-SDK compatible via base_url swap.
"""
from __future__ import annotations
import json, os
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

ANALYSIS_PROMPT = """
You are ContextLens, an expert in transformer NLP and contextual word embeddings.

Analyze how the word "{focus_word}" gets different meanings in these two sentences:
Sentence A: "{sentence_a}"
Sentence B: "{sentence_b}"

Return ONLY valid JSON (no markdown, no text outside JSON):
{{
  "focus_word": "{focus_word}",
  "sentence_a": {{
    "text": "{sentence_a}",
    "tokens": ["list","of","tokens","with","##subwords"],
    "focus_token_index": 0,
    "context_keywords": ["word1","word2"],
    "domain": "e.g. Education",
    "meaning": "one sentence definition in this context",
    "hidden_state_mock": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    "layer_analysis": {{
      "layers_1_5": "Syntactic role?",
      "layers_6_8": "Semantic relationships?",
      "layers_9_11": "Domain encoded?",
      "layer_12": "Final contextual meaning"
    }}
  }},
  "sentence_b": {{
    "text": "{sentence_b}",
    "tokens": ["list","of","tokens"],
    "focus_token_index": 0,
    "context_keywords": ["word1","word2"],
    "domain": "e.g. ML / AI",
    "meaning": "one sentence definition in this context",
    "hidden_state_mock": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
    "layer_analysis": {{
      "layers_1_5": "Syntactic role?",
      "layers_6_8": "Semantic relationships?",
      "layers_9_11": "Domain encoded?",
      "layer_12": "Final contextual meaning"
    }}
  }},
  "embedding_comparison": {{
    "cosine_similarity": 0.0,
    "shift_score": 0.0,
    "are_identical_at_embedding_layer": true,
    "divergence_starts_at_layer": "layers_6_8",
    "key_differentiating_words_a": ["words","shifting","meaning"],
    "key_differentiating_words_b": ["words","shifting","meaning"],
    "attention_heatmap_a": [[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]],
    "attention_heatmap_b": [[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]],
    "explanation": "2-3 sentence explanation of why the vectors differ"
  }},
  "fun_fact": "One surprising thing about this word polysemy"
}}

hidden_state_mock: 8 realistic floats between -2.0 and 2.0, different per sentence.
attention_heatmap: one row of weights (0-1, sum~1) per token in that sentence.
cosine_similarity: 0=opposite 1=identical. shift_score = 1 - cosine_similarity.
"""

def analyze_context(
    sentence_a: str,
    sentence_b: str,
    focus_word: str,
    model: Optional[str] = None,
) -> dict:
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY", ""),
        base_url=GROQ_BASE_URL,
    )
    groq_model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    prompt = ANALYSIS_PROMPT.format(
        focus_word=focus_word,
        sentence_a=sentence_a,
        sentence_b=sentence_b,
    )
    response = client.chat.completions.create(
        model=groq_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2000,
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())
