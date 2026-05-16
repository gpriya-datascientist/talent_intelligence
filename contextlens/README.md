# ContextLens 🔬

> **Full Transformer Architecture** — how the same word gets a different vector in different contexts

Powered by the **Grok API** (xAI). Visualizes contextual word embeddings across transformer layers.

## What it does

Enter two sentences containing the same focus word (e.g. *"learning"*):

| | Sentence | Domain |
|---|---|---|
| A | "The student showed rapid **learning** after tutoring" | Education |
| B | "The **learning** rate controls model updates" | ML / AI |

ContextLens shows:
- How the word is **identical** at the embedding layer
- How transformer layers **progressively shift** the vector
- Final **cosine similarity** + **shift score** between the two output vectors
- **Attention heatmaps** per sentence
- **Per-layer analysis** (syntax → semantic → domain → final)

## Quick start

```bash
cd contextlens

# Install
pip install -e .

# Configure
cp .env.example .env
# Edit .env — add your GROK_API_KEY from console.x.ai

# Run
streamlit run app.py
# → http://localhost:8501
```

## Project structure

```
app.py          Streamlit dashboard (main entry)
src/
  grok_client.py   Grok API call + prompt engineering
  visualizer.py    All Plotly charts
  examples.py      7 built-in word pairs for demo
```

## Built-in examples

| Focus word | Sentence A | Sentence B |
|---|---|---|
| learning | student tutoring | ML learning rate |
| model | fashion runway | language model |
| bank | financial deposit | river bank |
| right | direction | correct answer |
| python | snake | programming language |
| network | professional | neural network |
| sprint | running | agile sprint |

## Stack

Grok API (xAI) · Streamlit · Plotly · OpenAI SDK (compatible client)
