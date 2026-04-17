
```
 ______     ______     __    __     ______     __     ______    
/\___  \   /\  __ \   /\ "-./  \   /\  == \   /\ \   /\  ___\   
\/_/  /__  \ \ \/\ \  \ \ \-./\ \  \ \  __<   \ \ \  \ \  __\   
  /\_____\  \ \_____\  \ \_\ \ \_\  \ \_____\  \ \_\  \ \_____\ 
  \/_____/   \/_____/   \/_/  \/_/   \/_____/   \/_/   \/_____/ 
                                                                  
     ______     __         __    
    /\  ___\   /\ \       /\ \   
    \ \ \____  \ \ \____  \ \ \  
     \ \_____\  \ \_____\  \ \_\ 
      \/_____/   \/_____/   \/_/ 
```

<div align="center">

# 🧟 ZOMBIE CLI

### *A thousand narrow specialists beat one generalist — always.*

[![Python](https://img.shields.io/badge/python-3.12+-blue?style=flat-square&logo=python)](https://python.org)
[![LiteLLM](https://img.shields.io/badge/litellm-unified_LLM_API-green?style=flat-square)](https://github.com/BerriAI/litellm)
[![LangGraph](https://img.shields.io/badge/langgraph-execution_graph-orange?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow?style=flat-square)](LICENSE)

**MSR · Model Specialist Router**

</div>

---

## What Is This?

Zombie CLI is a **multi-specialist AI routing engine** (MSR) that decomposes any user query into atomic subtasks, dispatches each to the world's best narrow model for that domain, cross-verifies the output, and synthesizes one clean final answer.

The pipeline that runs under the hood:

```
Query  →  Orchestrate  →  Dispatch  →  [Specialists run in parallel]
                                              │
                                         fan-in ──→  Verify  ──→  Synthesize  →  Answer
```

Rather than sending everything to one generalist, MSR routes:

| Domain | Model | Why |
|--------|-------|-----|
| **Code** | `claude-sonnet-4-6` | Best coding + long-context explanations |
| **Math** | `groq/deepseek-r1-distill-llama-70b` | Fastest chain-of-thought reasoning |
| **Research** | `perplexity/sonar-pro` | Search-grounded, live citations |
| **Summarize** | `gemini/gemini-2.5-flash` | 1M-token context window, cheap |
| **Structured** | `gpt-4o` | Rock-solid JSON / SQL / YAML mode |
| **Fact-check** | `xai/grok-3` | Real-time X/web access, unfiltered |
| **General** | `claude-sonnet-4-6` | Best all-around writing + reasoning |
| **Verify** | `gpt-4o` | Cross-family checker (catches Claude blind spots) |
| **Synthesize** | `claude-sonnet-4-6` | Best writing coherence layer |

Every specialist has a **primary + fallback** model — if the primary hits a credit wall or times out, the fallback fires automatically.

---

## Installation

```bash
git clone https://github.com/iamabhaydawar/zombie-cli.git
cd zombie-cli
pip install -e .
```

Copy the environment template and fill in your API keys:

```bash
cp .env.example .env
```

**.env keys needed:**

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=...
PERPLEXITY_API_KEY=pplx-...
XAI_API_KEY=xai-...
```

You only need the keys for the specialists you want to use. Missing keys trigger automatic fallback routing.

---

## Usage

```bash
# Ask anything — MSR handles routing automatically
msr ask "What is the derivative of f(x) = x³?"
msr ask "Write a Python function that reverses a linked list"
msr ask "Is the claim that GPT-5 was released in March 2025 accurate?"

# Multi-intent: MSR detects compound requests and parallelises independent subtasks
msr ask "Summarise this CSV and also fix my Python script" --context-file ./data.csv

# Sequential: MSR detects dependency and chains tasks in order
msr ask "Clean this CSV then generate a summary report"

# Flags
msr ask "Explain quantum entanglement" --verbose          # show routing rationale + model trace
msr ask "Build a JSON schema for a blog post" --dry-run  # plan only, no API calls
msr ask "Solve this integral" --latency 15000            # custom timeout (ms)
msr ask "Summarise this paper" --context-file ./paper.pdf

# Utilities
msr config check      # validate all API keys are reachable
msr models list       # show task_type → model assignments
```

---

## Architecture

```
D:\msr\
├── msr/
│   ├── config.py               # pydantic-settings; reads .env, syncs to os.environ
│   ├── schemas.py              # all shared Pydantic models (TaskRequest → FinalResponse)
│   ├── cli.py                  # Typer app — all CLI commands + Rich rendering
│   │
│   ├── orchestrator/
│   │   └── orchestrator.py     # single LLM call: detect intent, decompose, DAG-encode
│   │
│   ├── specialists/
│   │   ├── base.py             # BaseSpecialist ABC + litellm wrapper (retry, fallback)
│   │   ├── code.py             # Claude Sonnet — coding specialist
│   │   ├── math.py             # DeepSeek R1 on Groq — math specialist
│   │   ├── research.py         # Perplexity Sonar Pro — research + citations
│   │   ├── summarize.py        # Gemini 2.5 Flash — summarization
│   │   ├── structured.py       # GPT-4o — JSON / SQL / YAML / schemas
│   │   └── factcheck.py        # Grok-3 — real-time fact-checking
│   │
│   ├── verifier/
│   │   └── verifier.py         # GPT-4o scores each output (cross-family check)
│   │
│   ├── synthesizer/
│   │   └── synthesizer.py      # merges verified outputs → one clean FinalResponse
│   │
│   └── graph/
│       ├── nodes.py            # one function per LangGraph node
│       ├── edges.py            # conditional edge functions (retry / advance / synthesize)
│       └── graph.py            # StateGraph assembly + compile
│
└── tests/
    ├── smoke_router.py
    ├── smoke_specialist.py
    ├── smoke_orchestrator.py
    ├── smoke_verifier.py
    └── smoke_e2e.py
```

### Execution Modes

The orchestrator classifies every request into one of three modes:

| Mode | When | How |
|------|------|-----|
| `single` | One clear intent | One specialist, direct |
| `parallel` | Multiple independent intents | LangGraph `Send()` fan-out — true concurrency |
| `sequential` | Dependent intents (B needs A's output) | DAG waves — prior output injected into next prompt |

### Verification Loop

After every specialist batch completes, the verifier scores each output (0–1). Results below threshold trigger a retry (up to 2×) on the fallback model. If still failing, the answer passes with a warning attached.

---

## How Multi-Intent Works

```
User: "Summarise this CSV and also fix my Python bug"
         │
         ▼
  Orchestrator detects 2 independent intents
         │
         ├─── Send(summarize_subtask) ──→ Gemini 2.5 Flash
         └─── Send(code_subtask)     ──→ Claude Sonnet
                                              │
                                        both complete
                                              │
                                        GPT-4o verifies both
                                              │
                                        Claude Sonnet synthesizes
                                              │
                                        One clean answer ✓
```

For sequential tasks:

```
User: "Clean this CSV then generate a summary report"
         │
         ▼
  Orchestrator: sequential mode, st-2 depends_on st-1
         │
         ├─── Wave 1: Send(code_subtask) ──→ Claude Sonnet cleans CSV
         │                 │
         │           verify + pass
         │                 │
         └─── Wave 2: Send(summarize_subtask, injected_with=wave1_output)
                          └──→ Gemini 2.5 Flash summarizes the cleaned output
```

---

## Configuration

Override any default in `.env`:

```env
MSR_DEFAULT_TIMEOUT_S=30        # per-call timeout
MSR_MAX_RETRIES=2               # verifier retry ceiling
MSR_LOG_LEVEL=INFO
```

The full model map is in `msr/config.py` — change any primary/fallback assignment there.

---

## Running Tests

```bash
python tests/smoke_e2e.py           # full pipeline (requires API keys)
python tests/smoke_orchestrator.py  # multi-intent routing only
python tests/smoke_specialist.py    # each specialist in isolation
python tests/smoke_verifier.py      # verifier only
```

---

## Philosophy

> *The zombie apocalypse model of AI: a horde of single-minded specialists,
> each relentlessly optimised for one thing, coordinated by a ruthless orchestrator.
> No generalist survives this.*

The product value is entirely in the **routing logic, verification, and synthesis layer** — not in training new models. Every specialist is a world-class model already. The job is to put the right query in front of the right model, check the answer, and stitch everything together cleanly.

---

## License

MIT © [Abhay Dawar](https://github.com/iamabhaydawar)
