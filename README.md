<div align="center">

# 🧭 MAVPose

**Ask questions about your drone flight in plain English. Get plots.**

*Inspired by the Log Pose from One Piece — it locks onto your flight data and charts a course through it.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/AyushMaria/MAVPose/actions/workflows/ci.yml/badge.svg)](https://github.com/AyushMaria/MAVPose/actions/workflows/ci.yml)

</div>

---

MAVPose is a headless CLI tool that turns a natural language prompt into a matplotlib plot of your MAVLink flight log. Under the hood it uses an LLM (default: **Z.ai GLM-5.1** via OpenRouter), ChromaDB semantic search, and a sandboxed Python executor with self-healing retry logic.

```
$ python cli.py flight.tlog --prompt "Plot altitude over time"

📂 Parsing log file: flight.tlog
✅ Log parsed and vector index built.

🔍 Finding relevant data types for: 'Plot altitude over time'
✍️  Generating plot script with GLM-5.1...
⚙️  Running script...
✅ Plot saved to: plot.png
```

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  cli.py                                                     │
│                                                             │
│  1.  validate_mavlink_file()   ← extension / size / symlink │
│  2.  parse_mavlink_log()       ← pymavlink reads all msgs   │
│  3.  _create_embeddings()      ← ChromaDB vector store      │
│                                                             │
│  4.  find_relevant_data_types() ← semantic search           │
│  5.  create_plot()              ← LLM writes the script     │
│  6.  run_script()               ← subprocess + sandbox      │
│       └─ attempt_to_fix_script() on failure (up to N×)      │
│                                                             │
│  → plot.png saved next to your log file                     │
└─────────────────────────────────────────────────────────────┘
```

When the generated script fails, MAVPose feeds the error back to the LLM and retries automatically (default: **3 retries**, configurable via `--retries`).

---

## Features

- **Plain-English plots** — describe any flight data; MAVPose figures out which MAVLink fields to use
- **Automatic log parsing** — discovers every message type and field present in your `.tlog` / `.bin` / `.log`
- **Semantic field search** — vector embeddings surface the most relevant message types for your query
- **Self-healing scripts** — LLM debugs and rewrites failing scripts up to N times automatically
- **Sandboxed execution** — generated code runs in an isolated subprocess with a blocked-import denylist and a 30 s timeout
- **Interactive REPL mode** — omit `--prompt` to enter a live loop and plot multiple things in one session
- **Configurable model** — drop in any OpenRouter model with a one-line `.env` change
- **Persisted vector store** — ChromaDB is saved to disk; re-running on the same log skips re-embedding
- **CI-tested** — lint (ruff) and pytest run on every push via GitHub Actions

---

## Tech Stack

| Layer | Library / Service |
|---|---|
| LLM | [OpenRouter](https://openrouter.ai) → Z.ai GLM-5.1 (default) |
| Embeddings | OpenRouter → `openai/text-embedding-3-small` |
| LLM framework | LangChain (LCEL) — `langchain-openai`, `langchain-chroma` |
| Vector store | ChromaDB ≥ 0.5 (persisted to disk) |
| Drone log parsing | pymavlink 2.4.37 |
| Plotting | matplotlib 3.7.1 |
| Sandbox | Custom subprocess executor with import denylist + timeout |
| Config | python-dotenv |
| Lint / CI | ruff + pytest + GitHub Actions |

---

## Prerequisites

- **Python 3.10+**
- An **OpenRouter API key** — [get one here](https://openrouter.ai/keys) (free tier available)

---

## Installation

```bash
# 1. Clone
git clone https://github.com/AyushMaria/MAVPose.git
cd MAVPose

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp template.env .env
```

Open `.env` and fill in your key:

```env
OPENROUTER_API_KEY=your-key-here
OPENROUTER_MODEL=z-ai/glm-5.1   # default — change to any OpenRouter model
```

> ⚠️ Never commit `.env` — it is already listed in `.gitignore`.

---

## Usage

### Single prompt

```bash
python cli.py flight.tlog --prompt "Plot altitude over time"
```

### Interactive REPL

Omit `--prompt` to enter a live loop:

```bash
python cli.py flight.tlog

📊 Plot request > Plot battery voltage and current
📊 Plot request > Show GPS latitude and longitude
📊 Plot request > quit
```

### All CLI flags

```
usage: mavplot [-h] [--prompt PROMPT] [--retries N] [--verbose] log_file

positional arguments:
  log_file              Path to a MAVLink log file (.tlog, .bin, .log)

options:
  -p, --prompt TEXT     Plot request. Omit for interactive mode.
  -r, --retries N       Max self-healing retries on script failure (default: 3)
  -v, --verbose         Enable debug logging
```

The output plot is saved as `plot.png` in the same directory as your log file.

> **Need a sample log file?**  
> [Download here](https://drive.google.com/file/d/1BKv-NbSvYQz9XqqmyOyOhe3o4PAFDyZa/view?usp=sharing)

---

## Switching Models

MAVPose routes through [OpenRouter](https://openrouter.ai/models), so any model on the platform works. Just update `.env`:

```env
# Z.ai options
OPENROUTER_MODEL=z-ai/glm-5.1          # default — long-horizon agentic coding
OPENROUTER_MODEL=z-ai/glm-5            # flagship, complex systems
OPENROUTER_MODEL=z-ai/glm-5-turbo      # fast inference
OPENROUTER_MODEL=z-ai/glm-4.5         # MoE, 355B params, switchable reasoning

# OpenAI fallback
OPENROUTER_MODEL=openai/gpt-4o
```

---

## Project Structure

```
MAVPose/
├── cli.py                         # CLI entry point — argument parsing, REPL loop
├── app.py                         # Legacy stub (Gradio UI removed — see note below)
├── llm/
│   ├── gptPlotCreator.py          # PlotCreator — core LLM + plotting pipeline
│   ├── safe_executor.py           # Subprocess sandbox with import denylist
│   └── file_validator.py          # File validation (extension, size, symlink)
├── tests/
│   ├── test_extract_code_snippets.py
│   ├── test_file_validator.py
│   └── test_safe_executor.py
├── .github/workflows/ci.yml       # GitHub Actions: ruff lint + pytest
├── docs/
│   └── GPT_MAVPlot_Arch.png       # Architecture diagram
├── target/                        # Output directory for plot.py and plot.png
├── template.env                   # Environment variable template
├── requirements.txt
└── LICENSE
```

> **Note on `app.py`:** The Gradio web UI was removed in favour of the headless CLI. `app.py` is kept as a stub that prints a helpful error if invoked directly.

---

## Core API — `PlotCreator`

All logic lives in `llm/gptPlotCreator.py`. You can use it programmatically:

```python
from llm.gptPlotCreator import PlotCreator

creator = PlotCreator(max_retries=3)
creator.set_logfile_name("flight.tlog")
creator.parse_mavlink_log()

data_types = creator.find_relevant_data_types("Plot altitude over time")
creator.create_plot("Plot altitude over time", data_types)
result, code = creator.run_script()
# → plot saved to flight_dir/plot.png
```

| Method | What it does |
|---|---|
| `set_logfile_name(path)` | Registers the log file; derives `plot.py` and `plot.png` output paths |
| `parse_mavlink_log()` | Reads all MAVLink message types and fields; builds the ChromaDB index |
| `find_relevant_data_types(query)` | Semantic search — returns the most relevant message-type JSON for a query |
| `create_plot(query, data_types)` | Calls the LLM (LCEL pipeline) to write and save a plotting script |
| `run_script()` | Executes the script; calls `attempt_to_fix_script()` on failure |
| `attempt_to_fix_script(error)` | Feeds the error back to the LLM and rewrites the script |

---

## Sandbox Security

Generated scripts run in an isolated subprocess (`safe_executor.py`) with:

- **Import denylist** — blocks `os`, `subprocess`, `socket`, `urllib`, `pickle`, and ~20 other dangerous modules from being imported *by the generated script itself*. Transitive stdlib imports are allowed through.
- **30 s timeout** — the subprocess is killed if it runs too long.
- **No RestrictedPython** — the approach uses a custom `__import__` hook injected at runtime, which is lighter and avoids the compatibility issues of RestrictedPython.

---

## Running Tests

```bash
pip install pytest pytest-cov ruff
pytest tests/ -v --cov=llm
```

To lint:
```bash
ruff check llm/ cli.py
```

---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `KeyError: OPENROUTER_API_KEY` | `.env` not configured | Copy `template.env` to `.env` and add your key |
| `ModuleNotFoundError` | Missing dependency | Run `pip install -r requirements.txt` in your venv |
| Plot not generated after N retries | LLM script failed repeatedly | Try a more specific prompt; use `--verbose` to inspect errors |
| `FileValidationError` | Wrong file type or empty file | Only `.tlog`, `.bin`, `.log` files ≤ 200 MB are accepted |
| `FileValidationError: Symlink not allowed` | Symlink passed as log path | Provide the real file path, not a symlink |
| ChromaDB version conflict | Stale venv | Delete `venv/` and reinstall with a fresh `pip install -r requirements.txt` |

---

## License

MIT — see [LICENSE](LICENSE) for details.
