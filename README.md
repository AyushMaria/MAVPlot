# MAVPlot

**A GPT-powered chatbot that turns natural language into drone flight plots.**

Upload a MAVLink `.tlog` file, describe the data you want to visualise, and MAVPlot will write a Python script using `pymavlink` + `matplotlib`, execute it, and display the resulting plot — all inside a Gradio chat interface.

![Architecture Diagram](docs/GPT_MAVPlot_Arch.png)

---

## How It Works

```
User uploads .tlog file
        ↓
 parse_mavlink_log()  ←  pymavlink reads all message types & fields
        ↓
 create_embeddings()  ←  OpenAI Ada-002 embeds each message type into ChromaDB
        ↓
User sends a chat prompt  (e.g. "Plot altitude over time")
        ↓
 find_relevant_data_types()  ←  semantic search in ChromaDB
        ↓
 create_plot()  ←  GPT writes a pymavlink + matplotlib script
        ↓
 run_script()  ←  script executed via subprocess
        ↓
    Plot image returned to chat  +  generated code shown
```

If the generated script throws an error, `attempt_to_fix_script()` automatically feeds the error and the original script back to GPT for a self-healing retry.

---

## Features

- **Natural language → plot** — describe any flight data in plain English
- **Auto log parsing** — extracts all MAVLink message types and fields automatically
- **Semantic data type search** — uses vector embeddings to find the most relevant data fields for your query
- **Self-healing scripts** — if the generated script fails, GPT debugs and retries it automatically
- **Code transparency** — the generated Python script is shown alongside every plot
- **Gradio UI** — clean web interface with file upload and chat, no frontend code needed
- **Configurable model** — swap GPT models via the `.env` file

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Gradio 3.40 |
| LLM | OpenAI GPT (via LangChain) |
| Embeddings | OpenAI `text-embedding-ada-002` |
| Vector Store | ChromaDB |
| Drone Log Parsing | pymavlink 2.4.37 |
| Plotting | matplotlib 3.7.1 |
| Orchestration | LangChain 0.0.268 |
| Config | python-dotenv |

---

## Project Structure

```
MAVPlot/
├── app.py                    # Gradio UI and chat event handlers
├── llm/
│   └── gptPlotCreator.py     # PlotCreator class — core LLM + plotting logic
├── docs/
│   └── GPT_MAVPlot_Arch.png  # Architecture diagram
├── target/                   # Output directory for generated plot.py and plot.png
├── template.env              # Environment variable template
├── requirements.txt          # Python dependencies
└── LICENSE
```

---

## Prerequisites

- Python 3.9+
- An **OpenAI API key** with access to the model you want to use (default: `gpt-3.5-turbo`)

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/AyushMaria/MAVPlot.git
cd MAVPlot
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure environment variables**

Copy `template.env` to `.env` and fill in your values:
```bash
cp template.env .env
```

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL=gpt-3.5-turbo
```

> You can swap `gpt-3.5-turbo` for `gpt-4` or any other model you have access to.

**4. Run the app**
```bash
python app.py
```

Open the local Gradio URL shown in the terminal (usually `http://127.0.0.1:7860`).

---

## Usage

1. **Upload a `.tlog` file** — click the upload button (📁) in the chat interface
2. Wait for MAVPlot to parse the log and confirm it's ready
3. **Type a plot request**, for example:
   - `"Plot altitude over time"`
   - `"Show me GPS latitude and longitude"`
   - `"Plot battery voltage and current"`
4. MAVPlot will:
   - Find relevant MAVLink message types using semantic search
   - Generate a Python plotting script with GPT
   - Run the script and display the plot
   - Show the generated code in the chat

> **Need a sample log file?**  
> Download one here: https://drive.google.com/file/d/1BKv-NbSvYQz9XqqmyOyOhe3o4PAFDyZa/view?usp=sharing

---

## PlotCreator Class

The core logic lives in `llm/gptPlotCreator.py` in the `PlotCreator` class:

| Method | Description |
|---|---|
| `parse_mavlink_log()` | Reads `.tlog`, extracts all message types and field names |
| `create_embeddings()` | Embeds each message type into a local ChromaDB vector store |
| `find_relevant_data_types()` | Semantic similarity search to find fields matching the user query |
| `create_plot()` | Calls GPT to generate a `pymavlink` + `matplotlib` Python script |
| `run_script()` | Executes the generated script via `subprocess` |
| `attempt_to_fix_script()` | Feeds errors back to GPT for automatic script debugging |

---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `AuthenticationError` | Invalid or missing OpenAI API key | Check your `.env` file has the correct `OPENAI_API_KEY` |
| `ModuleNotFoundError: pymavlink` | Missing dependency | Run `pip install -r requirements.txt` |
| Plot not generated | GPT script failed twice | Check that your log file is a valid `.tlog`; try a more specific prompt |
| `chromadb` version conflict | Dependency mismatch | Use a fresh virtual environment with pinned versions from `requirements.txt` |
| `gr.Chatbot.style()` deprecation | Gradio version mismatch | Use exactly `gradio==3.40.1` as pinned in `requirements.txt` |

---

## License

MIT — see [LICENSE](LICENSE) for details.
