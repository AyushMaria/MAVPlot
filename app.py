"""
app.py

Gradio 4.x UI for MAVPlot.

Improvements over original:
  - Gradio 4.x API (no deprecated .style() calls)
  - Progress bar during log parsing and plot generation
  - Status indicator chip (idle / processing / ready / error)
  - Conversation reset button (clears chat + resets PlotCreator state)
  - File type restricted to .tlog / .bin / .log at the upload button level
  - Fixed placeholder text
  - File validation wired in (from llm/file_validator.py)
"""

import os
import gradio as gr
from llm.gptPlotCreator import PlotCreator
from llm.file_validator import validate_mavlink_file, FileValidationError

# ---------------------------------------------------------------------------
# Status helper
# ---------------------------------------------------------------------------

def _status(state: str) -> str:
    """Return a short markdown status chip string."""
    icons = {
        "idle":       "⚪ Idle — upload a log file to begin",
        "uploading":  "🔵 Uploading & parsing log file...",
        "ready":      "🟢 Log loaded — ready to plot",
        "thinking":   "🟡 Generating plot script...",
        "running":    "🟡 Running script...",
        "error":      "🔴 Error — see chat for details",
    }
    return icons.get(state, "")


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def add_text(history, text, plot_creator):
    """Append user text to chat history and clear the input box."""
    history = history + [{"role": "user", "content": text}]
    return history, plot_creator, ""


def add_file(history, file, plot_creator):
    """Validate and register an uploaded log file."""
    if file is None:
        return history, plot_creator, _status("idle")

    try:
        validate_mavlink_file(file.name)
    except (FileValidationError, FileNotFoundError) as e:
        history = history + [{"role": "assistant", "content": f"❌ File rejected: {e}"}]
        return history, plot_creator, _status("error")

    filename = os.path.basename(file.name)
    history = history + [{"role": "user", "content": f"📁 Uploaded: **{filename}**"}]
    plot_creator.set_logfile_name(file.name)
    return history, plot_creator, _status("uploading")


def bot(history, plot_creator, progress=gr.Progress(track_tqdm=True)):
    """
    Main chat handler. Yields intermediate states for streaming UX.
    Uses gr.Progress for the log-parsing and script-generation steps.
    """
    if not history:
        yield history, plot_creator, _status("idle")
        return

    last = history[-1]
    user_input = last["content"] if last["role"] == "user" else None

    # --- Plot request (text message while a log is loaded) ---
    if isinstance(user_input, str) and not user_input.startswith("📁") and plot_creator.logfile_name:

        history.append({"role": "assistant", "content": "🔍 Finding relevant data types..."})
        yield history, plot_creator, _status("thinking")

        progress(0.2, desc="Searching vector store...")
        data_types_str = plot_creator.find_relevant_data_types(user_input)

        history[-1]["content"] = "✍️ Generating plot script with GPT..."
        yield history, plot_creator, _status("thinking")

        progress(0.5, desc="Calling GPT...")
        plot_creator.create_plot(user_input, data_types_str)

        history[-1]["content"] = "⚙️ Running the generated script..."
        yield history, plot_creator, _status("running")

        progress(0.8, desc="Executing script...")
        response = plot_creator.run_script()

        # Remove the status message and add real results
        history.pop()
        history.append({"role": "assistant", "content": "Here is the code used to generate the plot:"})
        history.append({"role": "assistant", "content": f"```python\n{response[1]}\n```"})
        # Add the plot image
        plot_path = response[0][0][1][0] if response[0] else None
        if plot_path and os.path.exists(plot_path):
            history.append({"role": "assistant", "content": gr.Image(value=plot_path)})

        progress(1.0, desc="Done!")
        yield history, plot_creator, _status("ready")

    # --- No log loaded yet ---
    elif not plot_creator.logfile_name and isinstance(user_input, str) and not user_input.startswith("📁"):
        history.append({
            "role": "assistant",
            "content": "⚠️ Please upload a `.tlog`, `.bin`, or `.log` file first using the 📁 button."
        })
        yield history, plot_creator, _status("idle")

    # --- File was just uploaded — parse it ---
    elif plot_creator.logfile_name and (user_input is None or user_input.startswith("📁")):
        history.append({"role": "assistant", "content": "⏳ Parsing log file and building vector index..."})
        yield history, plot_creator, _status("uploading")

        progress(0.1, desc="Parsing MAVLink messages...")
        plot_creator.parse_mavlink_log()

        progress(1.0, desc="Ready!")
        history[-1]["content"] = (
            "✅ Log file parsed and indexed! "
            "Now describe the plot you want, e.g.:\n"
            "- *Plot altitude over time*\n"
            "- *Show GPS latitude and longitude*\n"
            "- *Plot battery voltage and current*"
        )
        yield history, plot_creator, _status("ready")


def reset_session(plot_creator):
    """Clear chat history and reset PlotCreator to a fresh state."""
    fresh = PlotCreator()
    initial = [{
        "role": "assistant",
        "content": (
            "🔄 Session reset. Upload a new `.tlog` file to begin.\n\n"
            "Need a sample log? "
            "[Download here](https://drive.google.com/file/d/1BKv-NbSvYQz9XqqmyOyOhe3o4PAFDyZa/view?usp=sharing)"
        )
    }]
    return initial, fresh, _status("idle")


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

initial_history = [{
    "role": "assistant",
    "content": (
        "👋 Hello! I'm **MAVPlot** — a GPT-powered drone log visualiser.\n\n"
        "**To get started:**\n"
        "1. Click 📁 and upload a MAVLink `.tlog` file\n"
        "2. Wait for the log to be parsed\n"
        "3. Describe the plot you want (e.g. *Plot altitude over time*)\n\n"
        "Need a sample log? "
        "[Download here](https://drive.google.com/file/d/1BKv-NbSvYQz9XqqmyOyOhe3o4PAFDyZa/view?usp=sharing)"
    )
}]

with gr.Blocks(title="MAVPlot", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        "# ✈️ MAVPlot\n"
        "**Natural language → drone flight plots.** "
        "Upload a MAVLink `.tlog` file and describe what you want to visualise."
    )

    plot_creator_state = gr.State(PlotCreator())

    with gr.Row():
        status_md = gr.Markdown(value=_status("idle"), elem_id="status")
        reset_btn = gr.Button("🔄 Reset session", scale=0, size="sm", variant="secondary")

    chatbot = gr.Chatbot(
        value=initial_history,
        elem_id="chatbot",
        height=620,
        type="messages",       # Gradio 4.x messages format
        show_copy_button=True,
        bubble_full_width=False,
        avatar_images=(None, "https://cdn.simpleicons.org/openai/000000"),
    )

    with gr.Row():
        with gr.Column(scale=8):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Describe the plot you want, e.g. 'Plot altitude over time'",
                lines=1,
                max_lines=4,
                autofocus=True,
            )
        with gr.Column(scale=1, min_width=60):
            upload_btn = gr.UploadButton(
                "📁",
                file_types=[".tlog", ".bin", ".log"],
                size="lg",
            )

    # Wire events
    txt.submit(
        add_text, [chatbot, txt, plot_creator_state], [chatbot, plot_creator_state, txt]
    ).then(
        bot, [chatbot, plot_creator_state], [chatbot, plot_creator_state, status_md]
    )

    upload_btn.upload(
        add_file, [chatbot, upload_btn, plot_creator_state], [chatbot, plot_creator_state, status_md]
    ).then(
        bot, [chatbot, plot_creator_state], [chatbot, plot_creator_state, status_md]
    )

    reset_btn.click(
        reset_session, [plot_creator_state], [chatbot, plot_creator_state, status_md]
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch()
