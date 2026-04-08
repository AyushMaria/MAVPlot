import gradio as gr
import os
from llm.gptPlotCreator import PlotCreator
from llm.file_validator import validate_mavlink_file, FileValidationError


def add_text(history, text, plot_creator):
    history = history + [(text, None)]
    return history, plot_creator, ""


def add_file(history, file, plot_creator):
    print(file.name)

    # --- Security: validate the uploaded file before processing ---
    try:
        validate_mavlink_file(file.name)
    except FileValidationError as e:
        history = history + [(None, f"❌ File rejected: {e}")]
        return history, plot_creator
    except FileNotFoundError as e:
        history = history + [(None, f"❌ File not found: {e}")]
        return history, plot_creator

    history = history + [((file.name,), None)]
    plot_creator.set_logfile_name(file.name)
    return history, plot_creator


def format_history(history):
    return "\n".join([f"Human: {entry[0]}\nAI: {entry[1]}" for entry in history])


def bot(history, plot_creator):
    user_input = history[-1][0] if history and history[-1][0] else None

    print(user_input)
    print(type(plot_creator))

    if isinstance(user_input, str) and plot_creator.logfile_name != "":

        history[-1][1] = "I am figuring out what data types are relevant for the plot...\n"
        yield history, plot_creator
        data_types_str = plot_creator.find_relevant_data_types(user_input)

        history[-1][1] += "I am now generating a script to plot the data...\n"
        yield history, plot_creator
        plot_creator.create_plot(user_input, data_types_str)

        history[-1][1] += "I am now running the script I just generated...\n"
        yield history, plot_creator
        response = plot_creator.run_script()

        history = history + [(None, "Here is the code used to generate the plot:")]
        history = history + [(None, f"{response[1]}")]
        history = history + response[0]

        yield history, plot_creator

    elif not plot_creator.logfile_name:
        yield history + [(None, "Please upload a log file before attempting to create a plot.")], plot_creator

    else:
        plot_creator = PlotCreator()
        file_path = user_input[0]

        # --- Security: validate file again at the bot level ---
        try:
            validate_mavlink_file(file_path)
        except (FileValidationError, FileNotFoundError) as e:
            yield history + [(None, f"❌ File rejected: {e}")], plot_creator
            return

        plot_creator.set_logfile_name(file_path)
        filename, extension = os.path.splitext(os.path.basename(file_path))

        history[-1][0] = f"user uploaded file: {filename}{extension}"
        history[-1][1] = "processing file..."
        yield history, plot_creator

        plot_creator.parse_mavlink_log()
        history = history + [(None, "I am done processing the file. Now you can ask me to generate a plot.")]
        yield history, plot_creator

    return history, plot_creator


initial_message = (
    None,
    "Hello! I am a chatbot designed to plot logs from drones. "
    "To get started, please upload a MAVLink `.tlog` file. "
    "Then describe the plot you want, e.g. 'Plot altitude over time'. "
    "Need a sample log? Download one here: "
    "https://drive.google.com/file/d/1BKv-NbSvYQz9XqqmyOyOhe3o4PAFDyZa/view?usp=sharing",
)

with gr.Blocks() as demo:
    gr.Markdown(
        "# GPT MAVPlot\n\n"
        "Upload a MAVLink `.tlog` file and describe the plot you want. "
        "MAVPlot will generate and run a Python plotting script using `pymavlink` and `matplotlib`. "
        "The output includes the plot image and the code used to generate it."
    )
    plot_creator = gr.State(PlotCreator())

    chatbot = gr.Chatbot([initial_message], elem_id="chatbot", height=750)

    with gr.Row():
        with gr.Column(scale=7):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Describe the plot you want, e.g. 'Plot altitude over time'",
            )
        with gr.Column(scale=1, min_width=0):
            btn = gr.UploadButton("📁", file_types=[".tlog", ".bin", ".log"])

    txt.submit(add_text, [chatbot, txt, plot_creator], [chatbot, plot_creator, txt]).then(
        bot, [chatbot, plot_creator], [chatbot, plot_creator]
    )
    btn.upload(add_file, [chatbot, btn, plot_creator], [chatbot, plot_creator]).then(
        bot, [chatbot, plot_creator], [chatbot, plot_creator]
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch()
