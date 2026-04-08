import re
import logging
import subprocess
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from pymavlink import mavutil
import json

from llm.safe_executor import execute_script

logger = logging.getLogger(__name__)


class PlotCreator:
    """
    PlotCreator generates Python scripts to plot MAVLink data
    using OpenAI GPT. Scripts are executed inside a RestrictedPython
    sandbox to prevent arbitrary code execution.
    """

    def __init__(self):
        """
        Initialize an instance of PlotCreator.
        """
        # Instance variables (not class-level) to avoid shared state between instances
        self.last_code = ""
        self.logfile_name = ""
        self.script_path = ""
        self.plot_path = ""
        self.message_types = {}

        load_dotenv()
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

        llm = ChatOpenAI(model_name=self.model, max_tokens=2000, temperature=0)

        mavlink_data_prompt = PromptTemplate(
            input_variables=["data_types", "history", "human_input", "file", "output_file"],
            template=(
                "You are an AI conversation agent that generates Python scripts to plot MAVLink data. "
                "Create a Python script using matplotlib and pymavlink's mavutil to plot the requested data. "
                "Do not explain the code — return only the script. "
                "Plot each variable over time in seconds. "
                "Save the plot to {output_file} at 400+ dpi. Do not call plt.show(). "
                "Use blocking=False in recv_match and break the loop if msg is None.\n\n"
                "Relevant data types:\n{data_types}\n\n"
                "Chat History:\n{history}\n\n"
                "HUMAN: {human_input}\n\n"
                "Read data from file: {file}."
            ),
        )

        self.chain = LLMChain(verbose=True, llm=llm, prompt=mavlink_data_prompt)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_code_snippets(text):
        """
        Extracts code snippets enclosed in triple backticks from a text.
        Returns a list of code strings; falls back to [text] if none found.
        """
        pattern = r'```.*?\n(.*?)```'
        snippets = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
        return snippets if snippets else [text]

    @staticmethod
    def write_plot_script(filename, text):
        """Write a script string to disk."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write(text)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def set_logfile_name(self, filename):
        """Set the log file path and derive output paths."""
        path = os.path.dirname(filename)
        self.logfile_name = filename
        self.script_path = os.path.join(path, "plot.py")
        self.plot_path = os.path.join(path, "plot.png")

    def find_relevant_data_types(self, human_input):
        """Semantic similarity search for MAVLink message types matching the query."""
        docs = self.db.similarity_search(human_input)
        return "".join(doc.page_content + "\n\n" for doc in docs)

    def create_plot(self, human_input, data_type_info_text):
        """Call GPT to generate a plotting script, then write it to disk."""
        history = ("\n\nLast script generated:\n\n" + self.last_code) if self.last_code else ""

        response = self.chain.run({
            "data_types": data_type_info_text,
            "history": history,
            "file": self.logfile_name,
            "human_input": human_input,
            "output_file": self.plot_path,
        })
        logger.debug("GPT response: %s", response)

        code = self.extract_code_snippets(response)
        self.write_plot_script(self.script_path, code[0])
        self.last_code = code[0]
        return code[0]

    def run_script(self):
        """
        Execute the generated plot script inside the RestrictedPython sandbox.
        Falls back to attempt_to_fix_script on failure (up to MAX_RETRIES).
        """
        MAX_RETRIES = 2

        with open(self.script_path, "r") as f:
            script_code = f.read()

        success, output = execute_script(script_code)

        if not success:
            logger.warning("Script execution failed: %s", output)
            for attempt in range(1, MAX_RETRIES + 1):
                logger.info("Self-healing attempt %d/%d", attempt, MAX_RETRIES)
                fixed_code = self.attempt_to_fix_script(self.script_path, output)
                self.last_code = fixed_code
                success, output = execute_script(fixed_code)
                if success:
                    break
            else:
                self.last_code = (
                    f"Sorry, I was unable to fix the script after {MAX_RETRIES} attempts.\n"
                    f"Last error:\n{output}\n\nLast script:\n{self.last_code}"
                )

        return [[(None, (self.plot_path,))], self.last_code]

    def attempt_to_fix_script(self, filename, error_message):
        """
        Feed the failing script and its error back to GPT for a self-healing fix.

        Returns the fixed (or best-effort) script as a string.
        """
        llm = ChatOpenAI(model_name=self.model, max_tokens=8000, temperature=0)

        fix_prompt = PromptTemplate(
            input_variables=["data_types", "error", "script"],
            template=(
                "You are an AI agent that debugs Python scripts for plotting MAVLink data "
                "with matplotlib and pymavlink's mavutil.\n\n"
                "The following script produced an error:\n\n{script}\n\n"
                "Error:\n\n{error}\n\n"
                "Relevant message definitions:\n\n{data_types}\n\n"
                "Please fix the script and return it in a markdown code block."
            ),
        )

        with open(filename, "r") as f:
            script = f.read()

        chain = LLMChain(verbose=True, llm=llm, prompt=fix_prompt)
        try:
            response = chain.run({
                "data_types": self.message_types,
                "error": error_message,
                "script": script,
            })
        except Exception as e:  # noqa: BLE001
            logger.error("LLM fix request failed: %s", e)
            return script  # Return original if LLM call itself fails

        code = self.extract_code_snippets(response)
        fixed_code = code[0]
        self.write_plot_script(filename, fixed_code)
        return fixed_code

    # ------------------------------------------------------------------
    # Log parsing & embeddings
    # ------------------------------------------------------------------

    def parse_mavlink_log(self):
        """
        Parse the MAVLink log and extract all unique message types and fields.
        Creates ChromaDB embeddings for semantic search.

        Returns a JSON string of message types.
        """
        self.message_types = {}
        mav_log = mavutil.mavlink_connection(self.logfile_name)

        while True:
            try:
                msg = mav_log.recv_match(blocking=False, type=None)
                if msg is None:
                    break
                msg_type = msg.get_type()
                if msg_type not in self.message_types:
                    self.message_types[msg_type] = {
                        "count": 1,
                        "fields": {
                            field: type(getattr(msg, field)).__name__
                            for field in msg.get_fieldnames()
                        },
                    }
                else:
                    self.message_types[msg_type]["count"] += 1
            except KeyboardInterrupt:
                break
            except Exception as e:  # noqa: BLE001
                logger.error("Error reading MAVLink message: %s", e)
                break

        self._create_embeddings(self.message_types)
        return json.dumps(self.message_types, indent=4)

    def _create_embeddings(self, message_types):
        """
        Create OpenAI embeddings for all parsed message types and store in ChromaDB.
        The vector store is persisted to ./chroma_db to avoid re-embedding on restart.
        """
        texts = [
            json.dumps({msg_type: message_types[msg_type]})
            for msg_type in message_types
        ]
        logger.debug("Creating embeddings for %d message types", len(texts))
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.db = Chroma.from_texts(
            texts,
            embeddings,
            persist_directory="./chroma_db",
        )
        self.db.persist()
