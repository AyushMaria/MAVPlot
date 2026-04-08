"""
gptPlotCreator.py

Core LLM + plotting logic for MAVPlot.

Migrated from deprecated langchain 0.0.x APIs to:
  - langchain-openai  (ChatOpenAI, OpenAIEmbeddings)
  - langchain-chroma  (Chroma)
  - LCEL pipelines    (prompt | llm | StrOutputParser)
  - text-embedding-3-small  (replaces deprecated ada-002)
  - Persistent ChromaDB     (avoids re-embedding on restart)
  - Streaming LLM output    (tokens streamed to Gradio via callback)
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from pymavlink import mavutil

# --- Modern LangChain imports (langchain-openai / langchain-chroma) ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from llm.safe_executor import execute_script

logger = logging.getLogger(__name__)

# Path to persist ChromaDB so embeddings survive restarts
CHROMA_PERSIST_DIR = "./chroma_db"

# Embedding model — text-embedding-3-small replaces deprecated ada-002
EMBEDDING_MODEL = "text-embedding-3-small"

# Maximum self-healing retries before giving up
MAX_RETRIES = 2


class PlotCreator:
    """
    Generates Python plotting scripts from natural language using GPT,
    executes them in a sandbox, and self-heals on failure.
    """

    def __init__(self, stream_callback=None):
        """
        Args:
            stream_callback: Optional callable(token: str) invoked for each
                             streamed token. Pass a Gradio queue callback to
                             show partial GPT output in the chat UI.
        """
        # Instance variables — never class-level to avoid shared state
        self.last_code: str = ""
        self.logfile_name: str = ""
        self.script_path: str = ""
        self.plot_path: str = ""
        self.message_types: dict = {}
        self.db: Chroma | None = None

        load_dotenv()
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

        # Streaming callback — if provided, tokens are pushed to the UI in real time
        callbacks = [stream_callback] if stream_callback else [StreamingStdOutCallbackHandler()]

        # LCEL pipeline: prompt | llm | parser
        # Uses streaming=True so tokens arrive incrementally
        self._llm = ChatOpenAI(
            model_name=self.model,
            max_tokens=2000,
            temperature=0,
            streaming=True,
            callbacks=callbacks,
        )

        self._plot_prompt = PromptTemplate(
            input_variables=["data_types", "history", "human_input", "file", "output_file"],
            template=(
                "You are an AI agent that generates Python scripts to plot MAVLink drone flight data.\n"
                "Create a Python script using matplotlib and pymavlink's mavutil.\n"
                "Rules:\n"
                "- Return ONLY the script inside a markdown ```python code block. No explanations.\n"
                "- Plot each variable over time in seconds.\n"
                "- Save the plot to {output_file} at 400+ dpi. Do NOT call plt.show().\n"
                "- Use blocking=False in recv_match. Break the loop if msg is None.\n\n"
                "Relevant MAVLink data types in this log:\n{data_types}\n\n"
                "Chat history:\n{history}\n\n"
                "User request: {human_input}\n\n"
                "Read log data from: {file}"
            ),
        )

        self._fix_prompt = PromptTemplate(
            input_variables=["data_types", "error", "script"],
            template=(
                "You are an AI agent that debugs Python scripts for plotting MAVLink data "
                "using matplotlib and pymavlink's mavutil.\n\n"
                "Script that produced an error:\n```python\n{script}\n```\n\n"
                "Error message:\n{error}\n\n"
                "Relevant MAVLink message definitions:\n{data_types}\n\n"
                "Fix the script and return it in a markdown ```python code block."
            ),
        )

        self._parser = StrOutputParser()

        # LCEL chains
        self._plot_chain = self._plot_prompt | self._llm | self._parser
        self._fix_chain = self._fix_prompt | self._llm | self._parser

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_code_snippets(text: str) -> list[str]:
        """Extract ```...``` code blocks. Falls back to [text] if none found."""
        snippets = re.findall(r'```.*?\n(.*?)```', text, re.DOTALL | re.MULTILINE)
        return snippets if snippets else [text]

    @staticmethod
    def write_plot_script(filename: str, text: str) -> None:
        """Write script text to disk, creating parent directories if needed."""
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        Path(filename).write_text(text, encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_logfile_name(self, filename: str) -> None:
        """Set the active log file and derive output paths."""
        path = os.path.dirname(filename)
        self.logfile_name = filename
        self.script_path = os.path.join(path, "plot.py")
        self.plot_path = os.path.join(path, "plot.png")

    def find_relevant_data_types(self, human_input: str) -> str:
        """Semantic similarity search for MAVLink fields matching the user query."""
        docs = self.db.similarity_search(human_input)
        return "".join(doc.page_content + "\n\n" for doc in docs)

    def create_plot(self, human_input: str, data_type_info_text: str) -> str:
        """Call GPT via LCEL to generate a plotting script, then write it to disk."""
        history = ("Last script generated:\n" + self.last_code) if self.last_code else ""

        response = self._plot_chain.invoke({
            "data_types": data_type_info_text,
            "history": history,
            "file": self.logfile_name,
            "human_input": human_input,
            "output_file": self.plot_path,
        })
        logger.debug("GPT plot response: %s", response)

        code = self.extract_code_snippets(response)
        self.write_plot_script(self.script_path, code[0])
        self.last_code = code[0]
        return code[0]

    def run_script(self) -> list:
        """
        Execute the generated script in the RestrictedPython sandbox.
        Retries up to MAX_RETRIES times via GPT self-healing on failure.

        Returns:
            [[(None, (plot_path,))], last_code]
        """
        self.last_code = Path(self.script_path).read_text(encoding="utf-8")
        success, output = execute_script(self.last_code)

        if not success:
            logger.warning("Script execution failed (attempt 0): %s", output)
            for attempt in range(1, MAX_RETRIES + 1):
                logger.info("Self-healing attempt %d/%d", attempt, MAX_RETRIES)
                fixed_code = self.attempt_to_fix_script(self.script_path, output)
                self.last_code = fixed_code
                success, output = execute_script(fixed_code)
                if success:
                    logger.info("Script fixed on attempt %d", attempt)
                    break
            else:
                self.last_code = (
                    f"Sorry, I was unable to fix the script after {MAX_RETRIES} attempts.\n"
                    f"Last error:\n{output}\n\nLast script attempted:\n{self.last_code}"
                )

        return [[(None, (self.plot_path,))], self.last_code]

    def attempt_to_fix_script(self, filename: str, error_message: str) -> str:
        """
        Feed the failing script + error back to GPT for a self-healing fix.
        Returns the fixed script string (or the original on LLM failure).
        """
        script = Path(filename).read_text(encoding="utf-8")

        try:
            response = self._fix_chain.invoke({
                "data_types": json.dumps(self.message_types, indent=2),
                "error": error_message,
                "script": script,
            })
        except Exception as e:  # noqa: BLE001
            logger.error("LLM fix request failed: %s", e)
            return script

        code = self.extract_code_snippets(response)
        fixed_code = code[0]
        self.write_plot_script(filename, fixed_code)
        return fixed_code

    def parse_mavlink_log(self) -> str:
        """
        Parse the MAVLink .tlog and extract all unique message types + fields.
        Creates/updates a persisted ChromaDB vector store for semantic search.

        Returns a JSON string of {message_type: {count, fields}} dict.
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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_embeddings(self, message_types: dict) -> None:
        """
        Embed all MAVLink message types into a persisted ChromaDB vector store.

        Uses text-embedding-3-small (replaces deprecated text-embedding-ada-002).
        The store is persisted to CHROMA_PERSIST_DIR so re-uploading the same
        log file does not re-embed from scratch.
        """
        texts = [
            json.dumps({msg_type: message_types[msg_type]})
            for msg_type in message_types
        ]
        logger.info("Creating embeddings for %d message types -> %s", len(texts), CHROMA_PERSIST_DIR)

        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.db = Chroma.from_texts(
            texts,
            embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        logger.info("ChromaDB persisted to %s", CHROMA_PERSIST_DIR)
