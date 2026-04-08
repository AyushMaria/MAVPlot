"""
llm/gptPlotCreator.py

Core MAVPlot logic:
  1. Parse a MAVLink .tlog/.bin/.log file into message-type metadata
  2. Embed each message type into a persisted ChromaDB vector store
  3. Use semantic search to find relevant fields for a user query
  4. Generate a plotting script via an LLM (OpenRouter / GLM-5.1) using an LCEL pipeline
  5. Execute the script, self-healing up to max_retries times on failure
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from typing import Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pymavlink import mavutil

logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {".tlog", ".bin", ".log"}

# OpenRouter base URL — OpenAI-compatible endpoint
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default model — GLM-5.1 on OpenRouter
DEFAULT_MODEL = "z-ai/glm-5.1"

# Embedding model — OpenRouter proxies OpenAI embeddings
EMBEDDING_MODEL = "openai/text-embedding-3-small"


class PlotCreator:
    """
    Generates Python plotting scripts from MAVLink log files using an LLM
    served via the OpenRouter API (default: Z.ai GLM-5.1).

    Args:
        max_retries (int): Number of self-healing retries if the generated
            script fails. Defaults to 3.
    """

    def __init__(self, max_retries: int = 3) -> None:
        load_dotenv()

        self.logfile_name: str = ""
        self.script_path: str = ""
        self.plot_path: str = ""
        self.last_code: str = ""
        self.message_types: dict = {}
        self.db: Optional[Chroma] = None
        self.max_retries: int = max_retries

        api_key: str = os.environ["OPENROUTER_API_KEY"]
        self.model: str = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)

        # Both ChatOpenAI instances point at OpenRouter's OpenAI-compatible endpoint.
        llm = ChatOpenAI(
            model_name=self.model,
            max_tokens=2000,
            temperature=0,
            openai_api_key=api_key,
            openai_api_base=OPENROUTER_BASE_URL,
        )

        plot_prompt = PromptTemplate(
            input_variables=["data_types", "history", "human_input", "file", "output_file"],
            template=(
                "You are an AI agent that generates Python scripts to plot MAVLink data.\n"
                "Use matplotlib and pymavlink's mavutil. Do NOT explain the code — return ONLY "
                "the script inside a markdown ```python``` block.\n"
                "Plot each independent variable over time in seconds.\n"
                "Save the plot to {output_file} at >=400 dpi. Do NOT call plt.show().\n"
                "Use blocking=False in recv_match; break the loop if msg is None.\n\n"
                "Relevant data types:\n{data_types}\n\n"
                "Chat history:\n{history}\n\n"
                "HUMAN: {human_input}\n\n"
                "Read data from: {file}"
            ),
        )
        self._plot_chain = plot_prompt | llm | StrOutputParser()

        fix_prompt = PromptTemplate(
            input_variables=["data_types", "error", "script"],
            template=(
                "You are an AI agent that debugs MAVLink plotting scripts.\n"
                "The following script produced an error.\n\n"
                "Script:\n{script}\n\n"
                "Error:\n{error}\n\n"
                "Relevant message definitions:\n{data_types}\n\n"
                "Return ONLY the fixed script in a markdown ```python``` block."
            ),
        )
        fix_llm = ChatOpenAI(
            model_name=self.model,
            max_tokens=8000,
            temperature=0,
            openai_api_key=api_key,
            openai_api_base=OPENROUTER_BASE_URL,
        )
        self._fix_chain = fix_prompt | fix_llm | StrOutputParser()

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def extract_code_snippets(text: str) -> list:
        """
        Extract fenced code blocks from *text*.

        Falls back to returning [text] when no fences are found.
        """
        snippets = re.findall(r"```.*?\n(.*?)```", text, re.DOTALL | re.MULTILINE)
        return snippets if snippets else [text]

    @staticmethod
    def write_plot_script(filename: str, text: str) -> None:
        """Write *text* to *filename*."""
        with open(filename, "w") as fh:
            fh.write(text)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_logfile_name(self, filename: str) -> None:
        """Register *filename* as the active log file and derive output paths."""
        ext = os.path.splitext(filename)[1].lower()
        if ext not in VALID_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{ext}'. "
                f"Expected one of: {', '.join(sorted(VALID_EXTENSIONS))}"
            )
        path = os.path.dirname(filename)
        self.logfile_name = filename
        self.script_path = os.path.join(path, "plot.py")
        self.plot_path = os.path.join(path, "plot.png")

    # ------------------------------------------------------------------
    # Log parsing
    # ------------------------------------------------------------------

    def parse_mavlink_log(self) -> str:
        """
        Parse the MAVLink log and build a persisted ChromaDB vector store.

        Returns:
            JSON string of all discovered message types and their fields.
        """
        if not self.logfile_name:
            raise RuntimeError("No log file set. Call set_logfile_name() first.")

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
                logger.info("Log parsing interrupted by user.")
                break
            except Exception as exc:
                logger.warning("Error reading MAVLink message: %s", exc)
                break

        self._create_embeddings(self.message_types)
        return json.dumps(self.message_types, indent=4)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def _create_embeddings(self, message_types: dict) -> None:
        """
        Embed each message type into a persisted ChromaDB collection
        using OpenRouter's OpenAI-compatible embeddings endpoint.
        """
        texts = [json.dumps({mt: message_types[mt]}) for mt in message_types]
        logger.info("Creating embeddings for %d message types.", len(texts))

        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=os.environ["OPENROUTER_API_KEY"],
            openai_api_base=OPENROUTER_BASE_URL,
        )

        persist_dir = os.path.join(
            os.path.dirname(self.logfile_name), "chroma_db"
        )
        self.db = Chroma.from_texts(
            texts,
            embeddings,
            persist_directory=persist_dir,
        )
        logger.info("ChromaDB persisted to %s", persist_dir)

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    def find_relevant_data_types(self, human_input: str) -> str:
        """Return a text block of message-type JSON most relevant to *human_input*."""
        if self.db is None:
            raise RuntimeError("Vector store not initialised. Call parse_mavlink_log() first.")
        docs = self.db.similarity_search(human_input)
        return "\n\n".join(doc.page_content for doc in docs)

    # ------------------------------------------------------------------
    # Plot generation
    # ------------------------------------------------------------------

    def create_plot(self, human_input: str, data_type_info_text: str) -> str:
        """Ask the LLM to write a plotting script and save it to disk."""
        history = (
            f"\n\nLast script generated:\n\n{self.last_code}" if self.last_code else ""
        )
        response = self._plot_chain.invoke({
            "data_types": data_type_info_text,
            "history": history,
            "file": self.logfile_name,
            "human_input": human_input,
            "output_file": self.plot_path,
        })
        logger.debug("LLM plot response:\n%s", response)
        code = self.extract_code_snippets(response)
        self.write_plot_script(self.script_path, code[0])
        self.last_code = code[0]
        return code[0]

    # ------------------------------------------------------------------
    # Script execution with self-healing
    # ------------------------------------------------------------------

    def attempt_to_fix_script(self, error_message: str) -> str:
        """
        Ask the LLM to fix the last failing script.

        Args:
            error_message: The stderr/exception text from the failed run.

        Returns:
            The fixed (or best-effort) script as a string.
        """
        with open(self.script_path, "r") as fh:
            script = fh.read()

        try:
            response = self._fix_chain.invoke({
                "data_types": json.dumps(self.message_types, indent=2),
                "error": error_message,
                "script": script,
            })
        except Exception as exc:
            logger.error("LLM fix request failed: %s", exc)
            return script

        logger.debug("LLM fix response:\n%s", response)
        fixed = self.extract_code_snippets(response)[0]
        self.write_plot_script(self.script_path, fixed)
        return fixed

    def run_script(self) -> tuple:
        """
        Execute the generated plot script.

        On failure, self-heals up to self.max_retries times.

        Returns:
            ([(None, (plot_path,))], last_code)
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                subprocess.check_output(
                    ["python", self.script_path], stderr=subprocess.STDOUT
                )
                logger.info("Script succeeded on attempt %d.", attempt)
                break
            except subprocess.CalledProcessError as exc:
                error_text = exc.output.decode(errors="replace")
                logger.warning(
                    "Script attempt %d/%d failed:\n%s", attempt, self.max_retries, error_text
                )
                if attempt < self.max_retries:
                    self.last_code = self.attempt_to_fix_script(error_text)
                else:
                    logger.error("All %d attempts exhausted.", self.max_retries)
                    self.last_code = (
                        f"# All {self.max_retries} fix attempts failed.\n"
                        f"# Last error:\n# {error_text}\n\n"
                        + self.last_code
                    )
            except Exception as exc:
                logger.error("Unexpected error running script: %s", exc)
                if attempt < self.max_retries:
                    self.last_code = self.attempt_to_fix_script(str(exc))
                else:
                    self.last_code = f"# Unexpected error: {exc}\n\n" + self.last_code
                    break

        return [[(None, (self.plot_path,))]], self.last_code
