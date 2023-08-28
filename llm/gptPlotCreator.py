import re
import random
import linecache
import subprocess
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from PIL import Image

# Import required modules
from pymavlink import mavutil
import json

class PlotCreator:
    """
    PlotCreator is a class that generates Python scripts to plot MAVLink data
    provided by a user, leveraging OpenAI's models for conversational agents.
    """

    last_code = ""  # stores the last code generated
    logfile_name = ""
    script_path = ""
    plot_path = ""

    def __init__(self):
        """
        Initialize an instance of PlotCreator.
        """

        load_dotenv()  # load environment variables from a .env file

        self.model = os.getenv("OPENAI_MODEL")  # get the name of the OpenAI model to use

        # create an instance of ChatOpenAI with the specified model, maximum tokens, and temperature
        llm = ChatOpenAI(model_name=self.model, max_tokens=2000, temperature=0)

        # define the input variables and template for the prompt to generate Python scripts
        mavlink_data_prompt = PromptTemplate(
            input_variables=["data_types", "history", "human_input", "file", "output_file"],
            template="You are an AI conversation agent that will be used for generating python scripts to plot mavlink data provided by the user. Please create a python script using matplotlib and pymavlink's mavutil to plot the data provided by the user. Please do not explain the code just return the script. Please plot each independent variable over time in seconds. Please save the plot to file named {output_file} with at least 400 dpi and do not call plt.show(). please use blocking=false in your call to recv_match and be sure to break the loop if a msg in None. here are the relevant data types in the log:\n\n{data_types} \n\nChat History:\n{history} \n\nHUMAN: {human_input} \n\nplease read this data from the file {file}.",
        )

        # create an instance of LLMChain with the defined prompt and verbosity
        self.chain = LLMChain(verbose=True, llm=llm, prompt=mavlink_data_prompt)

    @staticmethod
    def extract_code_snippets(text):
        """
        Extracts code snippets from a text.

        This function searches the text for substrings enclosed in '```', which are assumed to be code snippets.

        Args:
            text (str): The text to search for code snippets.

        Returns:
            list: A list of code snippets found in the text. If no snippets are found, returns a list containing the original text.
        """

        pattern = r'```.*?\n(.*?)```'  # pattern to match code snippets enclosed in '```'
        # use regex to find all matches of the pattern in the text
        snippets = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
        if len(snippets) == 0:  # if no snippets were found
            snippets = [text]  # treat the entire text as a single snippet
        return snippets

    @staticmethod
    def write_plot_script(filename, text):
        """
        Writes a script to a file.

        Args:
            filename (str): The name of the file to write the script to.
            text (str): The script to write to the file.
        """

        with open(filename, 'w') as file:  # open the file for writing
            file.write(text)  # write the script to the file

    def attempt_to_fix_sctript(self, filename, error_message):
        """
        Attempts to fix a script that caused an error.

        Args:
            filename (str): The name of the file containing the script.
            error_message (str): The error message produced by the script.

        Returns:
            list: A list containing the fixed script, or the original script with an error message if it couldn't be fixed.
        """

        # create an instance of ChatOpenAI with the specified model, maximum tokens, and temperature
        llm = ChatOpenAI(model_name=self.model , max_tokens=8000, temperature=0)

        # define the input variables and template for the prompt to generate Python scripts
        fix_plot_script_template = PromptTemplate(
            input_variables=["data_types", "error", "script"],
            template="You are an AI agent that is designed to debug scripts created to plot mavlink data using matplotlib and pymavlink's mavutil. the following script produced this error: \n\n{script}\n\nThe error is: \n\n{error}\n\n Here are message definitions that are possibly relevant for the script:\n\n {data_types}\n\n. Please fix the script so that it produces the correct plot. please return the fixed script in a markdown code block.",
        )

        # read script from file
        with open(filename, 'r') as file:
            script = file.read()

        # create an instance of LLMChain with the defined prompt and verbosity
        chain = LLMChain(verbose=True, llm=llm, prompt=fix_plot_script_template)
        try:
            response = chain.run({"data_types" : self.message_types, "error": error_message, "script": script})  # run the LLMChain with the error and script as input
        except:
            return "Sorry I couldn't fix the script. Here is the original script I tried:\n\n" + script
        print(response)
        code = PlotCreator.extract_code_snippets(response)  # extract the fixed script from the response
        PlotCreator.write_plot_script("plot.py", code[0])  # write the fixed script to a file

        # run the fixed script 
        try:
            subprocess.check_output(["python", self.script_path], stderr=subprocess.STDOUT)
        except:
            code[0] = "Sorry I was unable to fix the script.\nThis is my attempt to fix it:\n\n" + code[0]
        return code

    def set_logfile_name(self, filename):
        """
        Set the name of the log file.
        
        :param filename: The name of the log file.
        :type filename: str
        """
        # extract the path to the log file

        path = os.path.dirname(filename)
        self.logfile_name = filename
        self.script_path = os.path.join(path, "plot.py")
        self.plot_path = os.path.join(path, "plot.png")


    def find_relevant_data_types(self, human_input):
        # Search the database for documents that are similar to the human input
        docs = self.db.similarity_search(human_input)

        # Concatenate the content of the documents into a string
        data_type_info_text = ""
        for doc in docs:
            data_type_info_text += doc.page_content + "\n\n"
        
        return data_type_info_text

    def run_script(self):
        # Run the script and if it doesn't work, capture the output and call attempt_to_fix_script
        try:
            subprocess.check_output(["python", self.script_path], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(e.output.decode())
            code = self.attempt_to_fix_sctript(self.script_path, e.output.decode())
            self.last_code = code[0]

        except Exception as e:
            print(e)
            code = self.attempt_to_fix_sctript(self.script_path, str(e))
            self.last_code = code[0]


        # Return a list containing the filename of the plot and the code used to generate it
        return [[(None, (self.plot_path,))], self.last_code]

    def create_plot(self, human_input, data_type_info_text):
        """
        Create a plot based on the input provided by the human.
        
        :param human_input: Input provided by the human.
        :type human_input: str
        """

        # Create a history of generated scripts if one exists
        if self.last_code != "":
            history = "\n\nLast script generated:\n\n" + self.last_code
        else:
            history = ""


        # Generate a response by running the chain with the relevant data types, history, file name and human input
        response = self.chain.run({"data_types" : data_type_info_text, "history" : history, "file": self.logfile_name, "human_input": human_input, "output_file": self.plot_path})
        print(response)

        # Parse the code from the response 
        code = self.extract_code_snippets(response)

        # Write the code to a file named "plot.py"
        self.write_plot_script(self.script_path, code[0])

        # Store the code for the next iteration
        self.last_code = code[0]


        return code[0]


    def parse_mavlink_log(self):
        """
        Parse the MAVLink log to extract unique message types and their fields.
        
        :return: A JSON string representation of the unique message types and their fields.
        :rtype: str
        """
        # Initialize a dictionary to store unique message types and their fields
        self.message_types = {}

        # Establish a MAVLink connection
        mav_log = mavutil.mavlink_connection(self.logfile_name)

        # Loop through the log file and extract all unique message types
        while True:
            try:
                # Receive a message
                msg = mav_log.recv_match(blocking=False, type=None)
                # Check if we received a message
                if msg is None:
                    break

                # Store the unique message types and their fields in the dictionary
                if msg.get_type() not in self.message_types:
                    # Add the message type and its fields to the dictionary
                    self.message_types[msg.get_type()] = {
                        "count": 1,
                        "fields": {field: type(getattr(msg, field)).__name__ for field in msg.get_fieldnames()}
                    }
                else:
                    # Increment the count for this message type
                    self.message_types[msg.get_type()]["count"] += 1

            except KeyboardInterrupt:
                break
            except:
                print("Unknown error")
                break

        # Create embeddings for the message types
        self.create_embeddings(self.message_types)

        # Return a JSON string of the message types
        return json.dumps(self.message_types, indent=4)

    def create_embedding(self, texts):
        """
        Create OpenAI embeddings for a list of texts.
        
        :param texts: A list of texts to create embeddings for.
        :type texts: list of str
        """
        # Initialize a dictionary to store the embeddings
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.db = Chroma.from_texts(texts, embeddings)

    def create_embeddings(self, message_types):
        """
        Create OpenAI embeddings for a dictionary of message types.
        
        :param message_types: A dictionary of message types to create embeddings for.
        :type message_types: dict
        """
        print(message_types)

        # Convert the message types to a list of JSON strings
        texts = []
        for message_type in message_types:
            texts.append(json.dumps({ message_type : message_types[message_type]}))

        print(f"Texts: {texts}")
        # Create the embeddings
        self.create_embedding(texts)