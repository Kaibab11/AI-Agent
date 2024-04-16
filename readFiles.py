import os
import re
from pathlib import Path
from prompts import new_prompt, instruction_str
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.readers.file import PDFReader, FlatReader
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import PandasQueryEngine
import pandas as pd
from llama_index.core import Settings
from embedding import get_index

tools = []

# Define the directory containing your documents
folder_path = os.path.join(os.getcwd(), "data")

def fileLoader():
    # Loop through files in the directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if the path is a file (not a directory)
        if os.path.isfile(file_path):
            # Get the file extension
            splitname = os.path.splitext(filename)
            name = sanitize_name(splitname[0])
            extension = splitname[1]
            # Call the appropriate function based on the file extension
            if extension.lower() == ".txt":
                handle_txt(file_path, name)
            elif extension.lower() == ".csv":
                handle_csv(file_path, name)
            elif extension.lower() == ".pdf":
                handle_pdf(file_path, name)
            else:
                print("Unhandled file type:", file_path)

    return tools

def handle_txt(file_path, filename):
    file_path_obj = Path(file_path)
    documents = FlatReader().load_data(file=file_path_obj)
    text_index = get_index(documents, filename)
    text_engine = text_index.as_query_engine(streaming=True)

    tools.append(
        QueryEngineTool(
            query_engine=text_engine,
            metadata=ToolMetadata(
                name=filename,
                description=f"this gives information on the contents of the text file {filename}",
            ),
        )
    )

def handle_csv(file_path, filename):

    Settings.llm = Ollama(model="mistral")
    csv_df = pd.read_csv(file_path)
    csv_query_engine = PandasQueryEngine(
        df=csv_df, verbose=True, instruction_str=instruction_str, llm=Settings.llm
    )
    csv_query_engine.update_prompts({"pandas_prompt": new_prompt})

    tools.append(
        QueryEngineTool(
            query_engine=csv_query_engine,
            metadata=ToolMetadata(
                name=filename,
                description=f"this gives information on the contents of the csv file {filename}",
            ),
        )
    )

def handle_pdf(file_path, filename):

    pdf = PDFReader().load_data(file=file_path)
    pdf_index = get_index(pdf, filename)
    pdf_engine = pdf_index.as_query_engine(streaming=True)

    tools.append(
        QueryEngineTool(
            query_engine=pdf_engine,
            metadata=ToolMetadata(
                name=filename,
                description=f"this gives informtion on the contents of the pdf file {filename}",
            ),
        )
    )

def sanitize_name(name):
    # Remove any characters not allowed in collection names
    sanitized_name = re.sub(r'[^\w-]', '', name)
    # Ensure it's within the length limit
    sanitized_name = sanitized_name[:63]
    # Ensure it starts and ends with an alphanumeric character
    sanitized_name = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', sanitized_name)
    return sanitized_name




