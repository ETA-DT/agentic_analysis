import os
import sys
import json
import ast
import datetime
from datetime import datetime
import time
import re
import configparser
from pathlib import Path
from typing import Any, List, Mapping, Optional, Iterator

# # Importation des bibliothèques liées à SQLite
# __import__("pysqlite3")
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# Gestion des variables d'environnement
from dotenv import load_dotenv

load_dotenv()

# Définition des constantes pour les variables d'environnement
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY", "")
WATSONX_PROJECT_ID = os.getenv("PROJECT_ID", "")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com/")

os.environ["WATSONX_URL"] = WATSONX_URL
os.environ["WATSONX_APIKEY"] = WATSONX_APIKEY
os.environ["WATSONX_PROJECT_ID"] = WATSONX_PROJECT_ID

# Importation des bibliothèques liées à IBM Watsonx
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model, ModelInference
from ibm_watsonx_ai.metanames import (
    EmbedTextParamsMetaNames as EmbedParams,
    GenTextParamsMetaNames as GenParams,
)
from ibm_watsonx_ai.foundation_models.utils.enums import (
    EmbeddingTypes,
    ModelTypes,
    DecodingMethods,
)

credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_APIKEY)

# # Bibliothèques liées à Streamlit
# import streamlit as st
# from streamlit_file_browser import st_file_browser

# Bibliothèques de traitement de données
import pandas as pd
import mdpd

# Bibliothèques de traitement de documents
import docling
from docling.backend.msword_backend import MsWordDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import (
    ConversionResult,
    InputDocument,
    SectionHeaderItem,
)
from docling.document_converter import DocumentConverter

# Bibliothèques liées à Langchain et CrewAI
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool, AgentType, AgentExecutor, create_react_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_core.runnables import chain
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM, ChatWatsonx
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.agents.agent_toolkits import create_csv_agent

# Bibliothèques liées à TM1py (IBM Planning Analytics)
from TM1py.Services import TM1Service
from TM1py.Utils.Utils import (
    build_pandas_dataframe_from_cellset,
    build_cellset_from_pandas_dataframe,
)


def get_credentials():
    return {
        "url": "https://us-south.ml.cloud.ibm.com",
        "apikey": os.getenv("WATSONX_APIKEY", ""),
    }


class DoclingPDFLoader(BaseLoader):
    def __init__(self, file_path: str | list[str]) -> None:
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._converter = DocumentConverter()

    def lazy_load(self) -> Iterator[LCDocument]:
        for source in self._file_paths:
            dl_doc = self._converter.convert(source).document
            text = dl_doc.export_to_markdown()
            yield LCDocument(page_content=text)


def create_documents(path):
    loader = DoclingPDFLoader(file_path=path)
    file_name = loader._converter.convert(path).document.name
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    docs = loader.load()
    docs[0].metadata = {"filename": file_name}
    return text_splitter.split_documents(docs)


embed_params = {
    EmbedParams.TRUNCATE_INPUT_TOKENS: 512,
    EmbedParams.RETURN_OPTIONS: {"input_text": True},
}

embeddings = WatsonxEmbeddings(
    model_id="intfloat/multilingual-e5-large",
    url=os.getenv("WATSONX_URL", ""),
    apikey=os.getenv("WATSONX_APIKEY", ""),
    project_id=os.getenv("WATSONX_PROJECT_ID", ""),
    params=embed_params,
)


def create_vectorstore(path):
    texts = create_documents(path)
    return Chroma.from_documents(texts, embeddings)


doc_folder = "D:/Applications/Tm1/Tango_Core_Model/Data/Python_Scripts/PAAgenticAnalysis/agentic_analysis/Documents RAG"
# doc_folder = "D:/Applications/Tm1/Tango_Core_Model/Data/Python_Scripts/PAAgenticAnalysis/agentic_analysis/Notes de Cadrage (target 2025)"


def update_doc_folder(folder):
    folder_list_dir = os.listdir(folder)
    folder_list_dir = [f for f in folder_list_dir if f != ".gitignore"]
    return folder_list_dir


def add_documents(vectorbase, path):
    new_documents = create_documents(path)
    vectorbase.add_documents(new_documents)


docsearch = None
files_name = update_doc_folder(doc_folder)

document_dataframe = []

if files_name:
    file_path = os.path.join(doc_folder, files_name[0])
    docsearch = create_vectorstore(file_path)

for filename in files_name:
    if docsearch:
        if filename not in [
            doc["filename"] for doc in docsearch.get()["metadatas"] if doc
        ]:
            add_documents(docsearch, os.path.join(doc_folder, filename))

document_dataframe = list(
    set([doc["filename"] for doc in docsearch.get()["metadatas"] if doc])
)

pd.DataFrame({"Documents": document_dataframe}).to_csv("document_list.csv", index=False)

# to keep track of tasks performed by agents
task_values = []


# define current directory
def set_current_directory():
    abspath = os.path.abspath(__file__)  # file absolute path
    directory = os.path.dirname(abspath)  # current file parent directory
    os.chdir(directory)
    return directory


CURRENT_DIRECTORY = set_current_directory()
config = configparser.ConfigParser()
config.read("config.ini")

WATSONX_LLAMA3_MODEL_ID = "watsonx/meta-llama/llama-3-2-3b-instruct"
model_id = "meta-llama/llama-3-2-3b-instruct"
model_id_mistral = "mistralai/mixtral-8x7b-instruct-v01"

parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 3000,
    "temperature": 0,
    "top_k": 25,
    "top_p": 1,
    "repetition_penalty": 1,
}

parameters_llama = {
    "decoding_method": "greedy",
    "max_new_tokens": 5000,
    "temperature": 0.5,
    "top_k": 25,
    "top_p": 1,
    "repetition_penalty": 1,
}

rag_model_id = "mistralai/mistral-large"

# Defining the model parameters

rag_model_parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 3000,
    "min_new_tokens": 1,
    "repetition_penalty": 1,
}

## Defining the Model object

rag_model = ModelInference(
    model_id=rag_model_id,
    params=rag_model_parameters,
    credentials=get_credentials(),
    project_id=WATSONX_PROJECT_ID,
)

# ibm_model = Model(
#     model_id=model_id,
#     params=parameters,
#     credentials=credentials,
#     project_id=WATSONX_PROJECT_ID,
# )

pandas_llm = WatsonxLLM(
    model_id="meta-llama/llama-3-405b-instruct",  # codellama/codellama-34b-instruct-hf", #"mistralai/mistral-large", #"google/flan-t5-xxl", "ibm/granite-34b-code-instruct",
    url=get_credentials().get("url"),
    apikey=get_credentials().get("apikey"),
    project_id=WATSONX_PROJECT_ID,
    params=parameters,
)

# Create the function calling llm
function_calling_llm = WatsonxLLM(
    model_id="mistralai/mistral-large",
    url="https://us-south.ml.cloud.ibm.com",
    params=parameters,
    project_id=os.getenv("PROJECT_ID", ""),
)

llm = LLM(
    model="watsonx/meta-llama/llama-3-405b-instruct",
    base_url="https://api.watsonx.ai/v1",
    temperature=0.03,
    max_tokens=5000,
    # frequency_penalty=0.1,
    # presence_penalty=0.1,
    seed=42,
    top_p=0.8,
)


# Streamlit interface
def main():
    output_cube_name = "TM1py_output"
    # définir et se connecter à l'instance tm1
    with TM1Service(**config["tango_core_model"]) as tm1:

        def elements_aliase(dim_name, elem_list, alias_name):
            elem_aliases_dict = tm1.elements.get_attribute_of_elements(
                dimension_name=dim_name,
                hierarchy_name=dim_name,
                attribute=alias_name,
                elements=elem_list,
            )
            elem_aliases = list(elem_aliases_dict.values())
            return elem_aliases

        cube_name = tm1.cells.get_value(
            cube_name=output_cube_name,
            elements=[
                ("TM1py_Scripts", "AgenticAnalysis"),
                ("TM1py_outputs", "NomCube"),
            ],
        )
        view_name = tm1.cells.get_value(
            cube_name=output_cube_name,
            elements=[
                ("TM1py_Scripts", "AgenticAnalysis"),
                ("TM1py_outputs", "NomVue"),
            ],
        )
        # dimensions de la vue
        measure_dim = tm1.cubes.get_measure_dimension(cube_name=cube_name)
        country_dim = "Pays"
        period_dim = "Period"

        # vérification d'existence d'alias pour la dimension indicateur (autre que l'attribut format)
        measure_alias_names = tm1.elements.get_element_attribute_names(
            measure_dim, measure_dim
        )
        period_alias_names = tm1.elements.get_element_attribute_names(
            period_dim, period_dim
        )
        period_alias = tm1.elements.get_attribute_of_elements(
            period_dim, period_dim, "English"
        )
        countries = tm1.elements.get_leaf_element_names(
            dimension_name=country_dim, hierarchy_name=country_dim
        )  # liste de tous les pays feuilles de la dimension indicateurs

        all_indicators = tm1.elements.get_leaf_element_names(
            dimension_name=measure_dim, hierarchy_name=measure_dim
        )  # liste de tous les indicateurs feuilles de la dimension indicateurs

    all_indicators_english = elements_aliase(
        "Indicateurs_Activité", all_indicators, "English"
    )
    countries_english = elements_aliase("Pays", countries, "English")

    def round_2(number):
        """
        Converts str to float and round to the tenth
        """
        try:
            return round(float(number), 1)
        except:
            print("not a str")

    def rename_period_alias(df):
        """
        Renames the periods column with period alias
        """
        df.rename(columns=period_alias, inplace=True)

    def preprocessing(df, cube_name):
        """
        Preprocessing the dataframe
        """
        df = df.fillna(0)
        rename_period_alias(df)
        numeric_columns = df.select_dtypes(include=float).columns.tolist()
        df[numeric_columns] = df[numeric_columns].apply(round_2)
        # df = df.set_index(df.columns[0])
        for dim in tm1.cubes.get_dimension_names(cube_name=cube_name):
            if dim in df.columns:
                df[dim] = df[dim].apply(
                    lambda x: tm1.elements.get_attribute_of_elements(
                        dimension_name=dim,
                        hierarchy_name=dim,
                        attribute="English",
                        elements=[x],
                    )[x]
                )
        return df

    def dimension_of_element(cube_name, view_name, element):
        cellset_sample = list(
            tm1.cubes.cells.execute_view(
                cube_name=cube_name, view_name=view_name
            ).keys()
        )[0]
        for dim in cellset_sample:
            if element in dim:
                first_bracket_index = dim.index("[")
                second_bracket_index = dim.index("]")
                return dim[first_bracket_index + 1 : second_bracket_index]

    def get_context(cube_name, view_name):
        list_context = tm1.cubes.cells.execute_view_ui_dygraph(
            cube_name=cube_name, view_name=view_name, skip_zeros=False
        )["titles"][0]["name"]
        list_context = list_context.split(" / ")
        context = "For the following dataframe, the "
        for i, element in enumerate(list_context):
            elem_dim = dimension_of_element(cube_name, view_name, element)

            alias_name = tm1.elements.get_alias_element_attributes(
                dimension_name=elem_dim, hierarchy_name=elem_dim
            )[-1]
            try:
                elem_alias = list(
                    tm1.elements.get_attribute_of_elements(
                        dimension_name=elem_dim,
                        hierarchy_name=elem_dim,
                        elements=[element],
                        attribute=alias_name,
                    ).values()
                )[0]
            except:
                elem_alias = element
            context += f"{elem_dim} is {elem_alias}"
            if i < len(list_context) - 1:
                context += " and the "
        context = context.replace(measure_dim, "data displayed")
        return context

    def view_dataframe(cube_name, view_name):
        return preprocessing(
            tm1.cubes.cells.execute_view_dataframe_shaped(
                cube_name=cube_name, view_name=view_name, skip_zeros=False
            ),
            cube_name,
        )

    current_dataframe = view_dataframe(cube_name, view_name)

    view_indicators_english = elements_aliase(
        dim_name="Indicateurs_Activité",
        elem_list=list(current_dataframe["Indicateurs_Activité"].unique()),
        alias_name="English",
    )
    view_countries_english = elements_aliase(
        dim_name="Pays",
        elem_list=list(current_dataframe["Pays"].unique()),
        alias_name="English",
    )

    def dataframe_prompt_input(cube_name, view_name):
        dataframe = view_dataframe(cube_name, view_name)
        dataframe_md = dataframe.to_markdown()
        return dataframe_md

    def dataframe_enriched_prompt_input(cube_name, view_name):
        context = f"""{get_context(cube_name,view_name)} \nDataframe:\n{dataframe_prompt_input(cube_name,view_name)}"""
        return context

    # Melted version of tm1 dataframe in english
    def round_value(df):
        df["Value"] = df["Value"].apply(round_2)

    def to_english(df, cube_name):
        for dim in tm1.cubes.get_dimension_names(cube_name=cube_name):
            if dim in df.columns and dim != "Period":
                df[dim] = df[dim].apply(
                    lambda x: tm1.elements.get_attribute_of_elements(
                        dimension_name=dim,
                        hierarchy_name=dim,
                        attribute="English",
                        elements=[x],
                    )[x]
                )
        return df

    def preprocessing_melt_english(cube_name, view_name):
        df = tm1.cells.execute_view_dataframe(cube_name, view_name)
        round_value(df)
        to_english(df, cube_name)
        df.rename(columns={"Period": "Month"})
        return df

    preprocessed_dataframe = preprocessing_melt_english(cube_name, view_name)

    def create_crewai_setup(cube_name, view_name):
        ### RAG Setup

        pythonREPL = PythonREPLTool()
        duckduckgo_search = DuckDuckGoSearchRun()

        user_question = tm1.cells.get_value(
            cube_name=output_cube_name,
            elements=f"AgenticAnalysis;;Question",
            element_separator=";;",
        )

        if user_question != "":

            @tool
            def retriever(query: str) -> List[LCDocument]:
                """
                Retrieve relevant contextual documents and generate an answer to the given query.

                This tool performs a semantic similarity search over a document index to retrieve
                the top-k most relevant documents for a given natural language query. and
                creates a clean context paragraph. This context is then used by
                a Retrieval-Augmented Generation (RAG) model to generate a response.

                Parameters:
                    query (str): A natural language question or prompt requiring contextual information.

                Returns:
                    List[LCDocument]: A list containing a single LCDocument where `page_content` is the
                    generated response from the RAG model, and `metadata` includes the relevance scores
                    of the underlying documents used for context.

                Usage:
                    Use this tool when a question requires factual or context-based information retrieved
                    from a knowledge base. The tool both retrieves relevant supporting context and answers
                    the query based on that information.
                """
                docs, scores = zip(
                    *docsearch.similarity_search_with_relevance_scores(
                        query, score_threshold=0.5, k=4
                    )
                )
                for doc, score in zip(docs, scores):
                    doc.metadata["score"] = score

                # gather all retrieved documents into single string paragraph
                # removed_n = [
                #     doc.page_content.replace("\n", " ") for doc in docs
                # ]  # remove \n
                unique_retrieval = list(
                    set([doc.page_content for doc in docs])
                )  # remove duplicates documents
                retrieved_context = "\n".join(unique_retrieval)

                rag_prompt_input = f"""Based on the retrieved CHUNKS, answer to the QUESTION.
                                    QUESTION: {query} ?
                                    CHUNKS: {retrieved_context}"""
                rag_response = rag_model.generate_text(
                    prompt=rag_prompt_input, guardrails=False
                )

                return rag_response

        else:

            @tool
            def retriever(query=user_question) -> str:
                """
                Retrieve relevant contextual documents and generate an answer to the given query.

                This tool performs a semantic similarity search over documents that is used by
                a Retrieval-Augmented Generation (RAG) model to generate a response.

                Parameters:
                    query (str): A natural language question or prompt requiring contextual information.

                Returns:
                    Returns the RAG response to the input query.

                Usage:
                    Use this tool with a very explicit query to retrieve factual or context-based information
                    from documents. This tool both retrieves relevant supporting context and answers
                    the query based on that information.
                """
                docs, scores = zip(
                    *docsearch.similarity_search_with_relevance_scores(
                        query, score_threshold=0.5, k=4
                    )
                )
                for doc, score in zip(docs, scores):
                    doc.metadata["score"] = score

                # gather all retrieved documents into single string paragraph
                # removed_n = [
                #     doc.page_content.replace("\n", " ") for doc in docs
                # ]  # remove \n
                unique_retrieval = list(
                    set([doc.page_content for doc in docs])
                )  # remove duplicates documents
                retrieved_context = "\n".join(unique_retrieval)

                rag_prompt_input = f"Based on the retrieved chunks, {query} ? CHUNKS: {retrieved_context}"
                rag_response = rag_model.generate_text(
                    prompt=rag_prompt_input, guardrails=False
                )

                return rag_response

        @tool
        def dataframe_creator(
            query: str,
            df=preprocessed_dataframe,
        ) -> str:
            """
            Generate an answer or perform an operation on a pandas DataFrame based on a natural language query.

            This tool uses a language model agent to interpret and execute user queries on a provided pandas
            DataFrame. It supports querying, filtering, summarization, and transformations by converting
            natural language instructions into code that operates on the DataFrame.

            Parameters:
                query (str): A natural language question or instruction related to the DataFrame.
                df (pd.DataFrame, optional): The DataFrame to run the query on. Defaults to `preprocessed_dataframe`.

            Returns:
                str: The textual output generated by the agent after interpreting and executing the query.

            Usage:
                Use this tool to interact with structured tabular data using natural language, especially
                when quick insights, filtering, or calculations are needed.
            """
            agent = create_pandas_dataframe_agent(
                pandas_llm,
                df,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                allow_dangerous_code=True,
                include_df_in_prompt=True,
                early_stopping_method="force",
                number_of_head_rows=len(df),
                max_iterations=2,
            )
            response = agent.invoke(
                query, handle_parsing_errors=True, return_intermediate_steps=True
            )
            return response["output"]

        @tool
        def difference(a: float, b: float) -> float:
            """
            Calculate the difference between two floating-point numbers.

            This function subtracts the second number (`b`) from the first number (`a`) and returns the result.

            Args:
                a (float): The first number (minuend).
                b (float): The second number (subtrahend).

            Returns:
                float: The result of subtracting `b` from `a` (i.e., `a - b`).
            """
            return a - b

        @tool
        def division(a: float, b: float) -> float:
            """
            Calculate the quotient of two floating-point numbers.

            Divides the first number (`a`) by the second number (`b`).
            The function requires the divisor (`b`) to be non-zero to avoid division errors.
            This function is commonly used for calculating ratios, proportions, or scaling factors.

            Args:
                a (float): The dividend (number to be divided).
                b (float): The divisor (number to divide by). Must be non-zero.

            Returns:
                float: The result of dividing `a` by `b` (i.e., `a / b`).
            """
            if b != 0:
                return a / b

        @tool
        def convert_period_to_year(period: str) -> str:
            """
            Convert a period string in the format 'YYYY.MM' to a 4-digit year string.

            This tool parses a date string representing a year and month, and extracts
            only the year component as a string.

            Parameters:
                period (str): A date string in the format 'YYYY.MM' (e.g., '2023.05').

            Returns:
                str: The 4-digit year extracted from the input period (e.g., '2023').

            Usage:
                Use this tool when you need to normalize or simplify period values to
                just the year for reporting, filtering, or aggregation purposes.
            """
            return datetime.strptime(period, "%Y.%M").date().strftime("%Y")

        # Agents Definition

        # DataCore Analyst
        DataCore = Agent(
            role="Business Performance Analyst",
            backstory="A accurate data scientist specializing in calculations with data values from multidimensional dataframes.",
            goal="Calculate relevant values such as maximum, minimum, total year by indicator and by country",
            tools=[dataframe_creator, convert_period_to_year],
            memory=True,
            verbose=True,
            allow_delegation=True,
            llm=llm,
            max_iter=4,
            function_calling_llm=function_calling_llm,
        )

        entity_identifier = Agent(
            role="Entity Extractor",
            backstory="A Entity Extractor expert skilled in extracting the required entities.",
            goal="Retrieve country-indicator pairs that could only be formed from the indicators and countries mentionned in the context.",
            verbose=True,
            allow_delegation=True,
            llm=llm,
            max_iter=2,
            function_calling_llm=function_calling_llm,
        )

        ## DocuMentor Analyst
        DocuMentor = Agent(
            role="Document Analyst",
            backstory="A Document Analyst expert skilled in extracting insights from internal business documents.",
            goal="Retrieve targets from internal documents regarding only the country-indicator pairs that could only be formed from the indicators and countries mentionned in the context.",
            verbose=True,
            allow_delegation=True,
            tools=[retriever],
            llm=llm,
            max_iter=3,
            function_calling_llm=function_calling_llm,
        )

        ## Gap Analyst
        GapAnalyst = Agent(
            role="Strategic Gap Quantifier",
            backstory="An operations researcher specializied in gap analysis between target objectives and actual performance.",
            goal="Calculate quantitative gaps between business performance data and target values for each indicator by country",
            verbose=True,
            allow_delegation=False,
            llm=llm,
            function_calling_llm=function_calling_llm,
            tools=[difference],
            max_iter=3,
            memory=True,
        )

        ## Insight Synthesizer
        InsightSynthesizer = Agent(
            role="Insight Synthesizer",
            backstory="A strategist blending AI-driven analytics with business insights.",
            goal="List of percent gaps to reduce and recommandation actions to tackle these identified gaps, and the objective value to achieve for each country-indicator pair",
            verbose=True,
            max_iter=2,
            llm=llm,
            function_calling_llm=function_calling_llm,
        )

        ## Strategy Navigator
        StrategyNavigator = Agent(
            role="Strategy Navigator",
            backstory="A business strategist ensuring insights align with company goals and market trends.",
            goal="Ensure actions directly address indicators blocking strategic goals and prioritize them.",
            # tools=[ai_tool],
            verbose=True,
            max_iter=2,
            llm=llm,
            function_calling_llm=function_calling_llm,
        )

        # Task Definitions
        data_task = Task(
            description="Analyze data to calculate the following information: annual minimum, maximum, sum grouped by country, indicator and year. You should never recreate the dataframe given as an input.",
            agent=DataCore,
            expected_output="Annual report of the performance by indicator and by country that appear in the dataframe input. You should never make up new indicators or new countries that does not appear in the dataframe",
            output_file="tasks_outputs/data_task.md",
        )

        identify_task = Task(
            description="Extract the country-indicator pairs mentionned in the context",
            agent=entity_identifier,
            expected_output="A structured list of country-indicator pairs that could only be formed by the set of countries and indicators in the context.",
            context=[data_task],
        )

        doc_task = Task(
            description=(
                "Analyze internal documents to find targets for the exact country-indicator pairs listed in the context. "
                "Only consider pairs where both the country and the indicator exactly match the names provided in the context. "
                "Do not include or mention any other countries or indicators. "
                "Do not guess, infer, or link similar indicators — matches must be exact."
            ),
            agent=DocuMentor,
            expected_output=(
                "A list of targets for each country-indicator pair from the context "
                "only if the exact pair is clearly stated in the internal documents. "
                "Do not make up targets, and do not mix indicators. "
                "Only include targets for exact country and indicator matches."
            ),
            context=[identify_task],
            output_file="tasks_outputs/doc_task.md",
        )

        gap_task = Task(
            description=f"For each country-indicator pair stated in the Context, calculate the gap between its target annual value and the annual current value. If an country-indicator pair does not have an attached target value, skip its gap calculation.",
            agent=GapAnalyst,
            expected_output="""
                Gap Analysis Report:
                - Target: [Target value]
                - Current Performance: [Metric from data]
                - Gap Size: [Quantitative difference (target - current)]
                - Percent Gap: [Quantitative difference ratio (Gap Size / Current)]
                - Criticality Score: [1-5 rating]
            """,
            context=[identify_task, data_task, doc_task],
            tools=[division],
            output_file="tasks_outputs/gap_analysis.md",
        )

        insight_task = Task(
            description="""Generate a report of actionable recommendations to reduce the identified percent gaps. 
                        The recommendations must ensure that the **total percent gap identified** is applied on **every single month's values**. 
                        Each action should align with the total percent gap and reflect the same reduction percentage across all months for a given country.""",
            agent=InsightSynthesizer,
            expected_output=f"""**Structured Output:**
                            1. A list of recommended actions to reduce the identified gaps with the target value to achieve for each country-indicator pair.
                            2. Actions should be specific, measurable, and tied directly to the total percent gap identified in the context.
                            3. Ensure the same percentage reduction is applied to every month for a given country, reflecting the total percent gap uniformly across all the months of the year.""",
            context=[gap_task],
            output_file="tasks_outputs/insight_task.md",
        )

        strategy_task = Task(
            description="Based on the identified gaps, elaborate a strategic annual plan with recommended actions to achieve the business targets.",
            agent=StrategyNavigator,
            expected_output="A prioritized action plan aligning insights with internal business goals, focusing on the identified gaps",
            context=[insight_task],
            output_file="tasks_outputs/strategy_task.md",
        )

        # Crew Assembly
        crew = Crew(
            agents=[
                DataCore,
                entity_identifier,
                DocuMentor,
                GapAnalyst,
                InsightSynthesizer,
                StrategyNavigator,
                # TechIntegrator,
            ],
            tasks=[
                data_task,
                identify_task,
                doc_task,
                gap_task,
                insight_task,
                strategy_task,
                # tech_task,
            ],
            verbose=True,
            process=Process.sequential,
        )

        return crew

        # Run the Crew

    crew = create_crewai_setup(cube_name, view_name)
    crew_result = crew.kickoff()

    tm1.cells.write_value(
        crew_result,
        cube_name="TM1py_output",
        element_tuple=["AgenticAnalysis", "Results"],
    )

    extract_model_id = "mistralai/mistral-large"

    # Defining the model parameters

    extract_model_parameters = {
        "decoding_method": "greedy",
        "max_new_tokens": 6000,
        "min_new_tokens": 1,
        "stop_sequences": ["\n\n"],
        "repetition_penalty": 1,
    }

    ## Defining the Model object

    extract_model = Model(
        model_id=extract_model_id,
        params=extract_model_parameters,
        credentials=get_credentials(),
        project_id=WATSONX_PROJECT_ID,
    )

    def extract_indicators_from_text(text):
        # Regex pour capturer les noms d'indicateurs après les puces
        pattern = r">\s*(.*)"
        found_indicators = re.findall(pattern, text)
        return found_indicators

    def match_indicators(found_indicators, reference_indicators):
        # Comparer les éléments extraits avec ceux de la liste de référence
        matched_indicators = [
            indicator
            for indicator in found_indicators
            if indicator in reference_indicators
        ]
        return matched_indicators

    extract_prompt_input_percent = f"""Here is a strict and exhaustive list of indicators:  
                                {view_indicators_english}  

                                Here is a list of countries:  
                                {view_countries_english}  

                                Extract only the indicators, the percentage of increase/decrease, the associated countries, and the months recommended for modification in this text.  

                                ### Strict Extraction Rules:  
                                1. **Convert periods into months**:  
                                - Q1 → January, February, March  
                                - Q2 → April, May, June  
                                - Q3 → July, August, September  
                                - Q4 → October, November, December  
                                - "Beginning of the year" → January, February, March  
                                - "End of the year" → October, November, December  
                                - "Summer" → June, July, August  
                                - "Winter" → December, January, February  

                                2. **Extract an indicator only if it is exactly in the provided list**  

                                3. **Keep the exact name** of the indicators  

                                4. **Determine Percentage Sign Based on Context**:  
                                - Use **positive percentages** (`p%`) for indicators like sales, revenue, or profit when terms like "increase," "improve," or "boost" are used.
                                - Use **negative percentages** (`-p%`) for indicators like costs, expenses, or losses when terms like "reduce," "improve," "lower," or "cut" are used.  

                                5. **Formatting**:  
                                - Each line must start with this symbol `>`  
                                - For month ranges (e.g., Q1-Q2), list all relevant months  

                                6. **Country/Region Hierarchy**:  
                                - If a group of countries is mentioned (e.g., Scandinavia), break it down into individual countries from the provided list  

                                ### Examples to Clarify Context:  
                                1. **Improving Costs (Negative %)**:  
                                Text: "Improve Maintenance Costs by 5% in Q4 in Scandinavia"  
                                Output:  
                                > {{'indicator':'Maintenance Costs','percent':'-5','country':'Finland','month':'October'}}  
                                > {{'indicator':'Maintenance Costs','percent':'-5','country':'Finland','month':'November'}}  
                                > [...]  

                                2. **Improving Sales (Positive %)**:  
                                Text: "Boost Sales Revenue by 10% in Q1 in France"  
                                Output:  
                                > {{'indicator':'Sales Revenue','percent':'10','country':'France','month':'January'}}  
                                > {{'indicator':'Sales Revenue','percent':'10','country':'France','month':'February'}}  
                                > [...]  

                                ### Text to Process:  
                                {crew_result}  
                                Extracted indicators: """

    extracted_percent = extract_model.generate_text(
        prompt=extract_prompt_input_percent, guardrails=False
    )
    found_percent = extract_indicators_from_text(extracted_percent)
    # matched_percent_picklist = "static::" + ":".join(found_percent)

    print("\nextracted_percent\n")
    print(extracted_percent)
    print("\nfound_percent\n")
    print(found_percent)

    def remove_incomplete_extraction(extraction):
        if extraction[-1][-1] != "}":
            del extraction[-1]

    if found_percent:
        remove_incomplete_extraction(found_percent)

    indicator_percent_country = [ast.literal_eval(val) for val in found_percent]

    print("\nindicator_percent_country\n")
    print(indicator_percent_country)

    indicateurs_trouves = list(
        set([cell["indicator"] for cell in indicator_percent_country])
    )
    print("\nindicateurs_trouves\n")
    print(indicateurs_trouves)
    matched_extracted_indicators = match_indicators(
        indicateurs_trouves, all_indicators_english
    )
    print("\nmatched_extracted_indicators\n")
    print(matched_extracted_indicators)

    if indicator_percent_country:

        def update_subset(subset_name, dimension_name, hierarchy_name, matched):
            indicator_subset = tm1.subsets.get(
                subset_name, dimension_name, hierarchy_name
            )
            indicator_subset.elements = []
            tm1.subsets.update(indicator_subset)
            indicator_subset.add_elements(matched)
            tm1.subsets.update(indicator_subset)

        print("SUBSET Indicateurs AVANT UPDATE")
        print(
            tm1.subsets.get_element_names(
                "Indicateurs_Activité", "Indicateurs_Activité", "IndicatorToModify"
            )
        )
        update_subset(
            "IndicatorToModify",
            "Indicateurs_Activité",
            "Indicateurs_Activité",
            list(set(matched_extracted_indicators)),
        )
        print("SUBSET Indicateur APRES UPDATE")
        print(
            tm1.subsets.get_element_names(
                "Indicateurs_Activité", "Indicateurs_Activité", "IndicatorToModify"
            )
        )
        pays_trouves = list(
            set(
                [
                    cell["country"]
                    for cell in indicator_percent_country
                    if "country" in cell.keys()
                    and tm1.elements.exists(
                        dimension_name=country_dim,
                        hierarchy_name=country_dim,
                        element_name=cell["country"],
                    )
                ]
            )
        )

        mois_trouves = list(
            set(
                [
                    cell["month"]
                    for cell in indicator_percent_country
                    if "month" in cell.keys()
                    and tm1.elements.exists(
                        dimension_name=period_dim,
                        hierarchy_name=period_dim,
                        element_name=cell["month"],
                    )
                ]
            )
        )

        def check_if_numeric(cur_string):
            symbols = ["-", "+", ",", "."]
            for symbol in symbols:
                cur_string = cur_string.replace(symbol, "")
            return cur_string.isnumeric()

        pourcentages_trouves = list(
            set(
                [
                    cell["percent"]
                    for cell in indicator_percent_country
                    if "percent" in cell.keys() and check_if_numeric(cell["percent"])
                ]
            )
        )

        print("\npays_trouves\n")
        print(list(set(pays_trouves)))

        print("\nmois_trouves\n")
        print(list(set(mois_trouves)))

        print("\npourcentages_trouves\n")
        print(pourcentages_trouves)

        print("SUBSET PAYS TROUVES AVANT UPDATE")
        print(tm1.subsets.get_element_names("Pays", "Pays", "Pays_Subset"))

        print("SUBSET PAYS CARTE TROUVES AVANT UPDATE")
        print(tm1.subsets.get_element_names("Pays", "Pays", "PaysExtraitsPourCarte"))

        update_subset("Pays_Subset", "Pays", "Pays", pays_trouves)
        update_subset("PaysExtraitsPourCarte", "Pays", "Pays", pays_trouves)

        print("SUBSET PAYS TROUVES APRES UPDATE")
        print(tm1.subsets.get_element_names("Pays", "Pays", "Pays_Subset"))

        print("SUBSET PAYS CARTE TROUVES APRES UPDATE")
        print(tm1.subsets.get_element_names("Pays", "Pays", "PaysExtraitsPourCarte"))

        update_subset(
            "IndicatorToModify",
            "Indicateurs_Activité",
            "Indicateurs_Activité",
            list(set(matched_extracted_indicators)),
        )
        print("SUBSET Indicateur APRES UPDATE")
        print(
            tm1.subsets.get_element_names(
                "Indicateurs_Activité", "Indicateurs_Activité", "IndicatorToModify"
            )
        )

        output_cube_name = "TM1py_output"
        cube_dimensions_names = tm1.cubes.get_dimension_names(cube_name=cube_name)

        year = "2025"
        all_months = {
            "january": "01",
            "february": "02",
            "march": "03",
            "april": "04",
            "may": "05",
            "june": "06",
            "july": "07",
            "august": "08",
            "september": "09",
            "october": "10",
            "november": "11",
            "december": "12",
        }

        if indicator_percent_country:
            for target in indicator_percent_country:
                if "percent" in target.keys() and check_if_numeric(target["percent"]):
                    percent = target["percent"]
                    new_value = float(percent) / 100
                    print(new_value)
                else:
                    continue
                if "country" in target.keys() and tm1.elements.exists(
                    dimension_name="Pays",
                    hierarchy_name="Pays",
                    element_name=target["country"],
                ):
                    target_country = target["country"]
                else:
                    continue
                if "indicator" in target.keys() and tm1.elements.exists(
                    dimension_name="Indicateurs_Activité",
                    hierarchy_name="Indicateurs_Activité",
                    element_name=target["indicator"],
                ):
                    target_indicator = target["indicator"]
                else:
                    continue
                if "month" in target.keys():
                    mois = target["month"]
                    if str(mois).lower() in all_months.keys():
                        tm1.cells.write(
                            cube_name=cube_name,
                            cellset_as_dict={
                                (
                                    "BUDG_VC_AJUST%",
                                    year + "." + all_months[str(mois).lower()],
                                    target_country,
                                    target_indicator,
                                ): new_value,
                            },
                        )
                    # print(percent, mois, target_country, target_indicator)
                else:
                    for period in tm1.subsets.get_element_names(
                        "Period", "Period", year + "_mois"
                    ):  # à généraliser pour récupérer le subset de period d'une vue donnée
                        old_value = tm1.cells.get_value(
                            cube_name=cube_name,
                            elements=f"BUDG_VC;;{period};;{target_country};;{target_indicator}",
                            element_separator=";;",
                        )
                        tm1.cells.write(
                            cube_name=cube_name,
                            cellset_as_dict={
                                (
                                    "BUDG_VC_AJUST%",
                                    period,
                                    target_country,
                                    target_indicator,
                                ): new_value
                            },
                            precision=4,
                        )
                    # print(percent, period, target_country, target_indicator)


if __name__ == "__main__":
    main()
