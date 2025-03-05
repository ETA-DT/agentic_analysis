import os
import sys
import json
import ast
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
from ibm_watsonx_ai.foundation_models import Model
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

# Bibliothèques liées à Streamlit
import streamlit as st
from streamlit_file_browser import st_file_browser

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
    "decoding_method": "sample",
    "max_new_tokens": 3000,
    "temperature": 0.1,
    "top_k": 50,
    "top_p": 1,
    "repetition_penalty": 1,
}

parameters_llama = {
    "decoding_method": "sample",
    "max_new_tokens": 5000,
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 1,
    "repetition_penalty": 1,
}

ibm_model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=WATSONX_PROJECT_ID,
)

chat = ChatWatsonx(
    model_id="ibm/granite-34b-code-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id=WATSONX_PROJECT_ID,
    params=parameters,
)

pandas_llm = WatsonxLLM(
    model_id="meta-llama/llama-3-405b-instruct",  # codellama/codellama-34b-instruct-hf", #"mistralai/mistral-large", #"google/flan-t5-xxl", "ibm/granite-34b-code-instruct",
    url=get_credentials().get("url"),
    apikey=get_credentials().get("apikey"),
    project_id=WATSONX_PROJECT_ID,
    params=parameters,
)

llm_llama = WatsonxLLM(
    model_id="meta-llama/llama-3-405b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    params=parameters_llama,
    project_id=os.getenv("PROJECT_ID", ""),
)

llm_mistral = WatsonxLLM(
    model_id="mistralai/mistral-large",
    url="https://us-south.ml.cloud.ibm.com",
    params=parameters_llama,
    project_id=os.getenv("PROJECT_ID", ""),
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
    temperature=0.1,
    max_tokens=5000,
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
        for col in df.columns[-12:]:
            df[col] = df[col].apply(round_2)
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

    def create_crewai_setup(cube_name, view_name):
        ### RAG Setup

        pythonREPL = PythonREPLTool()
        duckduckgo_search = DuckDuckGoSearchRun()

        @tool
        def retriever(query: str) -> List[LCDocument]:
            """
            Retrieve relevant documents based on a given query.

            This function performs a similarity search against a document store using the provided query.
            It retrieves up to 4 documents that meet a relevance score threshold of 0.5. Each retrieved
            document's metadata is updated with its corresponding relevance score.

            Args:
                query (str): The input query string to search for relevant documents.

            Returns:
                List[Document]: A list of documents that are relevant to the query. Each document contains
                metadata with an added "score" key indicating its relevance score.
            """
            docs, scores = zip(
                *docsearch.similarity_search_with_relevance_scores(
                    query, score_threshold=0.3, k=6
                )
            )
            for doc, score in zip(docs, scores):
                doc.metadata["score"] = score

            # gather all retrieved documents into single string paragraph
            removed_n = [
                doc.page_content.replace("\n", " ") for doc in docs
            ]  # remove \n
            unique_retrieval = list(set(removed_n))  # remove duplicates documents
            retrieved_context = "\n".join(unique_retrieval)
            return retrieved_context

        @tool
        def dataframe_creator(query: str, df=current_dataframe) -> str:
            """
            Execute a query on a pandas DataFrame using an agent.

            This function uses a pandas DataFrame agent to process the given query. The agent is configured to
            allow dangerous code execution and provides verbose output during execution. The query result is
            returned as a string, with intermediate steps suppressed and parsing errors handled.

            Args:
                query (str): The query to be executed on the DataFrame.

            Returns:
                str: The result of the query execution as a string.
            """
            agent = create_pandas_dataframe_agent(
                pandas_llm,
                df,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                # suffix= "Always return a JSON dictionary that can be parsed into a data frame containing the requested information.",
                verbose=True,
                allow_dangerous_code=True,
                include_df_in_prompt=True,
                number_of_head_rows=len(df),
            )
            response = agent.invoke(
                query, handle_parsing_errors=True, return_intermediate_steps=False
            )
            return response["output"]

        # Agents Definition

        ## DataCore Analyst
        DataCore = Agent(
            role="DataCore Analyst",
            backstory="A data scientist with expertise in statistical modeling and business intelligence.",
            goal="Transform raw data into clean, structured information ready for AI insights, identifying the most critical intersections for targeted recommendations to improve entreprise performance.",
            verbose=True,
            allow_delegation=True,
            tools=[dataframe_creator],
            llm=llm,
            function_calling_llm=function_calling_llm,
        )

        ## DocuMentor Analyst
        DocuMentor = Agent(
            role="DocuMentor Analyst",
            backstory="An NLP expert skilled in extracting insights from internal business documents.",
            goal="Analyze internal documents to identify, contextual insights, and strategic objectives.",
            verbose=True,
            allow_delegation=True,
            tools=[retriever],
            llm=llm,
            function_calling_llm=function_calling_llm,
        )

        ## Insight Synthesizer
        InsightSynthesizer = Agent(
            role="Insight Synthesizer",
            backstory="A strategist blending AI-driven analytics with business insights.",
            goal="Merge quantitative trends and qualitative insights into actionable business recommendations, prioritizing key countries based on data analysis and aligning with internal objectives.",
            verbose=True,
            llm=llm,
            function_calling_llm=function_calling_llm,
        )

        ## Strategy Navigator
        StrategyNavigator = Agent(
            role="Strategy Navigator",
            backstory="A business strategist ensuring insights align with company goals and market trends.",
            goal="Validate insights, identify strengths and weaknesses, and align findings with business strategy, ensuring focus on key countries and business-critical indicators.",
            # tools=[ai_tool],
            verbose=True,
            llm=llm,
            function_calling_llm=function_calling_llm,
        )

        # ## Tech Integrator
        # TechIntegrator = Agent(
        #     role="Tech Integrator",
        #     backstory="A systems engineer ensuring seamless integration of AI and analytics into workflows.",
        #     goal="Automate workflows, connect IBM Planning Analytics with AI models, and monitor performance.",
        #     # tools=[api_tool],
        #     verbose=True,
        #     llm=llm,
        #     function_calling_llm=function_calling_llm,
        # )

        # Task Definitions

        data_task = Task(
            description="Extract simple business insights, identify key strengths and weaknesses, and determine the most critical intersections for prioritization.",
            agent=DataCore,
            expected_output="A structured and report with key trends highlighted, including priority intersections and identified strengths and weaknesses to focus on.",
        )

        doc_task = Task(
            description="Analyze internal documents to extract relevant business insights and strategic objectives .",
            agent=DocuMentor,
            expected_output="A summary of contextual insights, and strategic goals from internal documents.",
        )

        insight_task = Task(
            description=f"Synthesize quantitative and qualitative insights into actionable recommendations, focusing on strengths and weaknesses, targeting only indicators in {view_indicators_english}, and prioritizing affected countries among {view_countries_english} in alignment with strategic objectives.",
            agent=InsightSynthesizer,
            expected_output="A list of data-backed business recommendations focusing on priorities targeted by both the DataCore Analyst and the DocuMentor Analyst, targeting priority indicators and countries, and addressing identified strengths and weaknesses. The output should keep the exact syntax of the indicators from the indicators list",
        )

        strategy_task = Task(
            description="Validate insights, prioritize strategic actions, and align them with business goals, focusing on strengths and weaknesses to meet internal objectives.",
            agent=StrategyNavigator,
            expected_output="A prioritized action plan aligning insights with business goals, focusing on defined indicators, critical country intersections, and strategies to strengthen weaknesses and leverage strengths.",
        )

        # tech_task = Task(
        #     description="Ensure seamless technical integration between AI models and Planning Analytics data.",
        #     agent=TechIntegrator,
        #     expected_output="An automated workflow integrating Planning Analytics and AI-driven insights."
        # )

        # Crew Assembly
        crew = Crew(
            agents=[
                DataCore,
                DocuMentor,
                InsightSynthesizer,
                StrategyNavigator,
                # TechIntegrator,
            ],
            tasks=[
                data_task,
                doc_task,
                insight_task,
                strategy_task,
                #    tech_task,
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

    model_id = "meta-llama/llama-3-3-70b-instruct"
    extract_model_id = "mistralai/mistral-large"

    # Defining the model parameters

    parameters = {
        "decoding_method": "greedy",
        "min_new_tokens": 20,
        "max_new_tokens": 900,
        "repetition_penalty": 1.1,
        "stop_sequences": ["'''''''''", '"""', "```"],
    }

    extract_model_parameters = {
        "decoding_method": "greedy",
        "max_new_tokens": 6000,
        "min_new_tokens": 1,
        "stop_sequences": ["\n\n"],
        "repetition_penalty": 1,
    }

    project_id = config.get(
        "Keys", "project_ID"
    )  # replace with new watsonx.ai project_id

    ## Defining the Model object

    model = Model(
        model_id=model_id,
        params=parameters,
        credentials=get_credentials(),
        project_id=project_id,
    )

    extract_model = Model(
        model_id=extract_model_id,
        params=extract_model_parameters,
        credentials=get_credentials(),
        project_id=project_id,
    )

    def extract_indicators_from_text(text):
        # Regex pour capturer les noms d'indicateurs après les puces
        pattern = r">\s*(.*)"
        found_indicators = re.findall(pattern, text)
        return found_indicators

    def match_indicators(found_indicators, reference_indicators):
        # Comparer les indicateurs extraits avec ceux de la liste de référence
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

                                4. **Formatting**:  
                                - Increase: 'p%'  
                                - Decrease: '-p%'  
                                - For month ranges (e.g., Q1-Q2), list all relevant months
                                - Each line must start with this symbol >

                                5. **Country/Region Hierarchy**:  
                                - If a group of countries is mentioned (e.g., Western Europe), break it down into individual countries  
                                - Only include countries from the provided list  

                                ### Example:  
                                Text: "Reduce Maintenance Costs by 5% in Q4 in Scandinavia"  
                                Output:  
                                > {{'indicator':'Maintenance Costs','percent':'-5','country':'Finland','month':'October'}}  
                                > {{'indicator':'Maintenance Costs','percent':'-5','country':'Finland','month':'November'}}  
                                > {{'indicator':'Maintenance Costs','percent':'-5','country':'Finland','month':'December'}}  
                                > {{'indicator':'Maintenance Costs','percent':'-5','country':'Sweden','month':'October'}}  
                                > {{'indicator':'Maintenance Costs','percent':'-5','country':'Sweden','month':'November'}}  
                                > {{'indicator':'Maintenance Costs','percent':'-5','country':'Sweden','month':'December'}}  

                                Text: {crew_result}  
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

    indicateurs_trouves = [cell["indicator"] for cell in indicator_percent_country]
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

        update_subset("PaysExtraits", "Pays", "Pays", pays_trouves)
        update_subset("PaysExtraitsPourCarte", "Pays", "Pays", pays_trouves)
        # print('SUBSET PAYS TROUVES APRES UPDATE')
        # print(tm1.subsets.get_element_names('Pays','Pays','PaysExtraits'))
        # print('SUBSET PAYS CARTE TROUVES APRES UPDATE')
        # print(tm1.subsets.get_element_names('Pays','Pays','PaysExtraitsPourCarte'))
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
            # tm1.cells.write_value(
            #     matched_extracted_indicators,
            #     cube_name=output_cube_name,
            #     element_tuple=["AgenticAnalysis", "IdentifiedIndicators"],
            # )
            # tm1.cells.write_value(
            #     pays_trouves,
            #     cube_name=output_cube_name,
            #     element_tuple=["AgenticAnalysis", "IdentifiedCountries"],
            # )
            # tm1.cells.write_value(
            #     mois_trouves,
            #     cube_name=output_cube_name,
            #     element_tuple=["AgenticAnalysis", "IdentifiedMonths"],
            # )
            # tm1.cells.write_value(
            #     pourcentages_trouves,
            #     cube_name=output_cube_name,
            #     element_tuple=["AgenticAnalysis", "IdentifiedPercent"],
            # )

            # tm1.cells.write_value(
            #     matched_indicators_picklist,
            #     cube_name="}PickList_" + output_cube_name,
            #     element_tuple=["AgenticAnalysis", "CurrentIndicatorsPickList", "Value"],
            # )
            for target in indicator_percent_country:
                if "percent" in target.keys() and check_if_numeric(target["percent"]):
                    percent = target["percent"]
                    new_value = float(percent) / 100
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
                        tm1.cells.write_value(
                            new_value,
                            cube_name=cube_name,
                            element_tuple=[
                                "BUDG_VC_AJUST%",
                                year + "." + all_months[str(mois).lower()],
                                target_country,
                                target_indicator,
                            ],
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
                        tm1.cells.write_value(
                            new_value,
                            cube_name=cube_name,
                            element_tuple=[
                                "BUDG_VC_AJUST%",
                                period,
                                target_country,
                                target_indicator,
                            ],
                        )
                    # print(percent, period, target_country, target_indicator)


if __name__ == "__main__":
    main()
