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

# cube_name = "00.Ventes"
# view_name_results = "00. Resultats"
# view_name_sales = "00. Ventes totales"
# view_name_costs = "01. Analyse Couts"
# view_name = view_name_results

# WATSONX_LLAMA3_MODEL_ID = "watsonx/mistralai/mistral-large"
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
    "temperature": 0.7,
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
    parameters=parameters_llama,
)


# Streamlit interface
def main():
    output_cube_name = "TM1py_output"
    # définir et se connecter à l'instance tm1
    with TM1Service(**config["tango_core_model"]) as tm1:

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
            period_dim, period_dim, "French"
        )
        countries = tm1.elements.get_elements_by_level(
            country_dim, country_dim, level=0
        )
        all_indicators = tm1.elements.get_elements_by_level(
            measure_dim, measure_dim, level=0
        )  # liste de tous les indicateurs de la dimension indicateurs

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

    def preprocessing(df):
        """
        Preprocessing the dataframe
        """
        df = df.fillna(0)
        rename_period_alias(df)
        for col in df.columns[-12:]:
            df[col] = df[col].apply(round_2)
        # df = df.set_index(df.columns[0])
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
            )
        )

    current_dataframe = view_dataframe(cube_name, view_name)

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
            goal="Transform raw data into clean, structured information ready for AI insights, identifying the most critical intersections for targeted recommendations.",
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
            goal="Analyze internal documents to identify trends, contextual insights, and strategic objectives.",
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
            description="Clean and analyze Planning Analytics data to extract trends, identify key strengths and weaknesses, and determine the most critical intersections for prioritization.",
            agent=DataCore,
            expected_output="A structured and cleaned DataFrame with key trends highlighted, including priority intersections and identified strengths and weaknesses.",
        )

        doc_task = Task(
            description="Analyze internal documents to extract relevant business insights and strategic objectives.",
            agent=DocuMentor,
            expected_output="A summary of key themes, contextual insights, and strategic goals from internal documents.",
        )

        insight_task = Task(
            description=f"Synthesize quantitative and qualitative insights into actionable recommendations, focusing on strengths and weaknesses, targeting only indicators in {all_indicators}, and prioritizing affected countries in alignment with strategic objectives.",
            agent=InsightSynthesizer,
            expected_output="A list of data-backed business recommendations integrating both datasets, targeting relevant indicators and key countries, and addressing identified strengths and weaknesses. The output should keep the exact syntax of the indicators from the indicators list",
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

    # crew = create_crewai_setup(cube_name, view_name)
    # crew_result = crew.kickoff()

    crew_result = f"""**Prioritized Action Plan: Aligning Insights with Business Goals**

                    Based on the analysis and recommendations provided, the following prioritized action plan is proposed to align insights with business goals, focusing on defined indicators, critical country intersections, and strategies to strengthen weaknesses and leverage strengths.

                    **Short-term priorities (Q1-Q2)**

                    1. Launch marketing campaigns in Q1 and Q2 to stimulate "Recettes commerciales" outside of peak periods in Europe de l’Ouest (Belgique, Allemagne, Pays-Bas).
                    2. Develop seasonal offers to increase "Recettes passager" during peak summer periods in Europe du Sud (Portugal, Espagne).
                    3. Reduce "Coûts de maintenance" by 5% by anticipating and budgeting for Q4 maintenance costs in Scandinavie (Finlande, Suède).
                    4. Centralize purchases of spare parts to reduce "Coûts des ventes" by 4% in Europe de l’Ouest (Belgique, Allemagne, Pays-Bas).

                    **Mid-term priorities (Q3-Q4)**

                    1. Invest in AI to optimize routes and reduce residual costs ("Coûts généraux") by 2% in Angleterre (Royaume-Uni, Irlande).
                    2. Develop premium subscriptions for businesses to increase "Recettes commerciales- engagement contractuel" by 5% in Angleterre (Royaume-Uni, Irlande).
                    3. Automate stock management to prevent seasonal overstocking and reduce "Coûts Stock Biens" by 5% in Europe de l’Est (République tchèque, Slovaquie).
                    4. Optimize routes to reduce "Frais de voyages (SO)" by 5% and increase efficiency in Europe du Sud (Portugal, Espagne).

                    **Long-term priorities (Year 2 and beyond)**

                    1. Develop local partnerships to attract tourists and increase "Revenus des activités de tourisme (occasionnel)" by 10% in Europe de l’Est (République tchèque, Slovaquie).
                    2. Invest in "Programmes Fidélité" to increase loyalty and "Recettes commerciales- engagement contractuel" by 10% in Europe du Sud (Portugal, Espagne).
                    3. Stabilize "Ventes Carburant" through promotional offers (e.g., winter subscriptions) to prevent a decrease in sales at the end of the year in Scandinavie (Finlande, Suède).
                    4. Enhance "Recettes commerciales" by maintaining operational excellence through quarterly audits in Angleterre (Royaume-Uni, Irlande).

                    **Key Performance Indicators (KPIs)**

                    1. Revenue growth
                    2. Cost reduction
                    3. Customer loyalty and retention
                    4. Operational efficiency

                    **Country-specific KPIs**

                    1. Scandinavie (Finlande, Suède): Stabilize "Ventes Carburant" and reduce "Coûts de maintenance" by 5%.
                    2. Angleterre (Royaume-Uni, Irlande): Enhance "Recettes commerciales" by 5% and reduce residual costs ("Coûts généraux") by 2%.
                    3. Europe de l’Est (République tchèque, Slovaquie): Increase "Revenus des activités de tourisme (occasionnel)" by 10% and reduce "Coûts Stock Biens" by 5%.
                    4. Europe de l’Ouest (Belgique, Allemagne, Pays-Bas): Stimulate "Recettes commerciales" outside of peak periods by 5% and reduce "Coûts des ventes" by 4%.
                    5. Europe du Sud (Portugal, Espagne): Increase "Recettes passager" during peak summer periods by 10% and reduce "Frais de voyages (SO)" by 5%.

                    By following this prioritized action plan, the company can achieve its objectives and improve overall performance across the different regions."""

    # tm1.cells.write_value(
    #     crew_result,
    #     cube_name="TM1py_output",
    #     element_tuple=["AgenticAnalysis", "Results"],
    # )

    model_id = "meta-llama/llama-3-1-70b-instruct"
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

    all_indicators = [elem for elem in all_indicators if elem[0] != "%"]

    extract_prompt_input_percent = f"""Voici une liste stricte et exhaustive d'indicateurs :
                                {all_indicators}

                                Voici une liste de pays :
                                {countries}
                                        
                                Extrait uniquement les indicateurs, le pourcentage d'augmentation/diminution, les pays et les mois associés qui sont recommandés de modifier dans ce texte. 

                                ### Règles strictes d'extraction :
                                1. **Conversion des périodes en mois** :
                                - Q1 → Janvier, Février, Mars
                                - Q2 → Avril, Mai, Juin  
                                - Q3 → Juillet, Août, Septembre
                                - Q4 → Octobre, Novembre, Décembre
                                - "Début d'année" → Janvier, Février, Mars
                                - "Fin d'année" → Octobre, Novembre, Décembre
                                - "Été" → Juin, Juillet, Août
                                - "Hiver" → Décembre, Janvier, Février

                                2. **N’extrais un indicateur que s’il est exactement dans la liste fournie**

                                3. **Conserve l’intitulé exact** des indicateurs

                                4. **Formatage** :
                                - Augmentation : 'p%'
                                - Réduction : '-p%'
                                - Pour les plages de mois (ex: Q1-Q2), liste tous les mois concernés

                                5. **Hiérarchie pays/région** :
                                - Si un groupe de pays est mentionné (ex: Europe de l'Ouest), décompose en pays individuels
                                - Uniquement les pays de la liste fournie

                                ### Exemple :
                                Texte: "Réduire Coûts de maintenance de 5% en Q4 en Scandinavie"
                                Sortie: 
                                > {{'indicator':'Coûts de maintenance','percent':'-5','country':'Finlande','mois':'Octobre'}}
                                > {{'indicator':'Coûts de maintenance','percent':'-5','country':'Finlande','mois':'Novembre'}}
                                > {{'indicator':'Coûts de maintenance','percent':'-5','country':'Finlande','mois':'Décembre'}}
                                > {{'indicator':'Coûts de maintenance','percent':'-5','country':'Suède','mois':'Octobre'}}
                                > {{'indicator':'Coûts de maintenance','percent':'-5','country':'Suède','mois':'Novembre'}}
                                > {{'indicator':'Coûts de maintenance','percent':'-5','country':'Suède','mois':'Décembre'}}
                                
                                Texte: {crew_result}
                                Indicateurs trouvés: """

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

    remove_incomplete_extraction(found_percent)

    indicator_percent_country = [ast.literal_eval(val) for val in found_percent]
    # for association in found_percent:
    #     if len(association.split(";;")) == 2:
    #         indicator, percent, pays = association.split(";;")
    #         if tm1.elements.exists(
    #             dimension_name="Pays", hierarchy_name="Pays", element_name=pays
    #         ):
    #             indicator_percent_country.append(
    #                 {"indicator": indicator, "percent": percent, "country": pays}
    #             )
    #     if len(association.split(";;")) == 3:
    #         indicator, percent, pays = association.split(";;")
    #         if tm1.elements.exists(
    #             dimension_name="Pays", hierarchy_name="Pays", element_name=pays
    #         ):
    #             indicator_percent_country.append(
    #                 {"indicator": indicator, "percent": percent, "country": pays}
    #             )
    #     if len(association.split(";;")) == 4:
    #         indicator, percent, pays, mois = association.split(";;")
    #         if tm1.elements.exists(
    #             dimension_name="Pays", hierarchy_name="Pays", element_name=pays
    #         ):
    #             indicator_percent_country.append(
    #                 {
    #                     "indicator": indicator,
    #                     "percent": percent,
    #                     "country": pays,
    #                     "mois": mois,
    #                 }
    #             )

    print("\nindicator_percent_country\n")
    print(indicator_percent_country)

    indicateurs_trouves = [cell["indicator"] for cell in indicator_percent_country]
    print("\nindicateurs_trouves\n")
    print(indicateurs_trouves)
    matched_extracted_indicators = match_indicators(indicateurs_trouves, all_indicators)
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

        print("SUBSET Indicateur AVANT UPDATE")
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

        # print('SUBSET PAYS TROUVES AVANT UPDATE')
        # print(tm1.subsets.get_element_names('Pays','Pays','PaysExtraits'))
        # print('SUBSET PAYS CARTE TROUVES AVANT UPDATE')
        # print(tm1.subsets.get_element_names('Pays','Pays','PaysExtraitsPourCarte'))

        pays_trouves = list(
            set([cell["country"] for cell in indicator_percent_country])
        )
        mois_trouves = list(
            set(
                [
                    cell["mois"]
                    for cell in indicator_percent_country
                    if "mois" in cell.keys()
                ]
            )
        )
        pourcentages_trouves = list(
            set([cell["percent"] for cell in indicator_percent_country])
        )
        print("\npays_trouves\n")
        print(pays_trouves)

        print("\nmois_trouves\n")
        print(mois_trouves)

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

        tm1.cells.write_value(
            crew_result,
            cube_name=output_cube_name,
            element_tuple=["AgenticAnalysis", "Results"],
        )
        year = "2025"
        all_months = {
            "janvier": "01",
            "février": "02",
            "mars": "03",
            "avril": "04",
            "mai": "05",
            "juin": "06",
            "juillet": "07",
            "août": "08",
            "septembre": "09",
            "octobre": "10",
            "novembre": "11",
            "décembre": "12",
        }

        if 1 == 1:
            # tm1.cells.write_value(
            #     matched_indicators,
            #     cube_name=output_cube_name,
            #     element_tuple=["AgenticAnalysis", "CurrentIndicators"],
            # )

            tm1.cells.write_value(
                matched_extracted_indicators,
                cube_name=output_cube_name,
                element_tuple=["AgenticAnalysis", "IdentifiedIndicators"],
            )
            tm1.cells.write_value(
                pays_trouves,
                cube_name=output_cube_name,
                element_tuple=["AgenticAnalysis", "IdentifiedCountries"],
            )
            tm1.cells.write_value(
                mois_trouves,
                cube_name=output_cube_name,
                element_tuple=["AgenticAnalysis", "IdentifiedMonths"],
            )
            tm1.cells.write_value(
                pourcentages_trouves,
                cube_name=output_cube_name,
                element_tuple=["AgenticAnalysis", "IdentifiedPercent"],
            )

            # tm1.cells.write_value(
            #     matched_indicators_picklist,
            #     cube_name="}PickList_" + output_cube_name,
            #     element_tuple=["AgenticAnalysis", "CurrentIndicatorsPickList", "Value"],
            # )
            for target in indicator_percent_country:
                if "percent" in target.keys():
                    percent = target["percent"]
                    new_value = int(percent)
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
                if "mois" in target.keys():
                    mois = target["mois"]
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
                    print(percent, mois, target_country, target_indicator)
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
                    print(percent, period, target_country, target_indicator)


if __name__ == "__main__":
    main()
