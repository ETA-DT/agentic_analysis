__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import time
import pandas as pd
import mdpd
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool, AgentType, AgentExecutor, create_react_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_core.runnables import chain
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai import Credentials

# from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from crewai.tools import tool
# from crewai_tools import CSVSearchTool
from pydantic import BaseModel
import docling

from docling.backend.msword_backend import MsWordDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import (ConversionResult, InputDocument, SectionHeaderItem)
from docling.document_converter import DocumentConverter

from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

import json
import os
import re
import configparser

from pathlib import Path

from TM1py.Services import TM1Service
from TM1py.Utils.Utils import build_pandas_dataframe_from_cellset
from TM1py.Utils.Utils import build_cellset_from_pandas_dataframe
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ibm import WatsonxEmbeddings
from langchain_ibm import WatsonxLLM
from langchain_ibm import ChatWatsonx
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from streamlit_file_browser import st_file_browser

# from crewai import LLM
from dotenv import load_dotenv

load_dotenv()

WATSONX_APIKEY = os.getenv("WATSONX_APIKEY", "")
WATSONX_PROJECT_ID = os.getenv("PROJECT_ID", "")

os.environ["WATSONX_URL"] = "https://us-south.ml.cloud.ibm.com/"
os.environ["WATSONX_APIKEY"] = WATSONX_APIKEY
os.environ["WATSONX_PROJECT_ID"] = WATSONX_PROJECT_ID

WATSONX_URL = os.getenv("WATSONX_URL", "")

credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_APIKEY)


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
    # # Loading files
    loader = DoclingPDFLoader(file_path=path)
    file_name = loader._converter.convert(path).document.name
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    )
    docs = loader.load()
    docs[0].metadata = {"filename": file_name}
    splits = text_splitter.split_documents(docs)
    return splits


# embedding
embed_params = {
    EmbedParams.TRUNCATE_INPUT_TOKENS: 512,
    EmbedParams.RETURN_OPTIONS: {"input_text": True},
}

embeddings = WatsonxEmbeddings(
    model_id="intfloat/multilingual-e5-large", #EmbeddingTypes.IBM_SLATE_125M_ENG.value,
    url=get_credentials()["url"],
    apikey=get_credentials()["apikey"],
    project_id=WATSONX_PROJECT_ID,
    params=embed_params,
)

def create_vectorstore(path):
    texts = create_documents(path)
    return Chroma.from_documents(texts, embeddings)

docsearch = None
doc_folder = "Documents RAG"

def update_doc_folder(folder): 
    folder_list_dir = os.listdir(folder)
    folder_list_dir.remove(".gitignore")
    return folder_list_dir

files_name = update_doc_folder(doc_folder)

if files_name:
    file_path = os.path.join(f'{doc_folder}/{files_name[0]}')
    docsearch = create_vectorstore(file_path)

def add_documents(vectorbase, path):
    new_documents = create_documents(path)
    vectorbase.add_documents(new_documents)

# import streamlit as st
# import requests
# import base64

# # GitHub configuration
# GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Store securely, e.g., in environment variables
# GITHUB_USERNAME = "ETA-DT"
# REPO_NAME = "agentic_analysis"
# BRANCH_NAME = "main"  # Or the branch you want to push to

# # GitHub API URL template
# GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_USERNAME}/{REPO_NAME}/contents"

# # Streamlit file uploader
# uploaded_file = st.file_uploader("Upload a file to GitHub", type=["txt", "csv", "png", "jpg", "pdf","docx"])

# if uploaded_file is not None:
#     file_content = uploaded_file.getvalue()
#     file_name = uploaded_file.name

#     # Encode the file content to base64 (required by GitHub API)
#     encoded_content = base64.b64encode(file_content).decode("utf-8")

#     # API payload
#     payload = {
#         "message": f"Add {file_name}",
#         "content": encoded_content,
#         "branch": BRANCH_NAME
#     }

#     # Make the API request
#     upload_url = f"{GITHUB_API_URL}/{file_name}"  # File will be saved at the root of the repository
#     headers = {"Authorization": f"token {GITHUB_TOKEN}"}

#     response = requests.put(upload_url, headers=headers, json=payload)

#     # Handle the API response
#     if response.status_code == 201:
#         st.success(f"File '{file_name}' successfully uploaded to GitHub!")
#     elif response.status_code == 422:  # Unprocessable Entity, file may already exist
#         st.error(f"File '{file_name}' already exists in the repository.")
#     else:
#         st.error(f"An error occurred: {response.json()}")

document_dataframe = []

st.set_page_config(layout="wide")
with st.sidebar:
    uploaded_files = st.file_uploader('Choose a Doc File',type="docx", accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file not in files_name:
                with open(os.path.join("Documents RAG",uploaded_file.name),"wb") as f: 
                    f.write(uploaded_file.getbuffer())         
                files_name = update_doc_folder(doc_folder)
            if not(docsearch):
                docsearch = create_vectorstore(os.path.join(f'{doc_folder}/{uploaded_file.name}'))
        document_dataframe = list(set([source['filename'] for source in [elem for elem in docsearch.get()['metadatas'] if elem]]))
        st.success("Saved File")
    
        st.sidebar.header('Directory')
        # event = st_file_browser(os.path.join("Documents RAG"),
        # key="deep",
        # use_static_file_server=False,
        # show_choose_file=False,
        # show_delete_file=True,
        # show_download_file=False,
        # show_new_folder=False,
        # show_upload_file=False,
        # )
    
    if document_dataframe:
        for filename in files_name:
            if filename not in document_dataframe:
                add_documents(
                    docsearch,
                    os.path.join(os.path.join(f'{doc_folder}/{filename}')),
                    )
            document_dataframe = list(set([source['filename'] for source in [elem for elem in docsearch.get()['metadatas'] if elem]]))
            # st.write(filename)

    st.dataframe(pd.DataFrame({"Documents":document_dataframe}), hide_index = True)

    complete_loading = st.button("Done")



# to keep track of tasks performed by agents
task_values = []

# # define current directory
# def set_current_directory():
#     abspath = os.path.abspath()  # file absolute path
#     directory = os.path.dirname(abspath)  # current file parent directory
#     os.chdir(directory)
#     return directory


# CURRENT_DIRECTORY = set_current_directory()
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
    "max_new_tokens": 2000,
    "temperature": 0.1,
    "top_k": 50,
    "top_p": 1,
    "repetition_penalty": 1,
}

parameters_llama = {
    "decoding_method": "sample",
    "max_new_tokens": 4000,
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


# display the console processing on streamlit UI
class StreamToExpander:
    def __init__(self, expander):
        self.expander = expander
        self.buffer = []
        self.colors = ["red", "green", "blue", "orange"]  # Define a list of colors
        self.color_index = 0  # Initialize color index

    def write(self, data):
        # Filter out ANSI escape codes using a regular expression
        cleaned_data = re.sub(r"\x1B\[[0-9;]*[mK]", "", data)

        # Check if the data contains 'task' information
        task_match_object = re.search(
            r"\"task\"\s*:\s*\"(.*?)\"", cleaned_data, re.IGNORECASE
        )
        task_match_input = re.search(
            r"task\s*:\s*([^\n]*)", cleaned_data, re.IGNORECASE
        )
        task_value = None
        if task_match_object:
            task_value = task_match_object.group(1)
        elif task_match_input:
            task_value = task_match_input.group(1).strip()

        if task_value:
            st.toast(":robot_face: " + task_value)

        # Check if the text contains the specified phrase and apply color
        if "Entering new CrewAgentExecutor chain" in cleaned_data:
            # Apply different color and switch color index
            self.color_index = (self.color_index + 1) % len(
                self.colors
            )  # Increment color index and wrap around if necessary

            cleaned_data = cleaned_data.replace(
                "Entering new CrewAgentExecutor chain",
                f":{self.colors[self.color_index]}[Entering new CrewAgentExecutor chain]",
            )

        if "Senior Data Analyst" in cleaned_data:
            # Apply different color
            cleaned_data = cleaned_data.replace(
                "Senior Data Analyst",
                f":{self.colors[self.color_index]}[Senior Data Analyst]",
            )
        if "Senior Business Advisor" in cleaned_data:
            cleaned_data = cleaned_data.replace(
                "Senior Business Advisor",
                f":{self.colors[self.color_index]}[Senior Business Advisor]",
            )
        if "Internal Document Researcher" in cleaned_data:
            cleaned_data = cleaned_data.replace(
                "Internal Document Researcher",
                f":{self.colors[self.color_index]}[Internal Document Researcher]",
            )
        if "Finished chain." in cleaned_data:
            cleaned_data = cleaned_data.replace(
                "Finished chain.", f":{self.colors[self.color_index]}[Finished chain.]"
            )

        self.buffer.append(cleaned_data)
        if "\n" in data:
            self.expander.markdown("".join(self.buffer), unsafe_allow_html=True)
            self.buffer = []


# Streamlit interface
def run_crewai_app():
    st.title("Watsonx AI Agent for dataframe analysis")
    cube_name = ""
    view_name = ""

    # définir et se connecter à l'instance tm1
    with TM1Service(**config["tango_core_model"]) as tm1:

        cube_name = st.text_input("Enter the cube name to analyze.")
        if cube_name == "":
            st.stop()
        else:
            view_name = st.selectbox(
                "Select the view name to analyze.",
                tm1.cubes.views.get_all_names(cube_name=cube_name)[-1],
                index=None,
                placeholder="Click to select...",
            )
            if not view_name:
                st.stop()
        
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
    
    with st.expander("See Data"):
        st.dataframe(current_dataframe, hide_index=True)
    
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
        def retriever(query: str) -> List[Document]:
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
            retrieved_context = ""
            for (
                chunk
            ) in unique_retrieval:  # concatenate documents into a single str paragraph
                retrieved_context += chunk + "\n"
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
            agent = create_pandas_dataframe_agent(pandas_llm,
                                    df,
                                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                                    prefix= "Always return a JSON dictionary that can be parsed into a data frame containing the requested information.",
                                    verbose=True, 
                                    allow_dangerous_code=True,
                                    include_df_in_prompt=True,
                                    number_of_head_rows=len(df)
                                    )
            response = agent.invoke(
                query, handle_parsing_errors=True, return_intermediate_steps=False
            )
            return response['output']

        # Define Senior Data Analyst Agent
        senior_data_analyst = Agent(
            role="Senior Data Analyst",
            goal=f"Analyze the dataframe and provide clear, actionable key points.",
            backstory="""You are a senior data analyst with a strong background 
                            in applied mathematics and computer science, skilled in deriving insights 
                            from complex datasets.""",
            verbose=True,
            allow_delegation=True,
            tools=[dataframe_creator],
            llm=llm,
            function_calling_llm=function_calling_llm,
        )

        # Define Senior Business Advisor Agent
        senior_business_advisor = Agent(
            role="Senior Business Advisor",
            goal="Summarize key points from data analysis and present actionable insights for decision-making with relevant figures and by clearly stating the scope of the action and the reason why the company should focus on this scope.",
            backstory="""You are an experienced business consultant with a strong foundation in data-driven 
                            decision-making, skilled at distilling complex analyses into clear, impactful recommendations.""",
            verbose=True,
            allow_delegation=True,
            llm=llm,
            function_calling_llm=function_calling_llm,
        )

        # Define Macroeconomics Researcher Agent
        macroeconomics_researcher = Agent(
            role="Macroeconomics Researcher",
            goal="Research macroeconomic insights to inform and optimize company budget planning for the upcoming year.",
            backstory="""You are a skilled economist with expertise in macroeconomic trends, forecasting, 
                            and their implications for corporate financial strategies.""",
            verbose=True,
            allow_delegation=True,
            llm=llm,
            tool=[duckduckgo_search],
            function_calling_llm=function_calling_llm,
        )

        # Define Internal Document Researcher Agent
        internal_document_researcher = Agent(
            role="Internal Document Researcher",
            goal="Extract insights and goals with targeted indicators and countries from internal documents to support strategic planning and decision-making",
            backstory="""You are a meticulous analyst with expertise in reviewing internal documents, 
                            identifying key insights with relevant figures and aligning them with organizational objectives.""",
            verbose=True,
            allow_delegation=True,
            tools=[retriever],
            llm=llm,
            function_calling_llm=function_calling_llm,
        )

        # Define Task 1: Data Analysis
        task1 = Task(
            description=f"Analyze the provided non indexed dataframe. {dataframe_enriched_prompt_input(cube_name=cube_name, view_name=view_name)}. Identify trends, anomalies, and key statistics. Provide actionable insights and key points for decision-makers.",
            expected_output="Analysis of the dataframe with key points, trends, and actionable insights to prioritize, and emphasize on pain points",
            agent=senior_data_analyst,
        )

        # Define Task 2: Business Insights
        task2 = Task(
            description="Based on the dataframe analysis and the internal documents insights, summarize briefly the analysis and provide clear, actionable business insights and recommendations for decision-making with figures and the targeted scope such as target indicator and country. The recommendations must be explicits with figures and precised indicators and justified according to the data and the goals stated in internal documents and why is this scope is primarily targeted.",
            expected_output="Business insights and actions recommendations based on the data analysis and the internal documents.",
            agent=senior_business_advisor,
        )

        # Define Task 3: Macroeconomic Insights
        task3 = Task(
            description="Research relevant macroeconomic trends and insights to inform company budget planning for the upcoming year. Provide a report on economic forecasts and their potential impact on business operations.",
            expected_output="Macroeconomic insights and recommendations to optimize the budget planning process.",
            agent=macroeconomics_researcher,
        )

        # Define Task 4: Internal Document Insights
        task4 = Task(
            description="Extract key insights and goals with the targeted indicators and countries from internal documents that align with company objectives and strategic planning for the upcoming year. Focus your research on the scope that seems primary according to the Data Analysis",
            expected_output="Insights and goals from internal documents relevant to strategic planning.",
            agent=internal_document_researcher,
        )

        task_values.append([task1.description,task4.description,task2.description])
        
        # Create and Run the Crew
        product_crew = Crew(
            agents=[
                senior_data_analyst,
                internal_document_researcher,
                senior_business_advisor,
            ],
            tasks=[task1, task4, task2],
            verbose=True,
            process=Process.sequential,
        )
        crew_result = product_crew.kickoff()
        return crew_result

    with st.expander("About the Team:"):
        # left_co, cent_co, last_co = st.columns(3)
        # with cent_co:
        #     # st.image("my_img.png")
        #     pass

        st.subheader("Senior Data Analyst")
        st.text(
            f"""
        Role = Senior Data Analyst
        Goal = Analyze the dataframe and provide clear, actionable key points.
        Backstory = You are a senior data analyst with a strong background 
                    in applied mathematics and computer science, skilled in deriving insights 
                    from complex datasets.
        Task = Analyze the provided dataframe. Identify trends, anomalies, and key statistics.
               Provide actionable insights and key points for decision-makers."""
        )

        st.subheader("Senior Business Advisor")
        st.text(
            """
        Role = Senior Business Advisor
        Goal = Summarize key points from data analysis and present actionable insights for decision-making with relevant figures and by clearly stating the scope of the action.
        Backstory = You are an experienced business consultant with a strong foundation in data-driven 
                    decision-making, skilled at distilling complex analyses into clear, impactful recommendations.
        Task = Based on the dataframe analysis and the internal documents insights, summarize briefly the analysis and provide clear,
               actionable business insights and recommendations for decision-making with figures and the targeted scope. The recommendations
               must be explicits with figures and precised indicators and scope and justified regarding the data and the internal documents."""
        )

        st.subheader("Internal Document Researcher")
        st.text(
            """
        Role = Internal Document Researcher
        Goal= Extract insights and goals with targeted indicators and countries from internal documents to support strategic planning and decision-making.
        Backstory = You are a meticulous analyst with expertise in reviewing internal documents, 
                    identifying key insights with relevant figures and aligning them with organizational objectives.
        Task = Extract key insights and goals from internal documents that align with company objectives and
               strategic planning for the upcoming year."""
        )

    if st.button("Run Analysis"):
        # Placeholder for stopwatch
        stopwatch_placeholder = st.empty()

        # Start the stopwatch
        start_time = time.time()
        with st.expander(
            "Processing!",
        ):
            sys.stdout = StreamToExpander(st)
            with st.spinner("Generating Results"):
                crew_result = create_crewai_setup(cube_name, view_name)

        # Stop the stopwatch
        end_time = time.time()
        total_time = end_time - start_time
        stopwatch_placeholder.text(f"Total Time Elapsed: {total_time:.2f} seconds")
        
        st.header("Tasks:")
        st.table({"Tasks": task_values})

        st.header("Results:")
        st.markdown(crew_result)


if __name__ == "__main__":
    run_crewai_app()
