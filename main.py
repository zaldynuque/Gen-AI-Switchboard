import streamlit as st

# General Data Manipulation and Analysis Libraries
import pandas as pd
import numpy as np
import scipy

import os

# Data Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import bokeh.plotting as bkp

import sqlite3
import uuid

from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.callbacks import StreamlitCallbackHandler

from langchain.agents.agent_types import AgentType
from langchain.agents import load_tools

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import (
    create_sql_agent, 
    SQLDatabaseToolkit, 
)

from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.tools.retriever import create_retriever_tool

from langchain_community.llms import HuggingFaceHub, HuggingFacePipeline

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

from langchain import hub
from langchain.schema.runnable import RunnablePassthrough

from utils.constants import (
    model_name,
    model_temp,
    model_top_p,
    page_title_top_df,
    page_title_df,
    welcome_header_df, 
    introduction_text_df, 
    features_header_df, 
    features_text_df,
    file_uploader_msg_df,
    query_box_text_df,
    logo_image,
)

from PIL import Image

from datetime import datetime, timedelta
from pprint import pprint
import time
# import torch
# import torch.nn as nn
# import torch.optim as optim
import plotly.io as pio
pio.templates.default = 'plotly' 

logo = Image.open(logo_image)

st.set_page_config(
    page_title=page_title_top_df,
    page_icon=logo,
    # layout="wide",
    # initial_sidebar_state="expanded",
)

openai_api_key = st.secrets["OPENAI_API_KEY"]
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set or is empty.")

# App Title
st.title(page_title_df)

# Introduction and Heading
st.header(welcome_header_df)
st.write(introduction_text_df)

# Feature List
st.subheader(features_header_df)
st.write(features_text_df)

uploaded_csv = st.file_uploader(
    file_uploader_msg_df, 
    type=['csv',],
    accept_multiple_files=False,
    key="pandasDfFileUploader"
)

def scroll_top():
    js = '''
    <script>
        // Select the correct container within Streamlit's layout
        var body = window.parent.document.querySelector(".main");

        // Log the element to console for debugging
        console.log(body);

        // Scroll to the bottom of the container
        body.scrollTop = body.scrollHeight;
    </script>
    '''

    # Create a temporary placeholder for displaying the HTML/JS
    temp = st.empty()

    # Use the placeholder to display the script
    with temp:
        st.components.v1.html(js, height=0)  # The height can be set to 0 as we don't need to display anything visibly
        time.sleep(0.5)  # Wait to allow the script to execute before the placeholder is emptied

    # Clear the placeholder after executing the script
    temp.empty() 

def truncate_text(text, max_length):
    """ Truncate text to a maximum length, appending '...' if the text was cut off. """
    if len(text) > max_length:
        return text[:max_length-3] + "..."
    return text

def extract_python_code(text):
    # Look for the start of the Python code block marked by ```python
    start_marker = "```python"
    end_marker = "```"
    
    # Find the starting position of the code block
    start = text.find(start_marker) + len(start_marker)
    
    # Find the end position of the code block using the end marker
    end = text.find(end_marker, start)
    
    # Extract and return the code segment, stripping any leading or trailing whitespace
    return text[start:end].strip()

def ensure_directory_exists(folder_path):
    # Check if the directory exists
    if not os.path.exists(folder_path):
        # Create the directory
        os.makedirs(folder_path)
        print(f"Directory created: {folder_path}")
    else:
        print(f"Directory already exists: {folder_path}")

def delete_old_images(folder_path):
    now = datetime.now()
    count_deleted = 0
    
    # Loop through all files in the directory
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if the file is an image
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            continue
        
        # Get the last modification time of the file
        file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
        
        # Calculate the age of the file
        if now - file_mtime > timedelta(days=1):
            os.remove(file_path)
            count_deleted += 1
            print(f"Deleted: {file_path}")
    
    print(f"Total files deleted: {count_deleted}")

if uploaded_csv:
    if "messages" not in st.session_state:
        scroll_top()

        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hi, How can I help you?"}
        ]

    for msg in st.session_state.messages:
        if msg["role"] == "steps":
            title = truncate_text(msg["content"]["title"], 50)
            expand = st.expander("🗸 " + title + "}")
            expand.markdown(msg["content"]["body"])    
        
        if msg["role"] != "steps":
            st.chat_message(msg["role"]).write(msg["content"])

    if query := st.chat_input(placeholder=query_box_text_df):
        scroll_top()

        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").write(query)

        llm = ChatOpenAI(
            temperature=model_temp, 
            model=model_name,
            openai_api_key=openai_api_key, 
            streaming=True,
            model_kwargs={
                "top_p":model_top_p,
            }
        )

        # Importing the structured data
        df = pd.read_csv(uploaded_csv);

        pandas_agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=False,
            return_intermediate_steps=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )
        
        query_extended = f"""To answer the question or perform the task, \
        don't use Matplotlib or TensorFlow, \
        you must use Plotly instead of Matplotlib \
        and you must use PyTorch instead of TensorFlow.
        import streamlit, torch and ploty (if necessary).\
        import all necessary package, function and method to answer the question or perform the task.
        if there are any figure, plot or chart, don't show that figure, plot or chart by `fig.show` or `plt.show`, \
        instead show plot or figure by `st.plotly_chart(fig, use_container_width=True)`.\
        don't return fig, instead display the fig using `st.plotly_chart(fig, use_container_width=True)`.\
        don't give code to run, it's your task to run the code to perform the task and answer the question based on the output.\
        Note: The df is `df`\

        Question or Task:
        {query}
        """

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            response = pandas_agent.invoke(
                {"input": query_extended}, {"callbacks": [st_cb]}
            )
            
            print(response) 

            # code = response["intermediate_steps"]
            # print(code)
            
            if 'intermediate_steps' in response:
                try:
                    for step in response["intermediate_steps"]:
                        try:
                            title = f"**{step[0].tool}:** " + "{" + f"'query': {step[0].tool_input['query']}"
                            body = f"""{step[0].message_log[0].content}

**Input:**

{step[0].tool_input['query']}

**Output:**

{step[0].message_log[1] if len(step[0].message_log)>1 else ""}
"""
                            st.session_state.messages.append({
                                "role": "steps", 
                                "content": {"title": title, "body": body.replace("#", "")}
                            })
                        except Exception as e:
                            print(e)

                        # code = response["intermediate_steps"][-1][0].tool_input["query"]
                        # if "plt.show" in code:
                        #     folder_path = "generated_img"  # Specify your folder path here
                        #     ensure_directory_exists(folder_path)
                        #     delete_old_images(folder_path)


                        #     # Generate date-time and unique key
                        #     now = datetime.now()
                        #     unique_key = uuid.uuid4()
                        #     date_time = now.strftime("%Y_%m_%d %H_%M_%S")
                        #     file_name = f"generated_img_{date_time}"
                        #     full_path = os.path.join(folder_path, file_name)
                        #     show_img = False
                        #     try:
                        #         exec(f"{code}\nplt.savefig('{full_path}.png')")
                        #         show_img = True
                        #     except Exception as e:
                        #         print(e)

                        #     if show_img:
                        #         st.image(f"{full_path}.png")

                        # if ("fig.show" in code or "\nfig" in code) and "st.plotly_chart(fig" not in code:
                        #     try:
                        #         exec(f"{code}\nst.plotly_chart(fig, use_container_width=True)")
                        #     except Exception as e:
                        #         print(e)
                except Exception as e:
                    print(e)

            if "output" in response: 
                # if "```python" in response["output"] and "st.plotly_chart(fig" in response["output"]:
                #     try:
                #         exec(extract_python_code(response["output"]))
                #     except Exception as e:
                #         print(e)
                st.write(response["output"])
                st.session_state.messages.append({"role": "assistant", "content": response["output"]})
            else:
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

