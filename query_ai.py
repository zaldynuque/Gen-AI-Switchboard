import streamlit as st

import pandas as pd
import numpy as np
import os

import sqlite3

from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.callbacks import StreamlitCallbackHandler

from langchain.agents.agent_types import AgentType
from langchain.agents import load_tools

# from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.utilities import SQLDatabase

from langchain_community.agent_toolkits import (
    create_sql_agent, 
    SQLDatabaseToolkit, 
)

# from langchain_community.agent_toolkits.sql.prompt import SQL_FUNCTIONS_SUFFIX
# from langchain_core.messages import AIMessage, SystemMessage
# from langchain_core.prompts.chat import (
#     ChatPromptTemplate,
#     HumanMessagePromptTemplate,
#     MessagesPlaceholder,
# )

# from langchain.agents import create_openai_tools_agent
# from langchain.agents.agent import AgentExecutor

from utils.constants import (
    model_name,
    model_temp,
    model_top_p,
    page_title_top_db,
    page_title_db,
    welcome_header_db, 
    introduction_text_db, 
    features_header_db, 
    features_text_db,
    file_uploader_msg_db,
    query_box_text_db,
    logo_image,
    init_msg_db,
)

from utils.utils import (
    sqlite_file_extensions,
    load_sqlite_to_dataframe,
    update_and_save_df,
    generate_update_confirmation_md,
    scroll_bottom,
    find_user_db_file,
    reload_app,
    adjust_df_change,
    get_first_table_name_db
)

from utils.menu import (
    show_menu,
)

from utils.chat_ui import (
    chat_ui,
)

from PIL import Image

logo = Image.open(logo_image)

st.set_page_config(
    page_title=page_title_top_db,
    page_icon=logo,
    # layout="wide",
    # initial_sidebar_state="expanded",
)

show_menu()

# App Title
st.title(page_title_db)

# Introduction and Heading
st.header(welcome_header_db)
st.write(introduction_text_db)

# Feature List
st.subheader(features_header_db)
st.write(features_text_db)

supported_file_extensions = sqlite_file_extensions()

uploaded_db = st.file_uploader(
    file_uploader_msg_db, 
    type=supported_file_extensions,
    accept_multiple_files=False,
    key="pandasDfFileUploader"
)

# Function to generate SQL create table statement
def create_table_sql(db_name, table_name, columns):
    sql = f'CREATE DATABASE IF NOT EXISTS "{db_name}"";\n'
    sql += f"USE {db_name};\n"
    sql += f"CREATE TABLE {table_name} (\n"
    col_defs = []
    for col in columns:
        col_defs.append(f"    {col['name']} {col['type']} {'DEFAULT ' + str(col['default']) if col['default'] else ''} {'NOT NULL' if col['required'] else ''}")
    sql += ",\n".join(col_defs)
    sql += "\n);"
    return sql

def generate_column_config(col_info, sqlite_datatypes):
    column_config = {}
    print(col_info)
    for col in col_info:
        name = col['name']
        col_type = col['type']
        default_value = col['default']
        required = col['required']

        if col_type.upper() in sqlite_datatypes:
            if col_type.upper() == "INTEGER" or col_type.upper() == "REAL" or col_type.upper() == "NUMERIC":
                column_config[name] = st.column_config.NumberColumn(
                    name,
                    default=default_value if default_value else 0,
                    required=required
                )
            elif col_type.upper() == "TEXT":
                column_config[name] = st.column_config.TextColumn(
                    name,
                    default=default_value if default_value else "",
                    required=required
                )

    return column_config

def generate_create_table_query(table_name, col_info):
    query1 = f"DROP TABLE IF EXISTS {table_name};"
    query2= f"CREATE TABLE {table_name} (\n"

    for idx, col in enumerate(col_info):
        col_name = col['name']
        col_type = col['type']
        default_value = col.get('default', '')
        required = "NOT NULL" if col['required'] else ""

        if default_value:
            query2 += f"    {col_name} {col_type} DEFAULT '{default_value}' {required}"
        else:
            query2 += f"    {col_name} {col_type} {required}"

        if idx < len(col_info) - 1:
            query2 += ",\n "

    query2 += ");"

    return query1, query2

def generate_insert_query(table_name, df):
    columns = df.columns.tolist()
    values = []

    for index, row in df.iterrows():
        row_values = [f"'{value}'" if isinstance(value, str) else str(value) for value in row.tolist()]
        values.append(f"({', '.join(row_values)})")

    insert_query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES\n"
    insert_query += ",\n".join(values) + ";"

    if len(values) < 1:
        return None

    return insert_query


def create_manual_db(file_path_db, drop_existing_query, create_db_query, insert_value_query):
    # Check if the database file exists, if not, create it
    if not os.path.isfile(file_path_db):
        open(file_path_db, 'w').close()
        print(f"Database file created at {file_path_db}")

    # Connect to the database
    conn = sqlite3.connect(file_path_db)
    cursor = conn.cursor()

    try:
        conn.execute("BEGIN;")  # Explicitly start a transaction

        try:
            table_names = get_first_table_name_db(cursor)

            if table_names:
                table_names = [table_name[0] for table_name in table_names]
            else:
                table_names = []

            for table_name in table_names:
                cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
                print(f"Existing table `{table_name}` dropped")

            # cursor.execute(drop_existing_query)
            # print(f"Existing table dropped")
        except Exception as e:
            print(f"Error dropping table: {e}")
            raise e

        try:
            cursor.execute(create_db_query)
            print(f"New table created")
        except Exception as e:
            print(f"Error creating table: {e}")
            raise e

        try:
            if insert_value_query:
                cursor.execute(insert_value_query)
                print(f"Values inserted into the table")
        except Exception as e:
            print(f"Error inserting values: {e}")
            raise e
        
        conn.commit()

        reload_app()

    except Exception as e:
        conn.rollback()  # Roll back the transaction on failure
        print(f"Error creating manuual database: {e}")
        raise e

    finally:
        cursor.close()
        conn.close()

sqllite_datatypes = ["INTEGER", "TEXT", "REAL", "NUMERIC"]
# sqllite_datatypes = ["INTEGER", "TEXT", "BLOB", "REAL", "NUMERIC"]


# file_path_db = st.session_state['file_path_db']
# table_name_db = st.session_state['table_name_db']

user_db_columns = None
user_df_from_db2 = pd.DataFrame()
user_db_columns = None
table_name_db = None
username = "guest_user_001"
db_directory = 'user_db_files'
file_path_db = os.path.join(db_directory, f"{username}_00.db")

if uploaded_db:
    user_file_name = uploaded_db.name

    user_df_from_db2, user_db_columns, table_name_db, file_path_db = load_sqlite_to_dataframe(
        username, 
        db_directory,
        user_file_name,
        None,
        uploaded_db, 
    )

else:
    # Attempt to find the database file
    file_path = find_user_db_file(username, db_directory)
    if not file_path:
        print("No file starting with the username was found.")
    else:
        user_df_from_db2, user_db_columns, table_name_db, file_path_db = load_sqlite_to_dataframe(
            username, 
            db_directory,
            None,
            file_path, 
        )

@st.experimental_dialog("SQL DB Generation")
def db_create(file_path_db):

    if 'table_name_mannual' not in st.session_state:
        st.session_state['table_name_mannual'] = "My Table Name"

    if 'col_names_mannual_str' not in st.session_state:
        st.session_state['col_names_mannual_str'] = "FirstName, LastName, Age"

    if 'df_mannual' not in st.session_state:
        st.session_state['df_mannual'] = pd.DataFrame(columns=st.session_state['col_names_mannual_str'].split(", "))

    st.markdown("\n ### 🚀 Add Database Using CSV File:")

    csv_file = st.file_uploader("Upload CSV file to set columns and data", type=['csv'])
    
    # with cols_change_btn[0]:
    btn_change = st.button("Change the input based on your CSV")
    
    if btn_change:
        if csv_file:
            st.session_state['df_mannual'] = pd.read_csv(csv_file)
            st.session_state['table_name_mannual'] = csv_file.name.replace(".csv", "")
            st.session_state['col_names_mannual_str'] = ", ".join(list(st.session_state['df_mannual'].columns))

    if st.session_state['col_names_mannual_str']:
        # Split the 'col_names_mannual_str' string into a list of desired column names
        desired_cols = st.session_state['col_names_mannual_str'].split(', ')

        # Fill in empty strings for missing columns
        for col in desired_cols:
            if col not in st.session_state['df_mannual'].columns:
                st.session_state['df_mannual'][col] = ''
            else:
                st.session_state['df_mannual'][col] = st.session_state['df_mannual'][col] 

        st.session_state['df_mannual'] = st.session_state['df_mannual'][desired_cols]

    st.write("\n")
    st.markdown("\n ### 📁 Settings for Table Creation:")

    table_name = st.text_input("Table Name", key="table_name_mannual")

    col_names_mannual_str_input = st.text_input(
        "Column Names (comma-separated)",
        "FirstName, LastName, Age",
        key="col_names_mannual_str"
    )

    # Manage columns dynamically
    columns = []

    for idx, name in enumerate(st.session_state['col_names_mannual_str'].split(', ')):
        if name:
            cols = st.columns(2)
            with cols[0]:
                col_type = st.selectbox(f"Type for {name.strip()}", sqllite_datatypes, key=f"type_{idx}")
            with cols[1]:

                if f"required_{idx}" not in st.session_state:
                    st.session_state[f"required_{idx}"] = False

                col_default = st.text_input(
                    f"Default value for {name.strip()}", 
                    key=f"default_{idx}",
                    disabled=st.session_state[f"required_{idx}"],
                    placeholder="default value",
                )
            
            cols_2 = st.columns(2)
            with cols_2[0]:
                col_required = st.checkbox(
                    f"Required for {name.strip()}", 
                    key=f"required_{idx}",
                    disabled=st.session_state[f"default_{idx}"].strip() != "",
                )
            
            columns.append({'name': name.strip(), 'type': col_type, 'default': col_default, 'required': col_required})

    column_config = generate_column_config(
        columns, 
        sqllite_datatypes
    )

    st.write("\n")
    st.markdown("\n ### 🆕 Table for Data Addition:")

    st.data_editor(
        st.session_state['df_mannual'], 
        key="user_df_mannual_changes", 
        num_rows="dynamic",
        hide_index=True,
        column_config=column_config
    )

    table_name_mannual = st.session_state['table_name_mannual'].strip().replace(" ", "_")

    # print(st.session_state['col_names_mannual_str'])
    drop_existing_query, create_db_query = generate_create_table_query(
        table_name_mannual, 
        columns
    )

    df_mannual_adjusted, edited_rows, added_rows, deleted_rows = adjust_df_change(
        st.session_state['df_mannual'], 
        st.session_state['user_df_mannual_changes']
    )

    insert_value_query = generate_insert_query(
        table_name_mannual,
        df_mannual_adjusted
    )

    st.write("\n")
    st.markdown("\n ### 🧠 Generated Queries:")
    
    st.write(drop_existing_query)
    st.write(create_db_query)
    st.write(insert_value_query)

    # with cols_change_btn[0]:
    btn_create_db = st.button("Create Database")
    
    if btn_create_db:
        create_manual_db(
            file_path_db,
            drop_existing_query,
            create_db_query,
            insert_value_query,
        )

if st.button("Or Create a Database Manualy"):
    db_create(file_path_db)

if not user_df_from_db2.empty:
    st.data_editor(
        user_df_from_db2, 
        key="user_df_changes", 
        num_rows="dynamic",
        hide_index=True,
        # column_config={
        # "index": st.column_config.Column(
        #         "index",
        #         # help="Streamlit **widget** commands 🎈",
        #         # width="medium",
        #         # required=True,
        #         disabled=True,
        #     )
        # },
    ) # <iframe allow="clipboard-write;clipboard-read;" ... src="https://your-app-url"></iframe>

    # Further check if any of the specific keys are non-empty
    if any(st.session_state["user_df_changes"].get(key) for key in ['edited_rows', 'added_rows', 'deleted_rows']):

        user_df_changes = st.session_state['user_df_changes']

        st.write("Click the reflect button below to update the database or cancel the changes.")

        cols_changes_btn = st.columns(2)
        with cols_changes_btn[0]:
            st.button("Reflect the Change on Your Database", on_click=update_and_save_df, args=(
                user_df_changes, 
                user_df_from_db2, 
                file_path_db,
                table_name_db,
                user_db_columns, 
            ))
        with cols_changes_btn[1]:
            st.button("Revert Back (Cencel the Changes)", on_click=reload_app)

    else:
        st.write("You can add, edit or delete row in the above database, if you want.")

    st.markdown("---")

    chat_ui(file_path_db)
