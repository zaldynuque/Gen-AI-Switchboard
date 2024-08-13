import time
import shutil
import uuid
import os
import pandas as pd
import sqlite3
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
from streamlit_js_eval import streamlit_js_eval

from .sqlite_operations import (
    get_column_data,
    get_column_info,
    delete_table,
    create_table_with_index,
    insert_data,
    insert_df_data
)

def scroll_top():
    js = '''
    <script>
        // Select the correct container within Streamlit's layout
        var body = window.parent.document.querySelector(".main");

        // Log the element to console for debugging
        console.log(body);

        // Scroll to the top of the container
        body.scrollTop = 0;
    </script>
    '''

    # Create a temporary placeholder for displaying the HTML/JS
    temp = st.empty()

    # Use the placeholder to display the script
    with temp:
        components.html(js, height=0)  # The height can be set to 0 as we don't need to display anything visibly
        time.sleep(0.5)  # Wait to allow the script to execute before the placeholder is emptied

    # Clear the placeholder after executing the script
    temp.empty()

def scroll_bottom():
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
        components.html(js, height=0)  # The height can be set to 0 as we don't need to display anything visibly
        time.sleep(0.5)  # Wait to allow the script to execute before the placeholder is emptied

    # Clear the placeholder after executing the script
    temp.empty() 


def sqlite_file_extensions():
    """
    Returns a list of common file extensions for SQLite databases.
    
    Returns:
    list: A list of common SQLite database file extensions.
    """
    return ['db', 'sqlite', 'sqlite3', 'db3', 's3db', 'sl3', 'sqlite2', 'db2']


def create_user_db_file(file_like, file_path):
    # Write the BytesIO object content to the new file
    with open(file_path, 'wb') as tmp_file:
        shutil.copyfileobj(file_like, tmp_file)

def get_table_col_info(cursor, table_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    return cursor.fetchall()

def ensure_directory_exists(directory):
    os.makedirs(directory, exist_ok=True)

# Function to find the first file that starts with the username
def find_user_db_file(username, directory):
    ensure_directory_exists(directory)

    for file in os.listdir(directory):
        if file.startswith(username):
            return os.path.join(directory, file)
    return None

def create_unique_filepath(username, user_file_name, directory):
    ensure_directory_exists(directory)

    # filename = f"{username}_{user_file_name}"
    filename = f"{username}_00.db"
    file_path = os.path.join(directory, filename)
    # st.session_state['file_path_db'] = file_path

    # unique_id = uuid.uuid4().hex[:4]
    # st.session_state['unique_id'] = unique_id
    return file_path

def get_first_table_name_db(cursor):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = cursor.fetchall()
    # return tables[0][0] if tables else None
    return tables if tables else None


def load_data_to_dataframe(conn, table_name, columns):
    select_query = f"SELECT {', '.join(columns)} FROM {table_name};"
    return pd.read_sql_query(select_query, conn)

def load_sqlite_to_dataframe(
    username, db_directory, user_file_name=None, file_path=None, file_like=None, table_name=None
):
    
    if not file_path:
        file_path = create_unique_filepath(username, user_file_name, db_directory)
        create_user_db_file(file_like, file_path)  # Ensure this function is correctly implemented

    try:
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()

        if not table_name:
            table_names = get_first_table_name_db(cursor)

            if table_names:
                table_names = [table_name[0] for table_name in table_names]
            else:
                table_names = []

            print(table_names)

            if len(table_names)>1:
                col_type = st.selectbox(f"Select a table to load", table_names, key="select_table")

                if st.session_state["select_table"]:
                    table_name = st.session_state["select_table"]
            else:
                table_name = table_names[0]
        if table_name:
            # Get column data
            column_data = get_column_data(table_name, '*', file_path)
            print("Column Data:", column_data)

            # Get column info
            user_db_columns = get_column_info(table_name, file_path)
            print("Column Info:", user_db_columns)

            # Delete table
            delete_table(table_name, file_path)

            # Create table with index if not present
            create_table_with_index(table_name, user_db_columns, file_path)

            # Insert data into table
            insert_data(table_name, user_db_columns, column_data, file_path)

            # new_table_name = f"{table_name}_new"
            # user_db_columns = get_table_col_info(cursor, table_name)
            
            # conn.execute("BEGIN;")  # Explicitly start a transaction
            
            # create_new_table(cursor, user_db_columns, new_table_name)

            columns = [col[1] for col in user_db_columns if col[1] != 'index']

            # copy_data_to_new_table(cursor, table_name, new_table_name, columns)

            # finalize_table_schema(cursor, table_name, new_table_name)
            
            # conn.commit()
            
            df = load_data_to_dataframe(conn, table_name, columns)
        else:
            print("No tables found in the database.")
            return pd.DataFrame(), None, None, None

    except Exception as e:
        conn.rollback()  # Roll back the transaction on failure
        print(e)
        return pd.DataFrame(), None, None, None
        # raise e

    finally:
        cursor.close()
        conn.close()

    return df, user_db_columns, table_name, file_path

def reload_app():
    streamlit_js_eval(js_expressions="parent.window.location.reload()")

def adjust_df_change(df, change_dict):
    # Edit rows
    edited_rows = []
    for row_id, row_data in change_dict["edited_rows"].items():
        row_id = int(row_id)  # Ensure row_id is of the correct type
        if row_id in df.index:
            for col, new_val in row_data.items():
                df.at[row_id, col] = new_val  # Using .at for better performance on single value assignment
            edited_rows.append(row_id)
        else:
            print(f"Row {row_id} not found. Cannot edit.")

    # Add rows
    added_rows = change_dict["added_rows"]
    new_rows = pd.DataFrame(added_rows)
    df = pd.concat([df, new_rows], ignore_index=True)

    # Delete rows
    deleted_rows = change_dict["deleted_rows"]
    df.drop(deleted_rows, inplace=True)

    return df, edited_rows, added_rows, deleted_rows

def update_and_save_df(change_dict, df, file_path, table_name, user_db_columns=None):
    try:
        
        df, edited_rows, added_rows, deleted_rows = adjust_df_change(df, change_dict)
        # st.session_state['user_df_from_db'] = df

        try:
            # Connect to SQLite database
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()

            column_data = get_column_data(table_name, '*', file_path)
            print("Column Data:", column_data)

            # Get column info
            user_db_columns = get_column_info(table_name, file_path)
            print("Column Info:", user_db_columns)

            # Delete table
            delete_table(table_name, file_path)

            # Create table with index if not present
            create_table_with_index(table_name, user_db_columns, file_path)

            insert_df_data(table_name, df, file_path)

            # Insert data into table
            # insert_data(table_name, user_db_columns, column_data, file_path)

            # new_table_name = f"{table_name}_new"

            # df.reset_index(inplace=True)  # Make the index into a column
            # df.rename(columns={'index': primary_key_col}, inplace=True)

            # # Save DataFrame to SQLite without index as a temporary table
            # df.to_sql('new_table_name', conn, if_exists='replace', index=True)

            # if not user_db_columns:
            #     # Retrieve column information from the temporary table
            #     user_db_columns = get_table_col_info(cursor, new_table_name)

            # conn.execute("BEGIN;")  # Explicitly start a transaction

            # create_new_table(cursor, user_db_columns, new_table_name)

            # columns = [col[1] for col in user_db_columns if col[1] != 'index']
            # copy_data_to_new_table(cursor, table_name, new_table_name, columns)

            # finalize_table_schema(cursor, table_name, new_table_name)

            # # Commit changes
            # conn.commit()

        except Exception as e:
            conn.rollback()  # Roll back the transaction on failure
            print(e)
            # raise e

        finally:
            cursor.close()
            conn.close()

        # Print summary of changes and file saved
        print(f"Changes applied successfully:")
        if edited_rows:
            st.toast(f"\n*Edited Rows*: { ', '.join(map(str, edited_rows)) }\n", icon='✍️')
            time.sleep(.5)
        if added_rows:
            st.toast(f"\n*Newly Added*: { len(added_rows) } row(s)\n", icon='🆕')
            time.sleep(.5)
        if deleted_rows:
            st.toast(f"\n*Deleted Rows*: { ', '.join(map(str, deleted_rows)) }\n", icon='🗑️')

        primary_key_manual_col = '"index"'

        print(f"\n# Database Updated\n\
        📂 Updated DataFrame saved to { file_path } with { primary_key_manual_col } as the primary key.")

        reload_app()

    except Exception as e:
        print(f"Error: {e}")
        raise e


def generate_update_confirmation_md(change_summary):
    """
    Generate a Markdown-formatted confirmation message detailing database changes with emojis and enhanced formatting.

    Parameters:
    change_summary (dict): A dictionary containing details of edited, added, and deleted rows.

    Returns:
    str: A Markdown-formatted string that summarizes the changes with emojis and appropriate formatting.
    """
    message = "#### 📊 Database Update Confirmation\n######"

    # Check if any changes occurred and add sections accordingly
    has_changes = any(key in change_summary for key in ['edited_rows', 'added_rows', 'deleted_rows'])
    if not has_changes:
        message += "\nNo changes were made to the database.\n"
    else:
        # Edited rows
        if change_summary.get('edited_rows'):
            message += "\n###### ✍️ Edited Rows\n"
            for row_id, changes in change_summary['edited_rows'].items():
                message += f"- Row ID {row_id}:\n"
                for key, value in changes.items():
                    message += f"  - **{key}**: `{value}`\n"

        # Added rows
        if change_summary.get('added_rows'):
            message += "\n###### 🆕 Added Rows\n"
            for row in change_summary['added_rows']:
                message += f"- New Row Details:\n"
                for key, value in row.items():
                    message += f"  - **{key}**: `{value}`\n"

        # Deleted rows
        if change_summary.get('deleted_rows'):
            message += "\n###### 🗑️ Deleted Rows\n"
            for row_id in change_summary['deleted_rows']:
                message += f"- Row ID {row_id}\n"

    return message

# def get_current_changes(prev_change_dict, curr_change_dict):
#     """
#     Compares the previous and current change dictionaries and returns a new change dictionary
#     with only the current changes.
    
#     Args:
#         prev_change_dict (dict): The previous change dictionary.
#         curr_change_dict (dict): The current change dictionary.
        
#     Returns:
#         dict: A dictionary containing the current changes, with the following keys:
#             'edited_rows': A dictionary with row IDs as keys and dictionaries of edited columns as values.
#             'added_rows': A list of dictionaries representing the added rows.
#             'deleted_rows': A list of row IDs representing the deleted rows.
#     """
#     current_change_dict = {
#         "edited_rows": {},
#         "added_rows": [],
#         "deleted_rows": []
#     }

#     st.session_state['user_df_changes_prev'] = prev_change_dict

#     # Find edited rows
#     for row_id, row_data in curr_change_dict["edited_rows"].items():
#         if row_id in prev_change_dict["edited_rows"]:
#             prev_row_data = prev_change_dict["edited_rows"][row_id]
#             edited_cols = {col: row_data[col] for col in row_data.keys() if row_data[col] != prev_row_data.get(col)}
#             if edited_cols:
#                 current_change_dict["edited_rows"][row_id] = edited_cols
#         else:
#             current_change_dict["edited_rows"][row_id] = row_data

#     # Find added rows
#     prev_added_rows = set(tuple(row.items()) for row in prev_change_dict["added_rows"])
#     curr_added_rows = set(tuple(row.items()) for row in curr_change_dict["added_rows"])
#     added_rows = curr_added_rows - prev_added_rows
#     current_change_dict["added_rows"] = [dict(row) for row in added_rows]

#     # Find deleted rows
#     prev_deleted_rows = set(prev_change_dict["deleted_rows"])
#     curr_deleted_rows = set(curr_change_dict["deleted_rows"])
#     deleted_rows = curr_deleted_rows - prev_deleted_rows
#     current_change_dict["deleted_rows"] = list(deleted_rows)

#     return current_change_dict