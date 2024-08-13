# sqlite_operations.py

import sqlite3
import pandas as pd

# Function to get all data from a column
def get_column_data(table_name, column_name, db_file):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute(f"SELECT {column_name} FROM {table_name}")
        print(f"SELECT {column_name} FROM {table_name}")
        data = cursor.fetchall()
        conn.close()
        return data
    except sqlite3.Error as e:
        print("Error fetching data:", e)
        return None

# Function to get column information of a table
def get_column_info(table_name, db_file):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        print(f"PRAGMA table_info({table_name})")
        cursor.execute(f"PRAGMA table_info({table_name})")
        column_info = cursor.fetchall()
        conn.close()
        return column_info
    except sqlite3.Error as e:
        print("Error fetching column info:", e)
        return None

# Function to delete a table
def delete_table(table_name, db_file):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        print(f"DROP TABLE IF EXISTS {table_name}")
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print("Error deleting table:", e)

# Function to create a table with all column info and add an index column if not present
def create_table_with_index(table_name, column_info, db_file):
    try:
        column_string = ''

        # Loop through column information
        for col in column_info:
            col_name = col[1]
            if col_name != "index":
                col_type = f" {col[2]}"

                is_not_null = " NOT NULL" if col[3] == 1 else ""
                default_val = f" DEFAULT {col[4]}" if col[4] is not None else ""
                
                # Add column definition to string
                column_string += f"{col_name}{col_type}{is_not_null}{default_val}, "

        column_string += '"index" INTEGER PRIMARY KEY AUTOINCREMENT, '

        # Remove trailing comma and space
        column_string = column_string.rstrip(", ")

        # Create table
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        print(f"CREATE TABLE IF NOT EXISTS {table_name} ({column_string})")
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({column_string})")
        conn.close()
    except sqlite3.Error as e:
        print("Error creating table:", e)


# Function to insert data into a table
def insert_data(table_name, column_info, data, db_file):
    try:
        # columns = [col[1] for col in column_info]
        columns = ', '.join(['"' + col[1] + '"' if col[1] == 'index' else col[1] for col in column_info])
        placeholders = ', '.join(['?' for _ in range(len(columns.split(', ')))])
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Get column names from DataFrame

        print(f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})", data)
        cursor.executemany(f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})", data)
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print("Error inserting data:", e)


# Function to insert DataFrame data into a table with DataFrame index as the database index
def insert_df_data(table_name, df, db_file):
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Drop the index from DataFrame to insert it into the database as the index
        df_to_insert = df.reset_index()

        # Convert DataFrame to list of tuples
        data = df_to_insert.to_records(index=False).tolist()
        print(data)

        # Get column names from DataFrame
        columns = ', '.join(['"' + col + '"' if col == 'index' else col for col in df_to_insert.columns])


        # Create placeholders for the INSERT statement including the index
        placeholders = ', '.join(['?' for _ in range(len(df.columns) + 1)])  # Add one placeholder for the index

        # Execute INSERT statement
        print(f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})", data)
        cursor.executemany(f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})", data)
        conn.commit()
        conn.close()
        print("Data inserted successfully.")
    except sqlite3.Error as e:
        print("Error inserting data:", e)
