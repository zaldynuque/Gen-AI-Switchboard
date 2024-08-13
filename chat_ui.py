import streamlit as st

from utils.utils import (
    scroll_bottom,
)

from utils.constants import (
    model_name,
    model_temp,
    model_top_p,
    query_box_text_db,
    init_msg_db,
)

from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.callbacks import StreamlitCallbackHandler

from langchain.agents.agent_types import AgentType
from langchain.agents import load_tools

from langchain_community.utilities import SQLDatabase

from langchain_community.agent_toolkits import (
    create_sql_agent, 
    SQLDatabaseToolkit, 
)

import re

def escape_dollar_signs(text):
    """
    Escapes dollar signs in a string where the dollar sign precedes numbers.
    This is useful for ensuring that dollar amounts are not mistakenly treated as LaTeX or markdown commands.

    Args:
        text (str): The input string containing dollar amounts.

    Returns:
        str: The modified string with escaped dollar signs.
    """
    # Regular expression to find dollar signs followed by numbers
    pattern = r'\$(?=\d)'
    
    # Replace the found patterns with an escaped dollar sign
    escaped_text = re.sub(pattern, r'\\$', text)
    return escaped_text

openai_api_key = st.secrets["OPENAI_API_KEY"]
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set or is empty.")

def get_db_agent(file_path_db):
    llm = ChatOpenAI(
            temperature=model_temp, 
            model=model_name,
            openai_api_key=openai_api_key, 
            # streaming=True,
            # model_kwargs={
            #     "top_p":model_top_p,
            # }
        )

    # In place of file, you can enter the path to your database file
    db = SQLDatabase.from_uri(f"sqlite:///{file_path_db}")
    db.get_usable_table_names()

    tools = ['llm-math']

    tools = load_tools(tools, llm)

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    db_agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        tools=tools,
        return_intermediate_steps=True,
        # verbose=True,
        # agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
    )

    return db_agent

# @st.experimental_fragment
def chat_ui(file_path_db):
    if "messages" not in st.session_state:
        scroll_bottom()

        st.session_state["messages"] = [
            {"role": "assistant", "content": init_msg_db}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if query := st.chat_input(placeholder=query_box_text_db):
        scroll_bottom()

        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").write(query)

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

            db_agent = get_db_agent(file_path_db)

            query_extended = f"""Don't give code to run, it's your task to run the code to \
            perform the task and answer the question based on the output.\
            if you can't find something during the code run, try importing it.

            Question or Task:
            {query}
            """


            response = db_agent.invoke(
                {"input": query_extended}, {"callbacks": [st_cb]}
            )


            if "output" in response: 
                output_res = escape_dollar_signs(response["output"])
                st.write(output_res)

                # print(response["output"])
                st.session_state.messages.append({"role": "assistant", "content": output_res})
            else:
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

