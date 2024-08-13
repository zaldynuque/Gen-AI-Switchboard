import os
import re
import tempfile
import tiktoken
import logging
import streamlit as st

from langchain import hub
from datetime import datetime
from utils.menu import show_menu
from utils.utils import scroll_bottom
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain.tools.retriever import create_retriever_tool
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

openai_api_key = st.secrets["OPENAI_API_KEY"]
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set or is empty.")

os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

# Constants
MAX_FILE_SIZE_MB = 50
CHAT_HISTORY_TOKEN_LIMIT = 5000
MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-ada-002"

# Setup logging
if 'ENV_STAGE_NAME' in st.secrets and st.secrets['ENV_STAGE_NAME'] == 'dev':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning(f"Model {model} not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    
    if model == "gpt-3.5-turbo-0613":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(str(value)))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens -= 1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        logger.warning(f"num_tokens_from_messages() is not presently implemented for model {model}. Using fallback method.")
        return sum(len(encoding.encode(str(message.get("content", "")))) for message in messages)

def update_chat_history(new_message: str, role: str, token_limit: int, model="gpt-3.5-turbo-0613"):

    while True:
        total_tokens = num_tokens_from_messages(st.session_state.chat_history, model)
        if total_tokens <= token_limit:
            break
        if len(st.session_state.chat_history) < 2:
            logger.warning("Chat history is too short to truncate further.")
            break
        # Remove the oldest user message and AI response pair
        st.session_state.chat_history.pop(0)
        st.session_state.chat_history.pop(0)
    
    logger.info(f"Chat history updated. Current token count: {total_tokens}")

def escape_dollar_signs(text: str) -> str:
    return re.sub(r'\$(?=\d)', r'\\$', text)

def load_file(uploaded_file):
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    loader = None
    tmp_file_path = ""

    try:
        if file_extension == ".pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                tmp_file_path = temp_file.name
                loader = PyPDFLoader(tmp_file_path)
        elif file_extension == ".txt":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                tmp_file_path = temp_file.name
                loader = TextLoader(tmp_file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        if loader and tmp_file_path:
            documents = loader.load()
            return documents
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
            except Exception as e:
                st.error(f"Error removing temporary file: {e}")

    return []

def process_uploaded_files(uploaded_files):
    total_size = sum(file.size for file in uploaded_files)
    if total_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"Total file size exceeds {MAX_FILE_SIZE_MB} MB. Please upload smaller files.")
        return None

    documents = []
    for uploaded_file in uploaded_files:
        try:
            documents.extend(load_file(uploaded_file))
            logger.info(f"Successfully processed file: {uploaded_file.name}")
        except Exception as e:
            logger.error(f"Error processing {uploaded_file.name}: {str(e)}")
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    if documents:
        st.success("Files successfully processed!")
        return documents
    else:
        st.warning("No valid documents were processed.")
        return None

def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=openai_api_key
    )
    vector_store = FAISS.from_documents(texts, embeddings)
    
    logger.info(f"Vector store created with {len(texts)} chunks")
    st.success("Document store created successfully!")
    return vector_store

def chat_interface(vector_store):
    st.subheader("Chat with your documents")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask a question about your documents"):
        
        scroll_bottom()

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            # Set up tools and agent
            llm = ChatOpenAI(
                model_name=MODEL_NAME, 
                temperature=0,
                openai_api_key=openai_api_key,
            )
            search = TavilySearchResults(max_results=5, search_depth="advanced")
            retriever = vector_store.as_retriever()

            # Get today's date
            today_date = datetime.now().strftime("%Y-%m-%d")
            
            retriever_tool = create_retriever_tool(
                retriever,
                "search_documents",
                f"Use this tool to search and retrieve information from the user documents. It provides relevant context for answering user queries based on the content of the user files. Provide a clear and concise answer only based on the retrieved content. Ensure the answer is well-formatted and easy to understand. If the needed information is not found in the retrieved content, or if you need more up-to-date information, use the tavily_answer tool to search the internet. When performing a web search, use the date {today_date} where necessary. Strive to keep your answers concise yet comprehensive. Avoid unnecessary details but ensure all relevant information is included."
            )

            tools = [search, retriever_tool]
            prompt = hub.pull("hwchase17/openai-functions-agent")
            agent = create_tool_calling_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

            response = agent_executor.invoke(
                {
                    "input": query,
                    "chat_history": [
                        HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) 
                        for m in st.session_state.chat_history
                    ]
                },
                {"callbacks": [st_cb]}
            )

            if "output" in response: 
                output_res = response["output"]
            else:
                output_res = response

            output_res = escape_dollar_signs(output_res)
            message_placeholder.markdown(output_res)

        logger.debug(f"User query: {query}")
        logger.debug(f"AI response: {output_res if 'output' in response else response}")

        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append({"role": "assistant", "content": output_res})

        # Update the chat history with both the user message and AI response
        update_chat_history(query, "user", CHAT_HISTORY_TOKEN_LIMIT)
        update_chat_history(output_res, "assistant", CHAT_HISTORY_TOKEN_LIMIT)

# Main application logic
def main():
    st.set_page_config(page_title="RAG Chat App", page_icon="📚")

    show_menu()
    
    st.title("📚 RAG Chat Application")
    st.write("Upload documents and chat with AI about their content! Get answers from your documents and web search.")
    
    st.subheader("How to use")
    st.markdown("""
    1. Upload PDF, TXT files (max 50MB total)
    2. Wait for the files to be processed
    3. Start asking questions about your documents
    4. The AI will respond based on the content of your documents and web search if needed
    """)

    # Initialize session state variables
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "uploaded_files_hash" not in st.session_state:
        st.session_state.uploaded_files_hash = None
        

    uploaded_files = st.file_uploader(
        "Upload PDF, TXT files (max 50MB total)",
        accept_multiple_files=True,
        type=["pdf", "txt"]
    )


    # Calculate a hash of the uploaded files
    current_files_hash = hash(tuple(f.name + str(f.size) for f in uploaded_files)) if uploaded_files else None

    # Check if new files are uploaded or if the uploaded files have changed
    if uploaded_files and current_files_hash != st.session_state.uploaded_files_hash:
        with st.spinner("Processing uploaded files..."):
            documents = process_uploaded_files(uploaded_files)
            if documents:
                st.session_state.vector_store = create_vector_store(documents)
                st.session_state.uploaded_files_hash = current_files_hash
                st.success("Documents processed and ready for chat!")
    elif not uploaded_files:
        st.info("Please upload your documents to start chatting!")

    if st.session_state.vector_store:
        chat_interface(st.session_state.vector_store)

    # st.write(st.session_state)

if __name__ == "__main__":
    main()
