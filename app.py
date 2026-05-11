__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import wikipedia
import tempfile
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

# Set up page config
st.set_page_config(page_title="Aegis-RAG Copilot", page_icon="🤖", layout="centered")
st.title("🤖 Aegis-RAG: Enterprise AI Agent")
st.write("Ask questions about uploaded documents, corporate policies, live stock prices, or general facts!")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize active vector retriever
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# ----------------- ADD THIS SESSION RESET BUTTON ---------------
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

# API Key input field
api_key = st.sidebar.text_input(
    "Enter your Gemini API Key:",
    value=os.getenv("GEMINI_API_KEY", ""),
    type="password"
)

# ----------------- FILE UPLOADER WIDGET -----------------
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader(
    "Upload corporate documents (PDF)", 
    type=["pdf"], 
    accept_multiple_files=False
)

# Set up local embedding caching
local_project_cache = os.path.join(os.getcwd(), "local_model_cache")
embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    cache_dir=local_project_cache
)

# ----------------- DYNAMIC DOCUMENT PROCESSING -----------------
@st.cache_resource
def get_default_vector_store():
    """Generates the fallback ACME Corp database."""
    docs = [
        Document(
            page_content="ACME Corp remote work policy: Employees can work remotely up to 3 days per week with manager approval.",
            metadata={"source": "HR Policy Manual"}
        ),
        Document(
            page_content="ACME Corp benefits: Full-time employees receive health insurance, dental insurance, and a $2000 annual learning stipend.",
            metadata={"source": "HR Benefits Guide"}
        )
    ]
    return Chroma.from_documents(docs, embeddings)

# Process the uploaded file if there is one
if uploaded_file is not None:
    st.sidebar.success(f"📄 {uploaded_file.name} uploaded successfully!")
    
    # Process only if we haven't already processed this specific file
    if "processed_file_name" not in st.session_state or st.session_state.processed_file_name != uploaded_file.name:
        with st.sidebar.status("🧠 Embedding document...", expanded=True) as embed_status:
            # 1. Save uploaded bytes to a temp file so PyPDFLoader can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name

            # 2. Extract and split text
            loader = PyPDFLoader(temp_path)
            raw_docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = text_splitter.split_documents(raw_docs)
            
            # 3. Store in transient database and assign to active retriever session
            db = Chroma.from_documents(split_docs, embeddings)
            st.session_state.retriever = db
            st.session_state.processed_file_name = uploaded_file.name
            embed_status.update(label="✅ Document embedded!", state="complete")
else:
    # If file is removed, fall back to default
    st.session_state.retriever = get_default_vector_store()
    if "processed_file_name" in st.session_state:
        del st.session_state.processed_file_name


# ----------------- TOOL DEFINITIONS -----------------
wikipedia.set_user_agent("MyPortfolioBot/1.0 (contact: test@example.com)")

if api_key:
    # Use the active session-state retriever (either uploaded file or default ACME)
    retriever = st.session_state.retriever

    @tool
    def query_company_policy(query: str) -> str:
        """Searches the internal corporate document database. Use this for questions about company rules, employee guides, benefits, or any uploaded document content."""
        docs = retriever.similarity_search(query, k=3)
        results = "\n".join([f"[{d.metadata.get('source', 'Document')}]: {d.page_content}" for d in docs])
        return f"Results from Document Database:\n{results}"

    @tool
    def search_wikipedia(query: str) -> str:
        """Searches Wikipedia for historical facts, general knowledge, or famous people/companies. Do not use for corporate internal data."""
        wikipedia_api = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
        return WikipediaQueryRun(api_wrapper=wikipedia_api).run(query)

    @tool
    def fetch_stock_price(ticker: str) -> str:
        """Fetches the current mock stock price for a given ticker symbol (e.g., AAPL, GOOG, TSLA)."""
        prices = {"AAPL": "$182.50", "GOOG": "$175.20", "TSLA": "$190.10", "MSFT": "$420.30"}
        ticker_upper = ticker.upper()
        if ticker_upper in prices:
            return f"The current stock price of {ticker_upper} is {prices[ticker_upper]}."
        return f"Stock ticker {ticker_upper} not found in the database."

    tools = [query_company_policy, search_wikipedia, fetch_stock_price]

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key)
    llm_with_tools = llm.bind_tools(tools)


# ----------------- CHAT INTERFACE -----------------
# Display previous chat messages
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage) and message.content:
        clean_text = message.content
        if isinstance(clean_text, list):
            clean_text = "".join([b.get("text", "") for b in clean_text if b.get("type") == "text"])
        with st.chat_message("assistant"):
            st.write(clean_text)

# Accept user input
if prompt := st.chat_input("Ask me anything..."):
    if not api_key:
        st.error("Please provide a Gemini API Key first!")
    else:
        # Render User Message
        with st.chat_message("user"):
            st.write(prompt)
        
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        
        # Build System Message Context
        messages = [
            SystemMessage(content="You are a smart assistant. Try to answer questions directly first. Only use tools if absolutely necessary and NEVER make more than one tool call per turn."),
        ] + st.session_state.chat_history
        
        with st.chat_message("assistant"):
            with st.status("🔍 Analyzing your request...", expanded=True) as status:
                response = llm_with_tools.invoke(messages)
                messages.append(response)
                
                while response.tool_calls:
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        
                        st.write(f"⚙️ Running tool **{tool_name}**...")
                        
                        selected_tool = next(t for t in tools if t.name == tool_name)
                        tool_output = selected_tool.invoke(tool_args)
                        
                        messages.append(
                            ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                        )
                    
                    response = llm_with_tools.invoke(messages)
                    messages.append(response)
                
                status.update(label="✅ Analysis complete!", state="complete", expanded=False)
                
            final_output = response.content
            if isinstance(final_output, list):
                final_output = "".join([b.get("text", "") for b in final_output if b.get("type") == "text"])
            
            st.write(final_output)
            st.session_state.chat_history.append(AIMessage(content=final_output))