__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import wikipedia
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document

# Set up page config
st.set_page_config(page_title="Aegis-RAG Copilot", page_icon="🤖", layout="centered")
st.title("🤖 Aegis-RAG: Enterprise AI Agent")
st.write("Ask questions about ACME Corp policies, live stock prices, or general facts!")

# ----------------- ADD THIS SESSION RESET BUTTON ---------------
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

# Force the input box to show up in the sidebar. 
# It will pre-fill with your system key if you have one, or stay blank so you can paste a new one!
api_key = st.sidebar.text_input(
    "Enter your Gemini API Key:",
    value=os.getenv("GEMINI_API_KEY", ""),
    type="password"
)

@st.cache_resource
def get_vector_store():
    # Load documents
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
    
    # Force FastEmbed to download directly into your local project directory (bypass Windows Admin blocks!)
    import os
    local_project_cache = os.path.join(os.getcwd(), "local_model_cache")
    
    embeddings = FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        cache_dir=local_project_cache
    )
    
    return Chroma.from_documents(docs, embeddings)

# ----------------- TOOL DEFINITIONS -----------------
wikipedia.set_user_agent("MyPortfolioBot/1.0 (contact: test@example.com)")

if api_key:
    # Get our robust, zero-dependency python retriever directly!
    retriever = get_vector_store()

    @tool
    def query_company_policy(query: str) -> str:
        """Searches the ACME Corp internal employee benefits database. Use this for questions about company rules, office locations, benefits, or budget."""
        docs = retriever.similarity_search(query,k=3)
        results = "\n".join([d.page_content for d in docs])
        return f"Results from ACME Policy Database:\n{results}"

    @tool
    def search_wikipedia(query: str) -> str:
        """Searches Wikipedia for historical facts, general knowledge, or famous people/companies. Do not use for ACME Corp internal data."""
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
# Keep track of conversation history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous chat messages
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage) and message.content:
        # Extract clean text from Gemini blocks if necessary
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
            # Step 1: Wrap the entire reasoning process inside an expandable status widget
            with st.status("🔍 Analyzing your request...", expanded=True) as status:
                response = llm_with_tools.invoke(messages)
                messages.append(response)
                
                while response.tool_calls:
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        
                        # Write the running tool status inside the status container
                        st.write(f"⚙️ Running tool **{tool_name}**...")
                        
                        selected_tool = next(t for t in tools if t.name == tool_name)
                        tool_output = selected_tool.invoke(tool_args)
                        
                        messages.append(
                            ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
                        )
                    
                    response = llm_with_tools.invoke(messages)
                    messages.append(response)
                
                # Update the status bar to show completion and collapse it!
                status.update(label="✅ Analysis complete!", state="complete", expanded=False)
                
            # Step 2: Extract and print final clean output directly to the main chat
            final_output = response.content
            if isinstance(final_output, list):
                final_output = "".join([b.get("text", "") for b in final_output if b.get("type") == "text"])
            
            st.write(final_output)
            
            # Store the final assistant message in history
            st.session_state.chat_history.append(AIMessage(content=final_output))