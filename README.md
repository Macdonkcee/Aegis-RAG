# 🛡️ Aegis-RAG Copilot

### An Agentic, Enterprise-Grade Retrieval-Augmented Generation (RAG) System

---

## 📌 Project Overview

Aegis-RAG Copilot is an end-to-end, dynamic AI assistant designed to converse with corporate intelligence. Built using **LangChain** and **Streamlit**, this agent empowers users to upload custom PDFs (like policy handbooks or technical manuals) and conduct instant, semantic queries over them. 

Equipped with advanced tool-routing capabilities and powered by `gemini-2.5-flash`, the agent dynamically decides whether to query local document vector embeddings, fetch real-time general knowledge from Wikipedia, or track market trends, all while rendering its live reasoning chain transparently to the user.

---

## 🚀 Live Demo

[🔗 Try Aegis-RAG Copilot on Streamlit Cloud](https://aegis-rag.streamlit.app/)

---

## 🛠️ Technical Stack

* **Language:** Python 3.10+
* **Orchestration:** LangChain, LangChain Community, LangChain Google GenAI
* **Vector Database:** ChromaDB
* **Embedding Model:** BAAI/bge-small-en-v1.5 (via FastEmbed)
* **LLM Engine:** Google Gemini (`gemini-2.5-flash`)
* **Web Framework:** Streamlit
* **Deployment:** GitHub & Streamlit Community Cloud

---

## 📊 Key Features & Workflow

* **Dynamic PDF Ingestion:** Instantly extracts, chunks, and vectorizes any uploaded PDF on-the-fly using `PyPDFLoader` and `RecursiveCharacterTextSplitter`.
* **Zero-Dependency Vector Store:** Leverages `FastEmbed` with a localized disk cache to bypass Windows/Linux administrative permission restrictions and eliminate heavy cold-start model downloads.
* **Agentic Multi-Tool Routing:** Binded to `gemini-2.5-flash` to selectively dispatch custom tools (`query_company_policy`, `search_wikipedia`, `fetch_stock_price`) depending on user intent.
* **Streamlit SQLite Solution:** Implements a runtime hot-swap utilizing `pysqlite3-binary` to bypass system-level SQLite library version mismatches common on cloud hosting containers.
* **Visualized Trace Logs:** Utilizes `st.status` containers to display clean, collapsible tool-execution logs during processing, hiding backend complexity but ensuring transparency.

---

## 🤝 Connect with Me

* **Lead Developer:** Kosisochukwu Ukwandu
* **Email:** macdonkcee@gmail.com
* **GitHub:** [Macdonkcee](https://github.com/Macdonkcee)
* **LinkedIn:** [Kosisochukwu Ukwandu](https://www.linkedin.com/in/kosisochukwu-ukwandu-b8695a294)
