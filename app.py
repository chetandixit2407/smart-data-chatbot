import streamlit as st
import pandas as pd
import sqlite3

from langchain_ollama import OllamaLLM

from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain 

st.title("📊 Smart Data Chatbot")

# ✅ Cache LLM (performance boost 🚀)
@st.cache_resource
def load_llm():
    return OllamaLLM(model="llama3")
llm = load_llm()
# ✅ Upload file
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"]) 

if uploaded_file is not None:

    # ✅ Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, sep=";")
    else:
        df = pd.read_excel(uploaded_file)

    st.write("Preview Data:", df.head())

    # ✅ Clean column names
    df.columns = [
        col.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(";", "")
        for col in df.columns
    ]

    # ✅ Store only once (avoid reload issue)
    conn = sqlite3.connect("data.db")

    if "data_loaded" not in st.session_state:
        df.to_sql("user_data", conn, if_exists="replace", index=False)
        st.session_state.data_loaded = True

    st.success("Data stored in database ✅")

    # ✅ Load LLM
    llm = load_llm()

    # ✅ DB
    db = SQLDatabase.from_uri(
        "sqlite:///data.db",
        sample_rows_in_table_info=2
    )

    db_chain = SQLDatabaseChain.from_llm(
        llm,
        db,
        return_direct=True,
        use_query_checker=True
    )

    st.write("Available columns:", df.columns.tolist())

    # ✅ Chat UI (better UX 🔥)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask your question")

    if user_input:
        user_input = user_input + " (Do not use double quotes in SQL query)"

        with st.spinner("Thinking..."):
            response = db_chain.invoke({"query": user_input})

            # Save chat
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", response))

    # ✅ Show chat history
    for role, msg in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            if isinstance(msg, list):
                st.dataframe(pd.DataFrame(msg))
            else:
                st.markdown(f"**🤖 Bot:** {msg}")