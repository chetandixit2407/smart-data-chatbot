import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import ast

from langchain_ollama import OllamaLLM
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_core.prompts import PromptTemplate   # ✅ FIXED IMPORT

# -------------------------------
# ✅ Page Config
# -------------------------------
st.set_page_config(page_title="AI Dashboard", layout="wide")
st.title("📊 AI Power BI Dashboard")

# -------------------------------
# ✅ Load LLM
# -------------------------------
@st.cache_resource
def load_llm():
    return OllamaLLM(model="mistral")

llm = load_llm()

# -------------------------------
# ✅ Load DB Chain (FIXED)
# -------------------------------
@st.cache_resource
def load_db_chain():

    db = SQLDatabase.from_uri("sqlite:///data.db")

    prompt = PromptTemplate(
        input_variables=["input", "table_info"],
        template="""
You are an expert SQLite SQL generator.

STRICT RULES:
- Use only table: user_data
- NEVER use quotes around column names
- ALWAYS use correct column names
- ALWAYS LIMIT results to 50 rows
- Return ONLY SQL query (no explanation)

Schema:
{table_info}

Examples:
SELECT MAX(total_profit) FROM user_data;
SELECT * FROM user_data LIMIT 50;

Question: {input}
"""
    )

    return SQLDatabaseChain.from_llm(
        llm,
        db,
        prompt=prompt,
        use_query_checker=True,
        return_direct=False
    )

# -------------------------------
# 📂 Upload File
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded_file:

    # -------------------------------
    # ✅ Read file
    # -------------------------------
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df.columns = [col.lower().replace(" ", "_") for col in df.columns]

    # -------------------------------
    # ✅ Store in DB
    # -------------------------------
    conn = sqlite3.connect("data.db")

    if st.session_state.get("last_file") != uploaded_file.name:
        df.to_sql("user_data", conn, if_exists="replace", index=False)
        st.session_state.last_file = uploaded_file.name

    # -------------------------------
    # 👀 Preview (LIMITED)
    # -------------------------------
    st.subheader("Preview Data")
    st.dataframe(df.head(100))   # 🔥 LIMIT

    # -------------------------------
    # 📊 KPI
    # -------------------------------
    st.markdown("## 📊 Dashboard Overview")

    col1, col2, col3 = st.columns(3)

    num_cols = df.select_dtypes(include='number').columns

    if len(num_cols) > 0:
        col1.metric("Total", round(df[num_cols[0]].sum(), 2))
        col2.metric("Average", round(df[num_cols[0]].mean(), 2))
        col3.metric("Max", round(df[num_cols[0]].max(), 2))

    # -------------------------------
    # 📈 Charts (LIMITED)
    # -------------------------------
    st.markdown("## 📈 Visualizations")

    cat_cols = df.select_dtypes(include='object').columns
    num_cols = df.select_dtypes(include='number').columns

    if len(cat_cols) > 0 and len(num_cols) > 0:

        chart_data = (
            df.groupby(cat_cols[0])[num_cols[0]]
            .sum()
            .reset_index()
            .sort_values(by=num_cols[0], ascending=False)
            .head(20)
        )

        col1, col2 = st.columns(2)

        with col1:
            st.bar_chart(chart_data.set_index(cat_cols[0]))

        with col2:
            fig, ax = plt.subplots()
            chart_data.set_index(cat_cols[0])[num_cols[0]].plot(
                kind='pie', autopct='%1.1f%%', ax=ax
            )
            st.pyplot(fig)

    if len(num_cols) > 0:
        st.line_chart(df[num_cols].head(200))  # 🔥 LIMIT

    # -------------------------------
    # 🤖 AI Chatbot
    # -------------------------------
    st.markdown("## 🤖 Ask AI")

    db_chain = load_db_chain()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask your data question")

    if user_input:

        with st.spinner("⚡ Generating..."):

            response = db_chain.invoke({"query": user_input})

            raw = response.get("result", response)

            # -------------------------------
            # ✅ SAFE PARSE
            # -------------------------------
            try:
                data = ast.literal_eval(str(raw))
            except:
                data = []

            # 🔥 LIMIT RESULT
            if isinstance(data, list):
                data = data[:50]

            # 🔥 HANDLE SINGLE VALUE
            if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], tuple) and len(data[0]) == 1:
                    data = [{"value": data[0][0]}]

            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", data))

    # -------------------------------
    # 💬 Chat Output
    # -------------------------------
    for role, msg in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            if isinstance(msg, list) and len(msg) > 0:

                df_res = pd.DataFrame(msg).head(50)
                st.dataframe(df_res)

                if df_res.shape[1] >= 2:
                    fig, ax = plt.subplots()
                    df_res.plot(kind="bar", ax=ax)
                    st.pyplot(fig)
            else:
                st.markdown(f"**🤖 Bot:** {msg}")
