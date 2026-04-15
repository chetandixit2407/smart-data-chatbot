import streamlit as st
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import re

from langchain_ollama import OllamaLLM
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_core.prompts import PromptTemplate

# -------------------------------
# ✅ Page Config
# -------------------------------
st.set_page_config(page_title="AI Dashboard", layout="wide")
st.title("📊 AI Power BI Dashboard")

# -------------------------------
# ✅ SQL Extractor (FIX)
# -------------------------------
def extract_sql(text):
    match = re.search(r"(SELECT .*?;)", str(text), re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else None

# -------------------------------
# ✅ Load LLM
# -------------------------------
@st.cache_resource
def load_llm():
    return OllamaLLM(model="llama3")  # 🔥 better than mistral

llm = load_llm()

# -------------------------------
# ✅ Load DB Chain
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
        return_sql=True,   # 🔥 FIX
        return_direct=False
    )

# -------------------------------
# 📂 Upload File
# -------------------------------
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded_file:

    # -------------------------------
    # ✅ Read File
    # -------------------------------
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Clean column names
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]

    # -------------------------------
    # ✅ Store in SQLite
    # -------------------------------
    conn = sqlite3.connect("data.db")

    if st.session_state.get("last_file") != uploaded_file.name:
        df.to_sql("user_data", conn, if_exists="replace", index=False)
        st.session_state.last_file = uploaded_file.name

    # -------------------------------
    # 👀 Preview
    # -------------------------------
    st.subheader("Preview Data")
    st.dataframe(df.head(100))

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
    # 📈 Charts
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
        st.line_chart(df[num_cols].head(200))

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

            # -------------------------------
            # ✅ Extract SQL
            # -------------------------------
            sql_query = response.get("sql", None)

            if not sql_query:
                sql_query = extract_sql(str(response))

            if not sql_query:
                st.error("❌ Failed to generate SQL")
            else:
                st.code(sql_query, language="sql")

                # सुरक्षा checks
                if not sql_query.lower().startswith("select"):
                    st.error("❌ Unsafe query detected!")
                elif "drop" in sql_query.lower():
                    st.error("❌ Dangerous query blocked!")
                else:
                    try:
                        result_df = pd.read_sql_query(sql_query, conn)

                        result_df = result_df.head(50)

                        st.session_state.chat_history.append(("You", user_input))
                        st.session_state.chat_history.append(("Bot", result_df))

                    except Exception as e:
                        st.error(f"❌ SQL Error: {e}")

    # -------------------------------
    # 💬 Chat Output
    # -------------------------------
    for role, msg in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            if isinstance(msg, pd.DataFrame):
                st.dataframe(msg)

                if msg.shape[1] >= 2:
                    fig, ax = plt.subplots()
                    msg.plot(kind="bar", ax=ax)
                    st.pyplot(fig)
            else:
                st.markdown(f"**🤖 Bot:** {msg}")
