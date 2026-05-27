import streamlit as st
import pandas as pd
import sqlite3

# Page title
st.title("LLM Evaluation Dashboard")

# Connect database
conn = sqlite3.connect("database/llm_logs.db")

# Load data
query = "SELECT * FROM evaluations"

df = pd.read_sql_query(query, conn)

# Close DB connection
conn.close()

# Show dataframe
st.subheader("Evaluation Results")

st.dataframe(
    df[
        [
            "model_name",
            "prompt_version",
            "latency",
            "quality_score",
            "estimated_cost",
            "hallucination_score",
            "consistency_score",
            "approval_status"
        ]
    ]
)

# Latency chart
st.subheader("Latency Comparison")

st.bar_chart(df["latency"])

# Quality Score chart
st.subheader("Quality Score Comparison")

st.bar_chart(df["quality_score"])

# Cost chart
st.subheader("Estimated Cost Comparison")

st.bar_chart(df["estimated_cost"])

# Model latency comparison
st.subheader("Model Latency Comparison")

st.bar_chart(
    df.groupby("model_name")["latency"].mean()
)

# Approval status comparison
st.subheader("Approval Status by Model")

approval_counts = df.groupby(
    ["model_name", "approval_status"]
).size().unstack(fill_value=0)

st.bar_chart(approval_counts)

# Consistency comparison
st.subheader("Consistency Score Comparison")

st.bar_chart(
    df.groupby("model_name")["consistency_score"].mean()
)

# Hallucination comparison
st.subheader("Hallucination Score Comparison")

st.bar_chart(
    df.groupby("model_name")["hallucination_score"].mean()
)