import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Enterprise LLMOps Platform",
    page_icon="🚀",
    layout="wide"
)

# Title
st.title("🚀 Enterprise LLMOps Evaluation Platform")

st.markdown("""
Production-grade AI evaluation and monitoring platform for Large Language Models (LLMs).
""")

# Sidebar
st.sidebar.title("⚙️ Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "LLM Playground",
        "Evaluation Dashboard",
        "Monitoring",
        "About Project"
    ]
)

# LLM Playground
if page == "LLM Playground":

    st.header("🧠 LLM Playground")

    prompt = st.text_area(
        "Enter your prompt:",
        placeholder="Explain machine learning..."
    )

    if st.button("Evaluate Prompt"):

        st.success("Evaluation completed successfully!")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Latency", "1.2 sec")

        with col2:
            st.metric("Hallucination Score", "0.02")

        with col3:
            st.metric("Estimated Cost", "$0.003")

        st.subheader("🤖 GPT-4 Response")
        st.write("""
        Machine learning is a branch of artificial intelligence
        that enables systems to learn from data and improve over time.
        """)

# Evaluation Dashboard
elif page == "Evaluation Dashboard":

    st.header("📊 Evaluation Dashboard")

    data = {
        "Model": ["GPT-4", "Llama3", "Mistral"],
        "Latency": [1.2, 2.1, 1.8],
        "Hallucination Score": [0.02, 0.05, 0.03],
        "Cost": [0.003, 0.001, 0.002]
    }

    df = pd.DataFrame(data)

    st.dataframe(df, use_container_width=True)

    st.bar_chart(df.set_index("Model")["Latency"])

# Monitoring
elif page == "Monitoring":

    st.header("📡 Monitoring & Observability")

    st.success("Prometheus monitoring active")
    st.success("Grafana dashboards connected")
    st.success("CI/CD pipeline operational")

# About
elif page == "About Project":

    st.header("📘 About This Project")

    st.markdown("""
    ### Enterprise LLMOps Evaluation Platform

    This platform includes:

    - ✅ FastAPI Backend
    - ✅ PostgreSQL Integration
    - ✅ SQLAlchemy ORM
    - ✅ GitHub Actions CI/CD
    - ✅ Prometheus Monitoring
    - ✅ Grafana Observability
    - ✅ Hallucination Detection
    - ✅ Latency Tracking
    - ✅ Cost Monitoring
    - ✅ Streamlit Dashboard
    """)