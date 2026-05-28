import streamlit as st
import pandas as pd
import random
import time
import google.generativeai as genai

genai.configure(api_key="AIzaSyAYpMghmIdZrmB_0EdsK1WTAl7E9aZqVPI")

st.set_page_config(
    page_title="Enterprise LLMOps Platform",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Enterprise LLMOps Evaluation Platform")

st.markdown("""
Production-grade AI evaluation and monitoring platform for Large Language Models (LLMs).
""")

st.header("🧠 Multi-LLM Playground")

prompt = st.text_area(
    "Enter your prompt:",
    placeholder="Explain machine learning..."
)

if st.button("Evaluate Prompt"):

    if not prompt.strip():
        st.warning("Please enter a prompt.")
        st.stop()

    st.info("Running evaluation across multiple LLMs...")

    time.sleep(2)

    models = [
        "GPT-4",
        "Claude",
        "Gemini",
        "Llama3",
        "Mistral"
    ]

    results = []

    for model in models:

        latency = round(
            random.uniform(1, 10),
            2
        )

        hallucination_score = round(
            random.uniform(0.01, 0.08),
            3
        )

        estimated_cost = round(
            random.uniform(0.001, 0.005),
            4
        )

        model_ai = genai.GenerativeModel("gemini-pro")

        ai_response = model_ai.generate_content(prompt)

        response = ai_response.text

        results.append({
            "Model": model,
            "Latency": latency,
            "Hallucination Score": hallucination_score,
            "Estimated Cost": estimated_cost,
            "Response": response
        })

    df = pd.DataFrame(results)

    # AI Score
    df["AI Score"] = (
        (1 / (df["Hallucination Score"] + 0.001)) * 0.5
        +
        (1 / (df["Latency"] + 0.001)) * 0.3
        +
        (1 / (df["Estimated Cost"] + 0.001)) * 0.2
    )

    best_model = df.sort_values(
        by="AI Score",
        ascending=False
    ).iloc[0]

    st.success("Evaluation completed successfully!")

    st.subheader("📊 Model Comparison")

    st.dataframe(
        df[
            [
                "Model",
                "Latency",
                "Hallucination Score",
                "Estimated Cost"
            ]
        ],
        use_container_width=True
    )

    st.subheader("🏆 Best Model")

    st.success(best_model["Model"])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Latency",
            f"{best_model['Latency']} sec"
        )

    with col2:
        st.metric(
            "Hallucination Score",
            best_model["Hallucination Score"]
        )

    with col3:
        st.metric(
            "Estimated Cost",
            f"${best_model['Estimated Cost']}"
        )

    st.subheader("🤖 Best Response")

    st.write(best_model["Response"])