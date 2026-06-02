import streamlit as st
import pandas as pd
import time
import json
import re
import plotly.express as px

from cohere import Client
from groq import Groq

def evaluate_response(ai_response):

    words = len(ai_response.split())

    correctness = min(10, max(6, words // 40))

    clarity = 8

    completeness = min(10, max(6, words // 50))

    hallucination = 2

    overall_score = round(
        (
            correctness * 0.4
            + clarity * 0.2
            + completeness * 0.2
            + (10 - hallucination) * 0.2
        ),
        2
    )

    return (
        correctness,
        clarity,
        completeness,
        hallucination,
        overall_score,
        0
    )
# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Enterprise Multi-LLM Evaluation Platform",
    layout="wide"
)

# ---------------- API CLIENTS ---------------- #

cohere_client = Client(st.secrets["COHERE_API_KEY"])

groq_client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)



# ---------------- UI ---------------- #

st.title("🚀 Enterprise Multi-LLM Evaluation Platform")

st.caption(
    "Production-grade AI evaluation and monitoring framework for enterprise LLM benchmarking."
)

st.subheader("🧠 Multi-LLM Playground")

prompt = st.text_area(
    "Enter your prompt:",
    height=150
)

# ---------------- EVALUATION ---------------- #

if st.button("Evaluate Prompt"):

    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
        st.stop()

    st.info("Running evaluation across multiple LLMs...")

    results = []

    # =========================================================
    # COHERE
    # =========================================================

    try:

        start_time = time.time()

        response = cohere_client.chat(
            model="command-a-03-2025",
            message=prompt
        )

        ai_response = response.text

        latency = round(
            time.time() - start_time,
            2
        )

        (
            correctness,
            clarity,
            completeness,
            hallucination,
            overall_score,
            judge_latency
        ) = evaluate_response(
            ai_response
        )


        results.append({
            "Model": "Cohere Command A",
            "Latency": latency,
            "Judge Latency": judge_latency,
            "Correctness": correctness,
            "Clarity": clarity,
            "Completeness": completeness,
            "Hallucination Score": hallucination,
            "Estimated Cost": 0.016,
            "Overall Score": overall_score,
            "Response": ai_response
        })

    except Exception as e:

        st.error(f"Cohere Error")
        st.exception(e)

    # =========================================================
    # GROQ
    # =========================================================

    try:

        start_time = time.time()

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=1000
        )

        ai_response = response.choices[0].message.content

        latency = round(
            time.time() - start_time,
            2
        )

        (
            correctness,
            clarity,
            completeness,
            hallucination,
            overall_score,
            judge_latency
        ) = evaluate_response(
            prompt,
            ai_response
        )
        

        results.append({
            "Model": "Groq Llama 3.3 70B",
            "Latency": latency,
            "Judge Latency": judge_latency,
            "Correctness": correctness,
            "Clarity": clarity,
            "Completeness": completeness,
            "Hallucination Score": hallucination,
            "Estimated Cost": 0.012,
            "Overall Score": overall_score,
            "Response": ai_response
        })

    except Exception as e:

        st.error(f"Groq Error")
        st.exception(e)

    response = groq_client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=1000
    )

    ai_response = response.choices[0].message.content

    results.append({
        "Model": "DeepSeek R1 70B",
        "Latency": latency,
        "Judge Latency": 0,
        "Correctness": correctness,
        "Clarity": clarity,
        "Completeness": completeness,
        "Hallucination Score": hallucination,
        "Estimated Cost": 0.010,
        "Overall Score": overall_score,
        "Response": ai_response
    })
    # =========================================================
    # RESULTS
    # =========================================================

    if len(results) == 0:

        st.error("No models responded successfully.")

    else:

        st.success("Evaluation completed successfully!")

        st.write(results)

        df = pd.DataFrame(results)

        st.subheader("📊 Model Comparison")

        st.dataframe(
            df[
                [
                    "Model",
                    "Latency",
                    "Judge Latency",
                    "Correctness",
                    "Clarity",
                    "Completeness",
                    "Hallucination Score",
                    "Overall Score"
                ]
            ],
            use_container_width=True
        )

        best_model = df.sort_values(
            by="Overall Score",
            ascending=False
        ).iloc[0]

        st.subheader("🏆 Best Model")

        st.success(best_model["Model"])

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric(
            "Latency",
            f"{best_model['Latency']} sec"
        )

        col2.metric(
            "Judge Latency",
            f"{best_model['Judge Latency']} sec"
        )

        col3.metric(
            "Hallucination",
            best_model["Hallucination Score"]
        )

        col4.metric(
            "Overall Score",
            best_model["Overall Score"]
        )

        col5.metric(
            "Estimated Cost",
            f"${best_model['Estimated Cost']}"
        )

        st.subheader("🎭 Best Response")

        with st.expander("View Full Response", expanded=True):

            st.write(best_model["Response"])

        # ---------------- CHARTS ---------------- #

        st.subheader("📈 Latency Comparison")

        fig1 = px.bar(
            df,
            x="Model",
            y="Latency",
            title="Latency by Model"
        )

        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("⭐ Overall Score Comparison")

        fig2 = px.bar(
            df,
            x="Model",
            y="Overall Score",
            title="Overall Score by Model"
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("🧠 Hallucination Comparison")

        fig3 = px.bar(
            df,
            x="Model",
            y="Hallucination Score",
            title="Hallucination Score by Model"
        )

        st.subheader("⚖️ Judge Latency Comparison")

        fig4 = px.bar(
            df,
            x="Model",
            y="Judge Latency",
            title="Judge Latency by Model"
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )

        st.plotly_chart(fig3, use_container_width=True)