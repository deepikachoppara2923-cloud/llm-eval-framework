import streamlit as st
import pandas as pd
import time
import json
import re
import plotly.express as px

from cohere import Client
from groq import Groq
import google.generativeai as genai

genai.configure(
    api_key=st.secrets["GEMINI_API_KEY"]
)


def evaluate_response(prompt, ai_response):

    judge_model = genai.GenerativeModel(
        "gemini-1.5-flash"
    )

    judge_prompt = f"""
You are an expert LLM evaluator.

Evaluate the following response.

USER QUESTION:
{prompt}

MODEL RESPONSE:
{ai_response}

Score the response on:

1. Correctness (0-10)
2. Clarity (0-10)
3. Completeness (0-10)
4. Hallucination Risk (0-10 where lower is better)

Return ONLY raw JSON.
Do not use markdown.
Do not use code blocks.
Do not add explanations.

Example:

{{
    "correctness": 8,
    "clarity": 9,
    "completeness": 8,
    "hallucination": 2
}}
"""

    try:

        judge_start = time.time()

        judge_response = judge_model.generate_content(
            judge_prompt
        )

        judge_latency = round(
            time.time() - judge_start,
            2
        )

        evaluation_text = judge_response.text

        json_match = re.search(
            r"\{[\s\S]*?\}",
            evaluation_text
        )

        if not json_match:
            raise ValueError("No JSON returned by judge")

        evaluation = json.loads(
            json_match.group()
        )

        correctness = evaluation["correctness"]
        clarity = evaluation["clarity"]
        completeness = evaluation["completeness"]
        hallucination = evaluation["hallucination"]

    except Exception:

        correctness = 0
        clarity = 0
        completeness = 0
        hallucination = 10
        judge_latency = 0

    overall_score = round(
        (
            correctness * 0.4 +
            clarity * 0.2 +
            completeness * 0.2 +
            (10 - hallucination) * 0.2
        ),
        2
    )

    return (
        correctness,
        clarity,
        completeness,
        hallucination,
        overall_score,
        judge_latency
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
            prompt,
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

    # =========================================================
    # GEMINI
    # =========================================================

    try:

        start_time = time.time()

        model_ai = genai.GenerativeModel(
            "gemini-2.0-flash"
        )

        response = model_ai.generate_content(prompt)

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
            prompt,
            ai_response
        )

        
        results.append({
            "Model": "gemini-2.0-flash",
            "Latency": latency,
            "Judge Latency": judge_latency,
            "Correctness": correctness,
            "Clarity": clarity,
            "Completeness": completeness,
            "Hallucination Score": hallucination,
            "Estimated Cost": 0.015,
            "Overall Score": overall_score,
            "Response": ai_response
        })

    except Exception as e:

        st.error(f"Gemini Error")
        st.exception(e)

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