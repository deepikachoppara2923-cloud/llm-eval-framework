import streamlit as st
import pandas as pd
import time
import json
import re
import plotly.express as px

from cohere import Client
from groq import Groq

def evaluate_response(prompt, ai_response):

    try:

        judge_start = time.time()

        judge_prompt = f"""
You are a strict enterprise LLM evaluator.

Evaluate the response critically.

Do NOT give high scores by default.

A score of 10 should be extremely rare.

Penalize:
- Missing information
- Incorrect facts
- Weak reasoning
- Unsupported claims
- Repetitive content

USER QUESTION:
{prompt}

MODEL RESPONSE:
{ai_response}

Evaluate:

- correctness (0-10)
- clarity (0-10)
- completeness (0-10)
- hallucination (0-10, lower is better)
- reasoning (0-10)

Return ONLY valid JSON:

{{
    "correctness": 0,
    "clarity": 0,
    "completeness": 0,
    "hallucination": 0,
    "reasoning": 0,
    "feedback": ""
}}
"""

        judge_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": judge_prompt
                }
            ],
            temperature=0
        )

        judge_latency = round(
            time.time() - judge_start,
            2
        )

        evaluation_text = (
            judge_response
            .choices[0]
            .message
            .content
        )

        json_match = re.search(
            r"\{[\s\S]*?\}",
            evaluation_text
        )

        evaluation = json.loads(
            json_match.group()
        )

        correctness = float(
            evaluation["correctness"]
        )

        clarity = float(
            evaluation["clarity"]
        )

        completeness = float(
            evaluation["completeness"]
        )

        hallucination = float(
            evaluation["hallucination"]
        )

        reasoning = float(
            evaluation["reasoning"]
        )

        feedback = evaluation["feedback"]

    except Exception:

        correctness = 5
        clarity = 5
        completeness = 5
        hallucination = 5
        reasoning = 5
        feedback = "Judge evaluation failed."
        judge_latency = 0

    overall_score = round(
        (
            correctness * 0.25 +
            clarity * 0.20 +
            completeness * 0.25 +
            reasoning * 0.20 +
            (10 - hallucination) * 0.10
        ),
        2
    )

    return (
        correctness,
        clarity,
        completeness,
        hallucination,
        reasoning,
        feedback,
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
    "Enterprise Multi-LLM Evaluation and Benchmarking Platform using LLM-as-a-Judge."
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
            reasoning,
            feedback,
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
            "Reasoning": reasoning,
            "Judge Feedback": feedback,
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
            reasoning,
            feedback,
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
            "Reasoning": reasoning,
            "Judge Feedback": feedback,
            "Overall Score": overall_score,
            "Response": ai_response
        })

    except Exception as e:

        st.error(f"Groq Error")
        st.exception(e)

    # =========================================================
    # LLAMA 3.1 8B
    # =========================================================

    try:

        start_time = time.time()

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
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
            reasoning,
            feedback,
            overall_score,
            judge_latency
        ) = evaluate_response(
            prompt,
            ai_response
        )

        results.append({
            "Model": "Llama 3.1 8B",
            "Latency": latency,
            "Judge Latency": judge_latency,
            "Correctness": correctness,
            "Clarity": clarity,
            "Completeness": completeness,
            "Hallucination Score": hallucination,
            "Reasoning": reasoning,
            "Judge Feedback": feedback,
            "Overall Score": overall_score,
            "Response": ai_response
        })

    except Exception as e:

        st.error("Llama 3.1 8B Error")
        st.exception(e)
    # =========================================================
    # RESULTS
    # =========================================================

    if len(results) == 0:

        st.error("No models responded successfully.")

    else:

        st.success("Evaluation completed successfully!")

        

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
                    "Reasoning",
                    "Judge Feedback"
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
        st.info(
            f"""
        Reasoning: {best_model['Reasoning']}
        Correctness: {best_model['Correctness']}
        Completeness: {best_model['Completeness']}
        Hallucination: {best_model['Hallucination Score']}
        """
        )

        col1, col2, col3, col4 = st.columns(4)

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

        st.subheader("🎭 Best Response")

        with st.expander("View Full Response", expanded=True):

            st.write(best_model["Response"])

            st.subheader("🧑‍⚖️ Judge Feedback")

            st.info(best_model["Judge Feedback"])

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

        st.subheader("🧩 Reasoning Comparison")

        fig5 = px.bar(
            df,
            x="Model",
            y="Reasoning",
            title="Reasoning Score by Model"
        )

        st.plotly_chart(
            fig5,
            use_container_width=True
        )

        st.plotly_chart(fig3, use_container_width=True)

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

        