import streamlit as st
import pandas as pd
import time
import cohere
import re
from groq import Groq

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Enterprise Multi-LLM Evaluation Platform",
    page_icon="🚀",
    layout="wide"
)

# ---------------- API CLIENTS ---------------- #

co = cohere.Client(
    st.secrets["COHERE_API_KEY"]
)

groq_client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

# ---------------- TITLE ---------------- #

st.title("🚀 Enterprise Multi-LLM Evaluation Platform")

st.markdown("""
Production-grade AI evaluation and monitoring platform for Large Language Models (LLMs).
""")

# ---------------- INPUT ---------------- #

st.header("🧠 Multi-LLM Playground")

prompt = st.text_area(
    "Enter your prompt:",
    placeholder="Explain Kubernetes vs Docker..."
)

# ---------------- EVALUATION ---------------- #

if st.button("Evaluate Prompt"):

    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
        st.stop()

    st.info("Running evaluation across multiple LLMs...")

    models = [
        "Cohere Command",
        "Groq Llama3"
    ]

    results = []

    for model in models:

        start_time = time.time()

        response = ""

        try:

            # ---------------- COHERE ---------------- #

            if model == "Cohere Command":

                ai_response = co.chat(
                    model="command-a-03-2025",
                    message=prompt
                )

                response = ai_response.text

            # ---------------- GROQ ---------------- #

            elif model == "Groq Llama3":

                ai_response = groq_client.chat.completions.create(
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

                response = ai_response.choices[0].message.content

                if response is None or response.strip() == "":
                    response = "Groq returned empty response."

        except Exception as e:

            st.error(f"{model} Error: {str(e)}")

            response = f"""
Unable to generate AI response currently.

Technical Error:
{str(e)}
"""

        # ---------------- LATENCY ---------------- #

        latency = round(
            time.time() - start_time,
            2
        )

        # ---------------- HANDLE FAILED RESPONSES ---------------- #

        if (
            "Unable to generate AI response" in response
            or response.strip() == ""
            or "empty response" in response.lower()
        ):

            correctness_score = 0
            clarity_score = 0
            completeness_score = 0
            hallucination_score = 10
            overall_score = 0

        else:

            # ---------------- AI JUDGE EVALUATION ---------------- #

            judge_prompt = f"""
You are an expert AI evaluator.

Evaluate the following AI response.

USER PROMPT:
{prompt}

MODEL RESPONSE:
{response}

Evaluate on:

1. Correctness (0-10)
2. Clarity (0-10)
3. Completeness (0-10)
4. Hallucination Risk (0-10 where lower is better)

Return ONLY this format:

Correctness: <score>
Clarity: <score>
Completeness: <score>
Hallucination: <score>
"""

            try:

                judge_response = co.chat(
                    model="command-a-03-2025",
                    message=judge_prompt
                )

                evaluation_text = judge_response.text

                correctness_match = re.search(
                    r"Correctness:\s*(\d+)",
                    evaluation_text
                )

                clarity_match = re.search(
                    r"Clarity:\s*(\d+)",
                    evaluation_text
                )

                completeness_match = re.search(
                    r"Completeness:\s*(\d+)",
                    evaluation_text
                )

                hallucination_match = re.search(
                    r"Hallucination:\s*(\d+)",
                    evaluation_text
                )

                correctness_score = int(
                    correctness_match.group(1)
                ) if correctness_match else 5

                clarity_score = int(
                    clarity_match.group(1)
                ) if clarity_match else 5

                completeness_score = int(
                    completeness_match.group(1)
                ) if completeness_match else 5

                hallucination_score = int(
                    hallucination_match.group(1)
                ) if hallucination_match else 5

            except Exception:

                correctness_score = 5
                clarity_score = 5
                completeness_score = 5
                hallucination_score = 5

            # ---------------- OVERALL SCORE ---------------- #

            overall_score = round(
                (
                    correctness_score
                    + clarity_score
                    + completeness_score
                    + (10 - hallucination_score)
                ) / 4,
                2
            )

        # ---------------- COST ESTIMATION ---------------- #

        estimated_cost = round(
            len(response.split()) * 0.00002,
            4
        )

        # ---------------- STORE RESULTS ---------------- #

        results.append({
            "Model": model,
            "Latency": latency,
            "Correctness": correctness_score,
            "Clarity": clarity_score,
            "Completeness": completeness_score,
            "Hallucination Score": hallucination_score,
            "Estimated Cost": estimated_cost,
            "Overall Score": overall_score,
            "Response": response
        })

    # ---------------- DATAFRAME ---------------- #

    df = pd.DataFrame(results)

    st.success("Evaluation completed successfully!")

    st.subheader("📊 Model Comparison")

    st.dataframe(
        df[
            [
                "Model",
                "Latency",
                "Correctness",
                "Clarity",
                "Completeness",
                "Hallucination Score",
                "Estimated Cost",
                "Overall Score"
            ]
        ],
        use_container_width=True
    )

    # ---------------- BEST MODEL ---------------- #

    best_model = df.sort_values(
        by=["Overall Score", "Latency"],
        ascending=[False, True]
    ).iloc[0]

    st.subheader("🏆 Best Model")

    st.success(best_model["Model"])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Latency",
            f"{best_model['Latency']} sec"
        )

    with col2:
        st.metric(
            "Hallucination",
            best_model["Hallucination Score"]
        )

    with col3:
        st.metric(
            "Overall Score",
            best_model["Overall Score"]
        )

    with col4:
        st.metric(
            "Estimated Cost",
            f"${best_model['Estimated Cost']}"
        )

    # ---------------- BEST RESPONSE ---------------- #

    st.subheader("🎭 Best Response")

    with st.expander("View Full Response", expanded=True):

        st.markdown(best_model["Response"])

    # ---------------- ALL MODEL RESPONSES ---------------- #

    st.subheader("🧠 Responses From All Models")

    for result in results:

        with st.expander(f"{result['Model']} Response"):

            st.markdown(result["Response"])