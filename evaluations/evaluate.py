import ollama
import time
import os
import csv
import mlflow
import pandas as pd

from sqlalchemy import create_engine

from metrics import evaluate_response
from metrics import estimate_cost
from hallucination import hallucination_score
from consistency import consistency_score
from approval import approval_gate


# -----------------------------
# MLflow Setup
# -----------------------------

mlflow.set_experiment("LLM_Evaluation")


# -----------------------------
# Prompt Folder
# -----------------------------

PROMPT_FOLDER = "prompts"

prompt_files = os.listdir(PROMPT_FOLDER)


# -----------------------------
# Models
# -----------------------------

models = [
    "llama3.2",
    "mistral",
    "gemma:2b"
]


# -----------------------------
# PostgreSQL Connection
# -----------------------------

DATABASE_URL = "postgresql://deepikachoppara@localhost:5432/llm_eval_db"

engine = create_engine(DATABASE_URL)


# -----------------------------
# CSV File
# -----------------------------

csv_file = "evaluations/results.csv"


# -----------------------------
# Create CSV Header
# -----------------------------

with open(csv_file, mode='w', newline='') as file:

    writer = csv.writer(file)

    writer.writerow([
        "Model Name",
        "Prompt Version",
        "Prompt",
        "Latency",
        "Quality Score",
        "Estimated Cost",
        "Hallucination Score",
        "Consistency Score",
        "Approval Status",
        "Response"
    ])

    print("\n===== PROMPT VERSION EVALUATION =====\n")

    # -----------------------------
    # Model Loop
    # -----------------------------

    for model_name in models:

        print(f"\n===== TESTING MODEL: {model_name} =====")

        # -----------------------------
        # Prompt Loop
        # -----------------------------

        for prompt_file in prompt_files:

            mlflow.end_run()

            with mlflow.start_run(
                run_name=f"{model_name}_{prompt_file}"
            ):

                file_path = os.path.join(
                    PROMPT_FOLDER,
                    prompt_file
                )

                # Read Prompt
                with open(file_path, "r") as f:
                    prompt = f.read()

                print(f"\nEvaluating: {prompt_file}")

                # -----------------------------
                # MLflow Parameters
                # -----------------------------

                mlflow.log_param(
                    "model_name",
                    model_name
                )

                mlflow.log_param(
                    "prompt_version",
                    prompt_file
                )

                # -----------------------------
                # Start Timer
                # -----------------------------

                start = time.time()

                # -----------------------------
                # Generate Multiple Responses
                # -----------------------------

                responses = []

                for _ in range(3):

                    response = ollama.chat(
                        model=model_name,
                        messages=[
                            {
                                'role': 'user',
                                'content': prompt
                            }
                        ]
                    )

                    answer = response['message']['content']

                    responses.append(answer)

                # -----------------------------
                # Use First Response
                # -----------------------------

                answer = responses[0]

                # -----------------------------
                # End Timer
                # -----------------------------

                end = time.time()

                latency = round(end - start, 2)

                # -----------------------------
                # Evaluation Metrics
                # -----------------------------

                score = evaluate_response(answer)

                cost = estimate_cost(answer)

                hallucination = hallucination_score(answer)

                consistency = consistency_score(responses)

                approval = approval_gate(
                    score,
                    hallucination,
                    consistency,
                    latency
                )

                # -----------------------------
                # MLflow Metrics
                # -----------------------------

                mlflow.log_metric(
                    "latency",
                    latency
                )

                mlflow.log_metric(
                    "quality_score",
                    score
                )

                mlflow.log_metric(
                    "hallucination_score",
                    hallucination
                )

                mlflow.log_metric(
                    "consistency_score",
                    consistency
                )

                mlflow.log_metric(
                    "estimated_cost",
                    cost
                )

                mlflow.log_param(
                    "approval_status",
                    approval
                )

                # -----------------------------
                # Save Response Artifact
                # -----------------------------

                with open("response.txt", "w") as f:
                    f.write(answer)

                mlflow.log_artifact("response.txt")

                # -----------------------------
                # Save to CSV
                # -----------------------------

                writer.writerow([
                    model_name,
                    prompt_file,
                    prompt,
                    latency,
                    score,
                    cost,
                    hallucination,
                    consistency,
                    approval,
                    answer
                ])

                print("\nLatency:", latency, "sec")


# -----------------------------
# Save CSV to PostgreSQL
# -----------------------------

df = pd.read_csv(csv_file)

df.to_sql(
    "evaluations",
    engine,
    if_exists="append",
    index=False
)


# -----------------------------
# Log CSV Artifact
# -----------------------------

mlflow.log_artifact(csv_file)

print("\nResults saved to results.csv")