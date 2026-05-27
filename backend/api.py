from fastapi import FastAPI
import sqlite3
import ollama
import time

from evaluations.metrics import evaluate_response
from evaluations.hallucination import hallucination_score
from evaluations.consistency import consistency_score
from evaluations.approval import approval_gate
from evaluations.metrics import estimate_cost
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

app = FastAPI()

# Metrics
REQUEST_COUNT = Counter(
    "llm_requests_total",
    "Total LLM requests"
)

REQUEST_LATENCY = Histogram(
    "llm_request_latency_seconds",
    "LLM request latency"
)

# Health check
@app.get("/")
def home():
    return {"message": "LLM Evaluation Framework Running"}

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )

# Evaluate endpoint
@app.post("/evaluate")
def evaluate(prompt: str):

    start = time.time()
    REQUEST_COUNT.inc()

    responses = []

    # Generate 3 responses
    for _ in range(3):

        response = ollama.chat(
            model='llama3.2',
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ]
        )

        answer = response['message']['content']

        responses.append(answer)

    # Use first response
    answer = responses[0]

    # Latency
    end = time.time()

    latency = round(end - start, 2)
    REQUEST_LATENCY.observe(latency)

    # Scores
    quality_score = evaluate_response(answer)

    hallucination = hallucination_score(answer)

    consistency = consistency_score(responses)

    estimated_cost = estimate_cost(answer)

    approval = approval_gate(
        quality_score,
        hallucination,
        consistency,
        latency
    )

    # Save database
    conn = sqlite3.connect("database/llm_logs.db")

    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO evaluations (
        prompt_version,
        prompt,
        latency,
        quality_score,
        estimated_cost,
        hallucination_score,
        consistency_score,
        approval_status,
        response
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "api_request",
        prompt,
        latency,
        quality_score,
        estimated_cost,
        hallucination,
        consistency,
        approval,
        answer
    ))

    conn.commit()

    conn.close()

    return {
        "prompt": prompt,
        "response": answer,
        "latency": float(latency),
        "quality_score": int(quality_score),
        "hallucination_score": int(hallucination),
        "consistency_score": float(consistency),
        "estimated_cost": float(estimated_cost),
        "approval_status": approval
    }