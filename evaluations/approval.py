def approval_gate(
    quality_score,
    hallucination_score,
    consistency_score,
    latency
):

    if (
        quality_score >= 7 and
        hallucination_score <= 1 and
        consistency_score >= 0.80 and
        latency <= 60
    ):
        return "APPROVED"

    return "REJECTED"