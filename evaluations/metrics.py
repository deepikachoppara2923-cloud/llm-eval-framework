def evaluate_response(response):

    score = 0

    # Length check
    if len(response) > 200:
        score += 2

    # Bullet point check
    if "*" in response or "-" in response:
        score += 2

    # Explanation quality
    keywords = [
        "machine learning",
        "data",
        "algorithm",
        "model"
    ]

    for word in keywords:
        if word.lower() in response.lower():
            score += 1

    return score

def estimate_cost(response):

    # Approximate token count
    tokens = len(response.split())

    # Fake cost estimation
    cost = tokens * 0.00001

    return round(cost, 5)