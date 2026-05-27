def hallucination_score(response):

    suspicious_words = [
        "always",
        "never",
        "guaranteed",
        "100%",
        "completely accurate",
        "no risk"
    ]

    score = 0

    for word in suspicious_words:
        if word.lower() in response.lower():
            score += 1

    return score