from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Lazy model loading
model = None

def get_model():

    global model

    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')

    return model

def consistency_score(responses):

    sentence_model = get_model()

    embeddings = sentence_model.encode(responses)

    similarity_matrix = cosine_similarity(embeddings)

    similarities = []

    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):

            similarities.append(float(similarity_matrix[i][j]))

    if len(similarities) == 0:
        return 1.0

    return round(float(np.mean(similarities)), 2)