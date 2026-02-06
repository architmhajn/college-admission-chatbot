import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    return " ".join(words)


def get_response(user_query, data):

    processed_query = preprocess(user_query)

    processed_questions = [preprocess(q) for q in data['query']]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(processed_questions + [processed_query])

    similarity = cosine_similarity(vectors[-1], vectors[:-1])

    index = similarity.argmax()

    if similarity[0][index] < 0.3:
        return "Sorry, I could not understand your query."

    return data['response'][index]
