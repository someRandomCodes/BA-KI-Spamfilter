import os
import numpy as np
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# Funktion zum Laden der Vektorisierer und Modelle
def load_model_and_vectorizer(model_name):
    model_path_tfidf = model_name + '_tfidf.joblib'
    vectorizer_path_tfidf = model_name + '_tfidf_vectorizer.joblib'

    if os.path.exists(model_path_tfidf) and os.path.exists(vectorizer_path_tfidf):
        print("Lade vorhandene Modelle und Vektorisierer...")
        model_tfidf = load(model_path_tfidf)
        vectorizer_tfidf = load(vectorizer_path_tfidf)
        return model_tfidf, vectorizer_tfidf
    else:
        raise FileNotFoundError("Model or vectorizer not found!")


# Funktion zur Ausgabe der signifikanten Wörter
def print_significant_words(model, vectorizer, top_n=100):
    feature_names = vectorizer.get_feature_names_out()
    log_prob_spam = model.feature_log_prob_[1]
    log_prob_ham = model.feature_log_prob_[0]
    log_prob_diff = log_prob_spam - log_prob_ham
    sorted_indices = np.argsort(log_prob_diff)
    top_spam_words = feature_names[sorted_indices[-top_n:]]
    top_ham_words = feature_names[sorted_indices[:top_n]]

    print("Top Spam-Wörter:")
    print(top_spam_words)
    print("\nTop Ham-Wörter:")
    print(top_ham_words)


if __name__ == "__main__":
    model_name = 'nn_vectorizer_BatchedEmails.joblib_0.joblib'
    model_tfidf, vectorizer_tfidf = load_model_and_vectorizer(model_name)
    print_significant_words(model_tfidf, vectorizer_tfidf)
