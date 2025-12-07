"""
Text preprocessing utilities for the Verilog Chatbot.
Includes tokenization, lemmatization, and bag-of-words representation.
"""

import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def clean_up_sentence(sentence: str):
    """
    Tokenizes and lemmatizes an input sentence.
    Returns a list of lowercased lemmas.
    """
    words = nltk.word_tokenize(sentence)
    words = [lemmatizer.lemmatize(w.lower()) for w in words]
    return words

def bow(sentence: str, vocabulary: list):
    """
    Builds a bag-of-words vector for the input sentence.
    Output is a list matching the vocab size with 1/0 entries.
    """
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in vocabulary]
    return bag

