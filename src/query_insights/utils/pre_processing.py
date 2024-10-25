import logging
import re
from typing import List, Tuple

import fsspec
import gensim.downloader as api
import gensim.models
import mysql.connector
import nltk
import numpy as np

# import pandas as pd
import torch
from mysql.connector import errorcode
from transformers import BertModel, BertTokenizer

from core.database.database_factory import DatabaseFactory
from core.utils.client_utils import get_database_client, get_storage_client
from core.utils.read_config import cloud_config, cloud_secrets, config, secrets_config

embedding_model = (
    "BERT"  # BERT/GloVe- Use BERT for better performance,for glove use threshold = 0.6
)
threshold = 0.7  # Threshold for semantic similarity classification

# TODO: Use TigerNLP one after its moved there

MYLOGGERNAME = "QueryInsights"


class HybridQuestionClassifier:
    """Class which helps in identifying 'why' questions.

    Parameters
    ----------
    embedding_model : str, optional
        Embedding model to be used for semantic similarity classification.
        Currently, only "BERT" and "GloVe" are supported, by default "BERT"

    Raises
    ------
    ValueError
        If the given embedding model is not supported.

    Example
    -------
    >>> questions = ["what is the reason for delay in shipment?",
        "why is the shipment delayed?"]
    >>> threshold = 0.7
    >>> classifier = HybridQuestionClassifier(embedding_model="bert")
    >>> reason_based_questions = classifier.find_reason_based_questions(questions, threshold)

    >>> print("Reason based questions:")
    >>> for question, score in reason_based_questions:
            print(f"{question} (score: {score:.2f})")
    """

    def __init__(self, embedding_model: str = embedding_model):
        """Initializes a HybridQuestionClassifier object."""
        self.stopwords = set(nltk.corpus.stopwords.words("english"))
        if embedding_model.lower() == "glove":
            self.embedding_model = api.load("glove-wiki-gigaword-100")
        elif embedding_model.lower() == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.model = BertModel.from_pretrained("bert-base-uncased")

    def preprocess_question(self, question: str) -> List[str]:
        """Preprocesses a given question.

        Parameters
        ----------
        question : str
            Question to be preprocessed.

        Returns
        -------
        List[str]
            List of tokens after preprocessing.

        Example
        -------
        >>> classifier = HybridQuestionClassifier(embedding_model="bert")
        >>> question = "What is the reason for delay in shipment?"
        >>> tokens = classifier.preprocess_question(question)
        >>> print(tokens)
        ['reason', 'delay', 'shipment']
        """
        tokens = nltk.word_tokenize(question.lower())
        tokens = [
            token for token in tokens if token not in self.stopwords and token.isalnum()
        ]
        return tokens

    def question_vector(self, tokens: List[str]):
        """Creates a vector representation of a given question.

        Parameters
        ----------
        tokens : List[str]
            List of tokens of a question.

        Returns
        -------
        np.ndarray
            Vector representation of the question.

        Example
        -------
        >>> classifier = HybridQuestionClassifier(embedding_model="bert")
        >>> tokens = ["reason", "delay", "shipment"]
        >>> vector = classifier.question_vector(tokens)
        >>> print(vector)
        [-0.123 0.456 0.789]
        """
        if hasattr(self, "embedding_model"):  # GloVe
            vectors = [
                self.embedding_model[token]
                for token in tokens
                if token in self.embedding_model
            ]

            return np.mean(vectors, axis=0)
        elif hasattr(self, "model"):  # BERT
            input_ids = self.tokenizer(
                tokens, return_tensors="pt", padding=True, truncation=True
            ).data["input_ids"]
            with torch.no_grad():
                outputs = self.model(input_ids)
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
            return embeddings[0]

    def regex_based_classification(self, question: str) -> bool:
        """Classifies a given question as a reason based question using regex.

        Parameters
        ----------
        question : str
            Question to be classified.

        Returns
        -------
        bool
            True if the question is a reason based question, False otherwise.

        Example
        -------
        >>> classifier = HybridQuestionClassifier(embedding_model="bert")
        >>> question = "What is the reason for delay in shipment?"
        >>> is_reason_based = classifier.regex_based_classification(question)
        >>> print(is_reason_based)
        True
        """
        # TODO: move this pattern to config
        pattern = r"""\b(why|reason|reasons|caused|causes|explain|because|purpose|what is the (cause|causing|motivation|rationale|basis
                                |reason|source|origin|root|underlying cause|account|clarification
                                |interpretation|elucidation|description|statement)
                                |what (led to|are the (causes|factors|elements|components|aspects|variables
                                |parameters|determinants|influences))|what is the (explanation
                                |account|clarification|interpretation|elucidation|description
                                |statement|justification|defense|vindication|excuse|warrant
                                |grounds|logic|argument))\b"""
        return bool(re.search(pattern, question, re.IGNORECASE))

    def semantic_similarity_classification(self, question: str) -> float:
        """Classifies a given question as a reason based question using semantic similarity.

        Parameters
        ----------
        question : str
            Question to be classified.

        Returns
        -------
        float
            Similarity score between the question and the reason based question.

        Example
        -------
        >>> classifier = HybridQuestionClassifier(embedding_model="bert")
        >>> question = "What is the reason for delay in shipment?"
        >>> similarity_score = classifier.semantic_similarity_classification(question)
        >>> print(similarity_score)
        0.89
        """
        # TODO: Use TigerNLP one after its moved there
        reason_vector = self.question_vector(self.preprocess_question("why reason"))
        question_vector = self.question_vector(self.preprocess_question(question))

        if question_vector is None:
            return 0

        similarity = gensim.models.KeyedVectors.cosine_similarities(
            reason_vector, question_vector.reshape(1, -1)
        )[0]
        return similarity

    def classify_question(
        self, question: str, weights: Tuple[float, float] = (0.5, 0.5)
    ) -> float:
        """Classifies a given question as a reason based question using both regex and semantic similarity.

        Parameters
        ----------
        question : str
            Question to be classified.
        weights : Tuple[float, float], optional
            Weights to be used for combining regex and semantic similarity scores, by default (0.5, 0.5)

        Returns
        -------
        float
            Weighted score between the regex and semantic similarity scores.

        Example
        -------
        >>> classifier = HybridQuestionClassifier(embedding_model="bert")
        >>> question = "What is the reason for delay in shipment?"
        >>> weighted_score = classifier.classify_question(question)
        >>> print(weighted_score)
        0.89
        """
        regex_score = self.regex_based_classification(question)
        semantic_similarity_score = self.semantic_similarity_classification(question)
        weighted_score = (
            weights[0] * regex_score + weights[1] * semantic_similarity_score
        )
        return weighted_score

    def find_reason_based_questions(
        self, questions: List[str], threshold: float
    ) -> List[Tuple[str, float]]:
        """Finds reason based questions from a list of questions.

        Parameters
        ----------
        questions : List[str]
            List of questions to be classified.
        threshold : float
            Threshold to be used for filtering the questions.

        Returns
        -------
        List[Tuple[str, float]]
            List of reason based questions with their scores.

        Example
        -------
        >>> classifier = HybridQuestionClassifier(embedding_model="bert")
        >>> questions = ["What is the reason for delay in shipment?", "What is the reason for delay in shipment?"]
        >>> reason_based_questions = classifier.find_reason_based_questions(questions, threshold=0.5)
        >>> print(reason_based_questions)
        [("What is the reason for delay in shipment?", 0.89), ("What is the reason for delay in shipment?", 0.89)]
        """
        reason_based_questions = []
        for question in questions:
            score = self.classify_question(question)
            if score >= threshold:
                reason_based_questions.append((question, score))
        return sorted(reason_based_questions, key=lambda x: x[1], reverse=True)
