# === Imports ===
import pandas as pd
import streamlit as st
import pickle
from nltk.corpus import stopwords
import nltk
import gensim.downloader as api

### Cosine dist
import torch
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoModel, AutoTokenizer
from roberta.predict import Pairs_Dataset, config, Classifier_Model, predict

### n-grams
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

### wordnet
from wordnet.wordnet_distances import penn_to_wn, tagged_to_synset, sentence_similarity, symmetric_sentence_similarity

from torch.utils.data import Dataset, DataLoader

nltk.download('stopwords',  quiet=True)
stop_words = stopwords.words('english')
model = api.load('word2vec-google-news-300')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sent1 = "Bradd Crellin represented BARLA Cumbria on a tour of Australia with 6 other players representing Britain , " \
        "also on a tour of Australia . "
sent2 = 'Bradd Crellin also represented BARLA Great Britain on a tour through Australia on a tour through Australia ' \
        'with 6 other players representing Cumbria . '


def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]


def get_cosine(sentence1, sentence2):
    """
    Feature # _.
    Calculate cosine similarity between vector embeddings,
    obtained by a forward pass of BERT pretrained model.

    :param sentence1:
    :param sentence2:
    :return: float
    """
    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)

    # {'input_ids': tensor([[...]]), 'attention_mask': tensor([[...]])}
    tokens_dict_first = tokenizer(sentence1, padding='max_length',
                                  truncation=True,
                                  max_length=128,
                                  return_tensors='pt',
                                  add_special_tokens=True)

    # unpack, send to gpu
    input_ids1, attention_mask1 = tokens_dict_first['input_ids'], tokens_dict_first['attention_mask']
    input_ids1, attention_mask1 = input_ids1.to(device), attention_mask1.to(device)

    tokens_dict_second = tokenizer(sentence2, padding='max_length',
                                   truncation=True,
                                   max_length=128,
                                   return_tensors='pt',
                                   add_special_tokens=True)

    input_ids2, attention_mask2 = tokens_dict_second['input_ids'], tokens_dict_second['attention_mask']
    input_ids2, attention_mask2 = input_ids2.to(device), attention_mask2.to(device)

    outputs1 = model(input_ids1, attention_mask1)
    # torch.Size([1, 128, 768])
    # [num_samples, max_sentence_length, embedding_dimension]
    embeddings1 = outputs1.last_hidden_state
    # mask: torch.Size([1, 128]) --> [num_samples, max_sentence_length]
    # add embedding dimension
    mask1 = attention_mask1.unsqueeze(-1).expand(embeddings1.shape).float()
    mask_embeddings1 = embeddings1 * mask1
    # torch.Size([1, 768])
    pulled_output1 = torch.mean(mask_embeddings1, 1).detach().cpu().numpy()

    # repeat same steps
    outputs2 = model(input_ids2, attention_mask2)
    embeddings2 = outputs2.last_hidden_state

    mask2 = attention_mask2.unsqueeze(-1).expand(embeddings2.shape).float()
    mask_embeddings2 = embeddings2 * mask2
    pulled_output2 = torch.mean(mask_embeddings2, 1).detach().cpu().numpy()

    # cos dist between two sentence vectors
    distance = cosine_similarity(pulled_output1, pulled_output2)[0][0]

    return distance


def get_ngrams(text, n):
    """
    Helper function.
    :param text:
    :param n:
    :return:
    """
    tokens = word_tokenize(text)
    tokens_clean = [word.lower() for word in tokens if word.isalpha()]
    n_grams = ngrams(tokens_clean, n)
    return [' '.join(grams) for grams in n_grams]


def jaccard_similarity(a, b):
    """
    Helper function.
    :param a:
    :param b:
    :return:
    """
    # Find intersection of two sets
    nominator = a.intersection(b)

    # Find union of two sets
    denominator = a.union(b)

    # Take the ratio of sizes
    try:
        similarity = len(nominator) / len(denominator)
    except ZeroDivisionError:
        print(denominator)
        similarity = 0

    return similarity


def n_gram_sim(s1, s2, n):
    """
    Feature #2.
    Jaccard n-gram similarity.
    :param s1:
    :param s2:
    :param n:
    :return:
    """
    set1 = set(get_ngrams(s1, n))
    set2 = set(get_ngrams(s2, n))
    try:
        distance = jaccard_similarity(set1, set2)
    except RuntimeError:
        print(s1, s2)
        distance = 0
    return distance


def word_movers(s1, s2):
    # lower, split, remove stop-words
    prep_1, prep_2 = preprocess(s1), preprocess(s2)
    # word movers dist
    distance = model.wmdistance(prep_1, prep_2)
    return distance


def wordnet_distance(s1, s2):
    try:
        distance = symmetric_sentence_similarity(s1, s2)
    except ValueError:
        distance = 0
    return distance


def roberta_prediction(s1, s2):
    model_name = 'roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens_dict = tokenizer(s1, s2, padding='max_length', truncation=True,
                            max_length=128,
                            return_tensors='pt',
                            add_special_tokens=True)
    model_roberta = Classifier_Model.load_from_checkpoint('roberta/weights-50-epochs.ckpt', config=config)
    model_roberta.to(device)

    with torch.no_grad():
        model_roberta.eval()
        ids = tokens_dict['input_ids']
        mask = tokens_dict['attention_mask']
        label = torch.tensor(0, dtype=torch.float32)

        ids = ids.to(device)
        mask = mask.to(device)
        label = label.to(device)

        _, outputs = model_roberta(ids, mask, label)

        pred = torch.sigmoid(outputs).detach().cpu().numpy()[0][0]
        pred_int = 1 if pred > 0.5 else 0

    return pred, pred_int


pred_raw, pred_int = roberta_prediction(sent1, sent2)

X = [[
    get_cosine(sent1, sent2),
    n_gram_sim(sent1, sent2, 3),
    n_gram_sim(sent1, sent2, 2),
    n_gram_sim(sent1, sent2, 4),
    pred_raw,
    pred_int,
    wordnet_distance(sent1, sent2),
    word_movers(sent1, sent2)
]]

print(X)

loaded_model = pickle.load(open('model_selected_features.pkl', 'rb'))
predicted = loaded_model.predict(X)

print('Predicted label', predicted[0])
print('Predicted probability', loaded_model.predict_proba(X))

