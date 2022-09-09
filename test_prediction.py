import pandas as pd
import csv
from nltk.corpus import stopwords
import nltk
import gensim.downloader as api

### Cosine dist
import torch
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoModel, AutoTokenizer

### n-grams
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

### wordnet
from wordnet.wordnet_distances import penn_to_wn, tagged_to_synset, sentence_similarity, symmetric_sentence_similarity

from torch.utils.data import Dataset, DataLoader

nltk.download('stopwords')
stop_words = stopwords.words('english')
model = api.load('word2vec-google-news-300')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]


def get_cosine(df, col1, col2):
    cosine_distances = []

    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)

    for i in range(len(df)):

        current_row = df.iloc[i]
        sentence1, sentence2 = current_row[col1], current_row[col2]

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

        cosine_distances.append(distance)
    return cosine_distances


def get_ngrams(text, n ):
    tokens = word_tokenize(text)
    tokens_clean = [word.lower() for word in tokens if word.isalpha()]
    n_grams = ngrams(tokens_clean, n)
    return [' '.join(grams) for grams in n_grams]

# get_ngrams('This is the simplest text i could think of', 3 )


def jaccard_similarity(A, B):
    # Find intersection of two sets
    nominator = A.intersection(B)

    # Find union of two sets
    denominator = A.union(B)

    # Take the ratio of sizes
    try:
        similarity = len(nominator)/len(denominator)
    except ZeroDivisionError:
        print(denominator)
        similarity = 0

    return similarity


def n_gr_sim(df, n, col1, col2):
    len_df = len(df)
    distances = []
    for i in range(len_df):
        current_row = df.iloc[i]
        s1, s2 = current_row[col1], current_row[col2]

        set1 = set(get_ngrams(s1, n))
        set2 = set(get_ngrams(s2, n))
        if i % 1000 == 0:
            print(i, set1, set2)
        try:
            distances.append(jaccard_similarity(set1, set2))
        except RuntimeError:
            print(i, set1, set2)
            distances.append(0)
            break

    return distances


def word_movers(df, col1, col2):
    # read df
    distances = []
    for i in range(len(df)):
        current_row = df.iloc[i]
        sentence1, sentence2 = current_row[col1], current_row[col2]

        # lower, split, remove stop-words
        prep_1, prep_2 = preprocess(sentence1), preprocess(sentence2)
        # word movers dist
        distance = model.wmdistance(prep_1, prep_2)
        # write to empty df
        distances.append(distance)
    return distances


def wordnet_distance(df, col1, col2):
    # read df
    distances = []

    for i in range(len(df)):
        if i % 1000 == 0:
            print(i)
        current_row = df.iloc[i]
        s1, s2 = current_row[col1], current_row[col2]


        try:
            distances.append(symmetric_sentence_similarity(s1, s2))
        except ValueError:
            print(s1, s2)
            distances.append(0)

    return distances


df = pd.read_csv('data/paws/test.tsv', sep='\t', nrows=3)
df.head()


from roberta.predict import Pairs_Dataset, config, Classifier_Model, predict
from transformers import AutoTokenizer
model_name = config['model_name']

tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset_test = Pairs_Dataset('data/paws/test.tsv', tokenizer, 'label', 'sentence1', 'sentence2')

BATCH_SIZE = config['batch_size']
test_loader = DataLoader(dataset_test, BATCH_SIZE, shuffle=False)

model_roberta = Classifier_Model.load_from_checkpoint('roberta/weights-50-epochs.ckpt', config=config)
model_roberta.to(device)

preditions_train_raw, predictions_train_int = predict(test_loader, model_roberta)

use_cols = ['bert_cosine_distance','2_grams_jaccard',
            '3_grams_jaccard',  '4_grams_jaccard',
            'predictions_raw', 'predictions',
            'shortest_path_distance',
            'wm_distance'
            ]

cos = get_cosine(df, 'sentence1', 'sentence2')
gr_2 = n_gr_sim(df, 2, 'sentence1', 'sentence2')
gr_3 = n_gr_sim(df, 3, 'sentence1', 'sentence2')
gr_4 = n_gr_sim(df, 4, 'sentence1', 'sentence2')
wn = wordnet_distance(df, 'sentence1', 'sentence2')
wm = word_movers(df, 'sentence1', 'sentence2')

X_test = pd.DataFrame({
    'bert_cosine_distance': cos,
    '2_grams_jaccard': gr_2,
    '3_grams_jaccard': gr_3,
    '4_grams_jaccard': gr_4,
    'predictions_raw': preditions_train_raw[2],
    'predictions': predictions_train_int[2],
    'shortest_path_distance': wn,
    'wm_distance': wm
})

X_test.head()

# X_test.to_csv('microsoft_features.csv')

df.head()

import pickle
loaded_model = pickle.load(open('model_selected_features.pkl', 'rb'))
predicted = loaded_model.predict(X_test)

print('Predicted label', predicted[2])

print(df.loc[2, 'sentence1'])
print(df.loc[2, 'sentence2'])







