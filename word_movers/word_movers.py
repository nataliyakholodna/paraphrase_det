import pandas as pd
from nltk.corpus import stopwords
from nltk import download
import gensim.downloader as api

from constants import test_path, train_path, dev_path

download('stopwords')
stop_words = stopwords.words('english')
model = api.load('word2vec-google-news-300')


def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]


# df = pd.read_csv('../data/paws/dev.tsv', sep='\t', nrows=10)
#
# for i in range(10):
#     current_row = df.iloc[i]
#     sentence1, sentence2 = current_row['sentence1'], current_row['sentence2']
#     prep_1, prep_2 = preprocess(sentence1), preprocess(sentence2)
#     print(sentence1, prep_1)
#     distance = model.wmdistance(prep_1, prep_2)
#     print('distance = %.4f' % distance)


def word_movers(csv_in_path, csv_out_path):
    # read df
    df = pd.read_csv(csv_in_path, sep='\t')
    # empty df, columns - id and distance
    df_wm = pd.DataFrame(index=range(len(df)), columns=['id', 'wm_distance'])

    for i in range(len(df)):
        current_row = df.iloc[i]
        sentence1, sentence2 = current_row['sentence1'], current_row['sentence2']

        # lower, split, remove stop-words
        prep_1, prep_2 = preprocess(sentence1), preprocess(sentence2)
        # word movers dist
        distance = model.wmdistance(prep_1, prep_2)
        # write to empty df
        df_wm.loc[i, 'id'] = current_row['id']
        df_wm.loc[i, 'wm_distance'] = distance

    # save df
    df_wm.to_csv(csv_out_path, index=False)
    # print info
    print('Saved', len(df_wm), 'rows to', csv_out_path)


# test func
word_movers(test_path, 'distances/test2_wm.csv')
