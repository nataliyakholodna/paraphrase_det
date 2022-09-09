from constants import train_path, test_path, dev_path
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.util import ngrams


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


mode = 'train'
num_grams = 2

df_this = pd.read_csv(f'../data/paws/{mode}.tsv', sep='\t')


def n_gr_sim(df, n):
    len_df = len(df)
    distances = []
    for i in range(len_df):
        current_row = df.iloc[i]
        s1, s2 = current_row['sentence1'], current_row['sentence2']

        set1 = set(get_ngrams(s1, n))
        set2 = set(get_ngrams(s2, n))
        if i % 1000 == 0:
            print(i, set1, set2)
        try:
            distances.append(jaccard_similarity(set1, set2))
        except RuntimeError:
            print(i, set1, set2)
            break

    return distances


train_d = pd.DataFrame({
    'id': df_this['id'].values,
    f'{num_grams}_grams_jaccard': n_gr_sim(df_this, num_grams)
}).to_csv(f'distances/{mode}_jaccard_{num_grams}.csv', index=False)

