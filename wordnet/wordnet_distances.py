import nltk
import pandas as pd
from constants import test_path, train_path, dev_path
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')


def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'

    if tag.startswith('V'):
        return 'v'

    if tag.startswith('J'):
        return 'a'

    if tag.startswith('R'):
        return 'r'

    return None


def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None


def sentence_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag

    words1 = word_tokenize(sentence1)
    tokens1 = [word.lower() for word in words1 if word.isalpha()]
    words2 = word_tokenize(sentence2)
    tokens2 = [word.lower() for word in words2 if word.isalpha()]

    sentence1 = pos_tag(tokens1)
    sentence2 = pos_tag(tokens2)

    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]

    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]

    score, count = 0.0, 0.00001

    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        # 1 - synset.path_similarity
        # 2 - synset.lch_similarity, Leacock-Chodorow Similarity
        # 3 - synset.wup_similarity, Wu-Palmer Similarity
        best_score = max([synset.path_similarity(ss) for ss in synsets2 if ss.pos == synset.pos])

        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1

    # Average the values
    score /= count
    return score


def symmetric_sentence_similarity(sentence1, sentence2):
    """ compute the symmetric sentence similarity using Wordnet """
    return (sentence_similarity(sentence1, sentence2)
            + sentence_similarity(sentence2, sentence1)) / 2


def wordnet_distance(csv_in_path, csv_out_path):
    # read df
    df = pd.read_csv(csv_in_path, sep='\t')
    # empty df, columns - id and distance
    df_wordnet = pd.DataFrame(index=range(len(df)), columns=['id', 'lch_similarity'])

    for i in range(len(df)):
        if i % 1000 == 0:
            print(i)
        current_row = df.iloc[i]
        s1, s2 = current_row['sentence1'], current_row['sentence2']

        # write to empty df
        df_wordnet.loc[i, 'id'] = current_row['id']
        try:
            df_wordnet.loc[i, 'lch_similarity'] = symmetric_sentence_similarity(s1, s2)
        except ValueError:
            print(s1, s2)
            df_wordnet.loc[i, 'lch_similarity'] = 0

    # save df
    df_wordnet.to_csv(csv_out_path, index=False)
    # print info
    print('Saved', len(df_wordnet), 'rows to', csv_out_path)


# # test func
# wordnet_distance(test_path, 'distances/test_lch_similarity.csv')
# wordnet_distance(train_path, 'distances/train_lch_similarity.csv')
# wordnet_distance(dev_path, 'distances/dev_lch_similarity.csv')

#%%
