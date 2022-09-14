'''
@InProceedings{pawsx2019emnlp,
  title = {{PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification}},
  author = {Yang, Yinfei and Zhang, Yuan and Tar, Chris and Baldridge, Jason},
  booktitle = {Proc. of EMNLP},
  year = {2019}
}
'''

# PAWS-Wiki
train_path = '../data/paws/train.tsv'
test_path = '../data/paws/test.tsv'
dev_path = '../data/paws/dev.tsv'

y_path = {
    'test': 'data/paws/test.tsv',
    'train': 'data/paws/train.tsv',
    'dev': 'data/paws/dev.tsv'
}

X_path = {
    'test': 'data/features/test.csv',
    'train': 'data/features/train.csv',
    'dev': 'data/features/dev.csv'
}
