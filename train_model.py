import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle

# ---------------------------  READ Y LABELS  ---------------------------

y_test = pd.read_csv('data/paws/test.tsv', sep='\t',
                     usecols=['label'])['label'].values

y_train = pd.read_csv('data/paws/train.tsv', sep='\t',
                      usecols=['label'])['label'].values

y_dev = pd.read_csv('data/paws/dev.tsv', sep='\t',
                    usecols=['label'])['label'].values

# ----------------------------  MERGE DATAFRAMES  ----------------------------

use_cols = ['bert_cosine_distance','2_grams_jaccard',
            '3_grams_jaccard', '4_grams_jaccard',
            'predictions_raw', 'predictions',
            'shortest_path_distance',
            'wm_distance'
            ]

X_test = pd.read_csv('data/features/test.csv', usecols=use_cols)
X_train = pd.read_csv('data/features/train.csv', usecols=use_cols)
X_dev = pd.read_csv('data/features/dev.csv', usecols=use_cols)

# Replace infinity values
X_train.replace(np.inf, 10, inplace=True)
X_test.replace(np.inf, 10, inplace=True)
X_dev.replace(np.inf, 10, inplace=True)

print(X_train.columns)
print(X_test.columns)
print(X_dev.columns)
#%%
model = LogisticRegression()
# fit the model
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
print(accuracy_score(y_test, y_pred_test))

y_pred_dev = model.predict(X_dev)
print(accuracy_score(y_dev, y_pred_dev))

pickle.dump(model, open("model_selected_features.pkl", "wb"))

X_test.head()

