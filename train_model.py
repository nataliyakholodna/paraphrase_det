import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle
from constants import X_path, y_path

y_test = pd.read_csv(y_path['test'], sep='\t',
                     usecols=['label'])['label'].values

y_train = pd.read_csv(y_path['train'], sep='\t',
                      usecols=['label'])['label'].values

y_dev = pd.read_csv(y_path['dev'], sep='\t',
                    usecols=['label'])['label'].values

use_cols = ['bert_cosine_distance',
            '2_grams_jaccard',
            '3_grams_jaccard',
            '4_grams_jaccard',
            'predictions_raw',
            'predictions',
            'shortest_path_distance',
            'wm_distance'
            ]

X_test = pd.read_csv(X_path['test'], usecols=use_cols)
X_train = pd.read_csv(X_path['train'], usecols=use_cols)
X_dev = pd.read_csv(X_path['dev'], usecols=use_cols)

# Replace infinity values
X_train.replace(np.inf, 10, inplace=True)
X_test.replace(np.inf, 10, inplace=True)
X_dev.replace(np.inf, 10, inplace=True)

model = LogisticRegression()
# fit the model
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
print(accuracy_score(y_test, y_pred_test))

y_pred_dev = model.predict(X_dev)
print(accuracy_score(y_dev, y_pred_dev))

pickle.dump(model, open("model_selected_features.pkl", "wb"))

X_test.head()
