### Intro
Paraphrasing is the process of rewriting text to change words and their order while preserving its original meaning. 

Paraphrased plagiarism is one of the complex issues facing plagiarism detection systems. Most plagiarism detection systems are designed to detect common words and minor changes but are unable to detect major semantic and structural changes. Therefore, many cases of plagiarism using paraphrasing stay unnoticed.

In the problem of detecting paraphrases, the result is a probability from 0 to 1, where a value close to 1 means a couple of sentences are the paraphrases of each other, 0 – sentences have different meanings.

In this project I try to investigate:
1) How well a fine-tuned Transformer model can distinguish paraphrases and non-paraphrases
2) Whether it is possible to obtain high accuracy of classification without the use of deep neural networks
3) The weight and impact of generated features on the result of classification

### Dataset

The Paraphrase Adversaries from Word Scrambling dataset, part of the PAWS-Wiki Labeled (Final), was selected to test and test the machine learning model, as well as to construct the features. The PAWS-Wiki contains 65,401 pairs of sentences, 44.2% of which are paraphrases of each other.

### Features

Because different studies use different methodologies (from calculating the number of common n-grams to using deep machine learning methods) to determine semantic similarity and paraphrases, different semantic measurement algorithms need to be compared and/or combined to maximize classification accuracy.
The following semantic similarity metrics or indicators were chosen for this study: Jaccard index for common n-grams, cosine distance between vector representations of sentences, Word Mover's Distance, WordNet dictionaries (Leacock and Chodorow, Wu and Palmer), predictions made by a Transformer neural network - RoBERTa.

# Results
⏩ ```evaluate.py```

### All features

To combine the features into final classification, the algorithm of classical machine learning (logistic regression) was chosen.
The results obtained are as follows:
* Accuracy on the test data set - 92.5%, the area under the ROC-curve is 97.11%, under the Precision-Recall curve - 95.23%.
* Accuracy on the validation data set - 93.7% of the area under the ROC curve is 97.73%, under the Precision-Recall curve - 96.30%.

As it can be seen in the classification report and confusion matrix for the test set, logistic regression erroneously marked almost twice as many **negative records as positive** than positive as negative.

![Classification report](https://github.com/nataliyakholodna/paraphrase_det/blob/main/images/class_report_1.png)
![Confusion matrix](https://github.com/nataliyakholodna/paraphrase_det/blob/main/images/confusion_matrix_1.png)

### Without RoBERTa's prediction

* Accuracy of classification on the test data set - 71.15%, area under the Precision-Recall curve - 73.1%.
The most important features for classification are the Jaccard index for 3-grams (normalized number of common 3-grams) and the cosine distance between vector representations of sentences.

![Feature importance](https://github.com/nataliyakholodna/paraphrase_det/blob/main/images/feature_imp_2.png)

### RoBERTa classification results

* Accuracy on the test set: 91.96%
* Area under ROC-curve = 97.59%
* Area under Precision-Recall curve = 96.34%

Despite great accuracy, RoBERTa classifies more negative classes as positive, while **reducing the recall to pairs of sentences that are not paraphrases** of each other:

<img src="https://github.com/nataliyakholodna/paraphrase_det/blob/main/images/class_report_3.png">
![](https://github.com/nataliyakholodna/paraphrase_det/blob/main/images/matrix_3.png)

Smaller recall of the model may lead to incorrect allegations of plagiarism or incorrect aggregation of user-generated content.

Transformer neural networks do not require additional generation of features and are able to detect paraphrasing with high accuracy. The disadvantage of this type of networks is a significant number of parameters (and, accordingly, a long time to calculate the results).







Links:
* [RoBERTA paraphrase detection via Flask API](https://github.com/nataliyakholodna/roberta_paraphrase_detection)
* [Project's presentation](https://www.canva.com/design/DAE4zcIOvwo/XAH-wxQl0wnIiPhywJEjgw/view?utm_content=DAE4zcIOvwo&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)