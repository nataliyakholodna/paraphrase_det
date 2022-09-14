### Intro
Paraphrasing is the process of rewriting text to change words and their order while preserving its original meaning. 

Paraphrased plagiarism is one of the complex issues facing plagiarism detection systems. Most plagiarism detection systems are designed to detect common words and minor changes but are unable to detect major semantic and structural changes. Therefore, many cases of plagiarism using paraphrasing stay unnoticed.

This project approaches paraphrase classification task as binary classification with label ```1``` denoting a pair of paraphrases and ```0``` otherwise.

In this project I try to investigate:
1) How well a fine-tuned Transformer model can distinguish paraphrases and non-paraphrases
2) Whether it is possible to obtain high accuracy of paraphrase detection without using deep neural networks
3) The weight and impact of generated features on the result of classification

# Dataset

[The Paraphrase Adversaries from Word Scrambling dataset](https://github.com/google-research-datasets/paws), part of the PAWS-Wiki Labeled (Final), was chosen to train and test the machine learning model, as well as to construct the features. The PAWS-Wiki contains 65 401 pairs of sentences, 44.2% of which are paraphrases of each other.

# Features

Because different studies use different methodologies (from calculating the number of common n-grams to using deep machine learning methods) to determine semantic similarity and paraphrases, different semantic measurement algorithms need to be compared and/or combined to maximize classification accuracy.
The following semantic similarity metrics or indicators were chosen for this study: Jaccard index for common n-grams, cosine distance between vector representations of sentences, Word Mover's Distance, distances by WordNet dictionaries (Leacock and Chodorow, Wu and Palmer), predictions made by a RoBERTa neural network.

### Jaccard index for common n-grams
N-gram is a sequence of n words. In the context paraphrase detection in the text, the number of common n-grams divided by the total number of n-grams in both sentences can help identify semantically similar sentences that are close in their meaning but are not paraphrases because the second sentence was obtained by permutation of words in the first sentence, and, accordingly, has a very different meaning.

![](https://github.com/nataliyakholodna/paraphrase_det/blob/main/images/formulas/jaccard.png)

To calculate n-grams in each sentence, the sentences are first lowercased, all punctuation marks and extra characters are removed. In total, Jaccard index was calculated for 2-, 3-, 4-grams.

### Cosine distance between vector representations of sentences

There is a great variety of methods for obtaining vector embeddings such as GloVe, Word2Vec (CBOW / Skip-Gram), fastText etc. Here, vector embeddings of the BERT deep learning model are used.


BERT is a Transformer deep learning model created for NLP tasks. The basic BERT model has 110 million customizable parameters, the large version has 345 million.
A peculiarity of the Transformer architecture is a mechanism of attention, which allows the data to be processed simultaneously (as opposed to recurrent neural networks, where data is processed sequentially). In addition, another peculiarity of BERT is the pre-training for solving two tasks: to predict a word in a sentence and to determine whether the second sentence is a logical continuation of the first one. The machine learning model was previously trained on unlabelled data from BooksCorpus (800 million words) and English Wikipedia (2,500 million words). Pretraining and attention mechanism allow to obtain contextualised word embeddings after the forward pass.

Using a pre-learned model, the vector representation for each sentence has the following dimensions: torch.Size ([1, 128, 768]), where 1 is the number of sentences in the batch, 128 is the maximum sentence length, 768 is the dimension of the word embedding vector. To get the a sentence embeddings the data is averaged for each word in the sentence, resulting in an embedding vector with a length of 768 values.
Cosine distance is calculated as follows:

![](https://github.com/nataliyakholodna/paraphrase_det/blob/main/images/formulas/cosine_sim.png)


### Word Mover's Distance

Word Mover's Distance uses vector embeddings to calculate the semantic distance between sentences. WMD distance measures the difference between two text documents as the minimum distance that word embedding vector of one document must "come" to reach the embedding vector of another document.

### Distances by WordNet dictionaries

To measure the semantic similarity of sentences, two metrics of semantic distance of synsets from WordNet dictionaries were used: Leacock and Chodorow, Wu and Palmer.
The semantic similarity of words according to Leacock and Chodorow is determined by the following formula:

![](https://github.com/nataliyakholodna/paraphrase_det/blob/main/images/formulas/leacock_chodorow.png)

where length is the length of the shortest path between two concepts (number of nodes), D is the maximum depth of the corresponding taxonomy.
The function of the semantic distance Wu and Palmer depends on the depth of the two concepts ```depth(concept_i)```  in the taxonomy and the depth of their nearest common ancestor depth (LCS):

![](https://github.com/nataliyakholodna/paraphrase_det/blob/main/images/formulas/wu_parmer.png)

Because WordNet calculates the distance for each pair of synsets separately, the approach proposed by researchers Courtney Corley and Rada Mihalcea was used to represent the distance between two sentences. To introduce bidirectionality, the arithmetic mean of two values of a certain distance is used, the value of which depends on the maximum similarity of the word pair. However, in this case the specificity of words is not taken into account (there is no multiplication by the value of tf-idf). Thus, to calculate the distances between a pair of sentences, one has to compare each word of the first sentence with each word of the second one.


### RoBERTa neural network

RoBERTa is a pre-trained neural network that can be considered as an improved analogue of BERT: its main difference is the selection of hyperparameters during pre-training and bigger training corpus volume (16GB sentences from Books Corpus and English Wikipedia, CommonCrawl News dataset (63 million articles, 76 GB) , Web text corpus (38 GB), Stories from Common Crawl (31 GB)), the use of dynamic token masking for word prediction in a sentence: the word to be predicted in a particular sentence changes with each epoch.
RoBERTa has 124 million customizable parameters. Because the result of direct propagation is a matrix of vector embeddings of the size (batch size, max sentence length, embedding dimension), a hidden and fully connected output layers with 256 and 1 neurons, respectively, were added for further classification. Similar to the feature "Cosine distance…", to obtain a vector representation of sentences, the average value was obtained for each dimention of the embedding vector. The resulting matrix of weights between the obtained vector representations of words and the hidden layer with 256 neurons will have a dimension of 768 * 256, between the hidden and the original layer - 256 * 1. The activation function of the last layer is the sigmoid, the loss function is binary cross-entropy.

The pre-trained neural network was fine-tuned to detect paraphrases in the text using training and validation samples. In total, the neural network was additionally trained for 50 epochs, the weights were preserved at the lowest value of the loss function in the validation sample (last epoch).

![](https://github.com/nataliyakholodna/paraphrase_det/blob/main/images/formulas/roberta.png)

For the final combination of features, both the probabilities that two records belong to a certain class and the predictions themselves were chosen. Decision threshold = 0.5, i.e., y ̂ = 1 if p(y ̂ )>0.5.
The final table contains 10 features with the above metrics for each pair of sentences available in the selected dataset. These features were calculated for the training, test and validation samples.

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


![](https://github.com/nataliyakholodna/paraphrase_det/blob/main/images/class_report_3.png)
![](https://github.com/nataliyakholodna/paraphrase_det/blob/main/images/matrix_3.png)

Smaller recall of the model may lead to incorrect allegations of plagiarism or incorrect aggregation of user-generated content.

Transformer neural networks do not require additional generation of features and are able to detect paraphrasing with high accuracy. The disadvantage of this type of networks is a significant number of parameters (and, accordingly, a long time to calculate the results).







Links:
* [RoBERTA paraphrase detection via Flask API](https://github.com/nataliyakholodna/roberta_paraphrase_detection)
* [Project's presentation](https://www.canva.com/design/DAE4zcIOvwo/XAH-wxQl0wnIiPhywJEjgw/view?utm_content=DAE4zcIOvwo&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)