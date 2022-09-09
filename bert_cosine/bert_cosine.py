from constants import train_path, test_path, dev_path
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoModel, AutoTokenizer

model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print('Model', device)


def bert_cosine(csv_in_path, csv_out_path):

    print(csv_in_path)

    # read df
    df = pd.read_csv(csv_in_path, sep='\t')
    df_cos = pd.DataFrame(index=range(len(df)), columns=['id', 'bert_cosine_distance'])

    for i in range(len(df)):

        # print every 1000th iteration
        if i % 1000 == 0:
            print(i)

        current_row = df.iloc[i]
        sentence1, sentence2 = current_row['sentence1'], current_row['sentence2']

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

        # fill in table
        df_cos.loc[i, 'id'] = current_row['id']
        df_cos.loc[i, 'bert_cosine_distance'] = distance

    # save df
    df_cos.to_csv(csv_out_path, index=False)
    # print info
    print('Saved', len(df_cos), 'rows to', csv_out_path)


bert_cosine(train_path, 'distances/train_bert_cosine.csv')
bert_cosine(test_path, 'distances/test_bert_cosine.csv')
bert_cosine(dev_path, 'distances/dev_bert_cosine.csv')
