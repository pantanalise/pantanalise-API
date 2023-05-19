from os.path import dirname, join

from transformers import AutoTokenizer

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import joblib

import torch
import torch.nn as nn
from typing import Tuple

INPUT = "Além disso, considerou o Brasil um lider na promoção do esporte escolar no mundo após a volta dos Jogos Escolares Brasileiros (JEBs), passados mais de 15 anos sem sua realização."
MODEL_PATH = join(dirname(__file__), "regressor_model_0.pth")
SCALER_PATH= join(dirname(__file__), "fitted_std_scaler.save")


tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('rslp')

def clean_tweet(tweet):
    # remove links
    tweet = re.sub(r'http(\S)+', '', tweet)
    # remove pontuação
    tweet = re.sub(r'[^\w\s]', '', tweet)
    # converte para minúsculas
    tweet = tweet.lower()
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    tweet = emoji_pattern.sub(r'', tweet)
    # remove stop words em português
    stop_words = set(stopwords.words('portuguese'))
    words = nltk.word_tokenize(tweet)
    words = [word for word in words if not word in stop_words]
    # aplica stemização
    stemmer = RSLPStemmer()
    words = [stemmer.stem(word) for word in words]
    # junta as palavras novamente
    tweet = ' '.join(words)
    tweet = tokenizer.tokenize(tweet)
    return tweet

def predict(data):
    print(MODEL_PATH)
    print(SCALER_PATH)
    tweet = clean_tweet(INPUT)
    tweet = tokenizer(tweet, truncation=True, padding=True, max_length=512, is_split_into_words=True,
                       return_tensors='pt')
    regressor_model = torch.load(MODEL_PATH)
    #
    # device = torch.device("cpu")
    #
    # regressor_model.to(device)
    #
    # regressor_model.eval()
    # output = regressor_model(tweet['input_ids'], tweet['attention_mask'])
    # engagement_scaler = joblib.load(SCALER_PATH)
    # output = engagement_scaler.inverse_transform(output.detach().numpy().reshape(-1, 1))
    # print(f'likes:{int(output[0][0] * 8 / 9)}')
    # print(f'Retweets:{int(output[0][0] / 9)}')

class BertEngagementRegressor(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.bert = model.bert
        self.config = model.config
        self.linear = nn.Linear(self.config.hidden_size,200)
        self.dropout = nn.Dropout(p=0.1)
        self.linear2 = nn.Linear(200,1)
        self.double()

    def forward(self, input_ids, attention_masks) ->Tuple[torch.Tensor]:
        output = self.bert(input_ids, attention_masks)[1]
        output = self.linear(output)
        output = self.dropout(output)
        output = self.linear2(output)
        return output.squeeze()