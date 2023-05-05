from functools import reduce
from os.path import dirname, join

import torch
from transformers import AutoModelForPreTraining, AutoTokenizer

from pantanalise.model.lia_bert_classifier import LIABertClassifier


class SentimentAnalysisRepository:
    def __init__(self):
        WEIGHTS_PATH = join(dirname(__file__), "model.pth")
        model_base = AutoModelForPreTraining.from_pretrained('neuralmind/bert-base-portuguese-cased')
        self.model = LIABertClassifier(model=model_base, num_labels=3)
        state_dict = torch.load(WEIGHTS_PATH)
        self.model.eval()



    def predict(self, text):
        tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)
        token = tokenizer.tokenize(text, truncation=True, padding=True, max_length=512, return_tensors='pt')
        ids = tokenizer.convert_tokens_to_ids(token)
        tensor = torch.tensor([ids])
        with torch.no_grad():
            output = self.model(tensor)
        sentiments_predict = []
        sentiments = [ 'positivo', 'negativo', 'neutro']
        for key, sentiment in enumerate(output[0]):
           sentiments_predict.append({ 'feeling': sentiments[key] , 'strength':sentiment.item()})

        max_feeling = reduce(
            lambda x, y: x if x['strength'] > y['strength'] else y,
            sentiments_predict
        )
        return max_feeling['feeling']



