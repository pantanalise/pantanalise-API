from functools import reduce
from os.path import dirname, join

import torch
import numpy as np
from transformers import AutoModelForPreTraining, AutoTokenizer

from pantanalise.repository.model.lia_bert_classifier import LIABertClassifier
from pantanalise.repository.predict import Predict

from transformers import BertTokenizer, BertConfig, BertForTokenClassification



class SentimentAnalysisDirectRepository(Predict):

    def predict(self, text):
        result = {}
        model = BertForTokenClassification.from_pretrained(
            "neuralmind/bert-base-portuguese-cased",
            num_labels=6,
            output_attentions=False,
            output_hidden_states=False
        )
        model.load_state_dict(torch.load(join(dirname(__file__), 'token_classification.pth')))
        model.cuda()
        model.eval()
        tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)
        tokenized_sentence = tokenizer.encode(text)
        input_ids = torch.tensor([tokenized_sentence]).cuda()
        with torch.no_grad():
            output = model(input_ids)
        label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
        tag_values = ['I-neg', 'B-neg', 'O', 'B-pos', 'I-pos', 'PAD']
        tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
        new_tokens, new_labels = [], []
        for token, label_idx in zip(tokens, label_indices[0]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(tag_values[label_idx])
                new_tokens.append(token)
                for token, label in zip(new_tokens, new_labels):
                    print("{}\t{}".format(label, token))
                    if token != '[CLS]' and token != '[SEP]'and token != 'PAD':
                        if label == 'I-neg' or label == 'B-neg':
                            result.update({token: 'negative'})
                        elif label == 'I-pos' or label == 'B-pos':
                            result.update({token: 'positive'})
                        elif label == 'O':
                            result.update({token: 'neutral'})
        return result

