import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import AutoModelForPreTraining
from typing import Optional, Tuple

class LIABertClassifier(nn.Module):
    def __init__(self,model,num_labels):
        super(LIABertClassifier,self).__init__()
        self.bert = model.bert
        self.config = model.config
        self.num_labels = num_labels
        self.cls = nn.Linear(self.config.hidden_size,num_labels)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        ) ->Tuple[torch.Tensor]:

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0][:,0,:]
        prediction = self.cls(sequence_output)
        return prediction