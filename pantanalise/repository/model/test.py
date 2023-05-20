import torch
import torch.nn as nn
from typing import Tuple

class BertEngagementRegressor(nn.Module):
    def _init_(self, model):
        super()._init_()
        self.bert = model.bert
        self.config = model.config
        self.linear = nn.Linear(self.config.hidden_size, 200)
        self.dropout = nn.Dropout(p=0.1)
        self.linear2 = nn.Linear(200, 1)
        self.double()

    def forward(self, input_ids, attention_masks) -> Tuple[torch.Tensor]:
        output = self.bert(input_ids, attention_masks)[1]
        output = self.linear(output)
        output = self.dropout(output)
        output = self.linear2(output)
        return output.squeeze()