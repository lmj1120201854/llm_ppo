import torch
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained('/home/mjli/projects/models/bert-base')
print(model.d_embed)