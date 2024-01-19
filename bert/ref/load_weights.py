
import torch
import numpy as np

from .. import model

def make_torch_tensor(name, data):
    if 'weight' in name or 'bias' in name:
        return torch.nn.Parameter(torch.tensor(data))
    else: return torch.tensor(data)

def bert_load_from_weights(self : model.Bert, weights):
    self.embed.word_embed.weight = weights['bert.embeddings.word_embeddings.weight']
    self.embed.pos_embed.weight = weights['bert.embeddings.position_embeddings.weight']
    self.embed.seg_embed.weight = weights['bert.embeddings.token_type_embeddings.weight']
    self.embed.ln.weight = weights['bert.embeddings.LayerNorm.weight']
    self.embed.ln.bias = weights['bert.embeddings.LayerNorm.bias']

    def map_transformer_layer(dest, src_name_base):
        dest.query.weight = weights[f'{src_name_base}.attention.self.query.weight']
        dest.query.bias = weights[f'{src_name_base}.attention.self.query.bias']
        dest.key.weight = weights[f'{src_name_base}.attention.self.key.weight']
        dest.key.bias = weights[f'{src_name_base}.attention.self.key.bias']
        dest.value.weight = weights[f'{src_name_base}.attention.self.value.weight']
        dest.value.bias = weights[f'{src_name_base}.attention.self.value.bias']
        dest.y0w0.weight = weights[f'{src_name_base}.attention.output.dense.weight']
        dest.y0w0.bias = weights[f'{src_name_base}.attention.output.dense.bias']
        dest.ln0.weight = weights[f'{src_name_base}.attention.output.LayerNorm.weight']
        dest.ln0.bias = weights[f'{src_name_base}.attention.output.LayerNorm.bias']
        dest.y1w1.weight = weights[f'{src_name_base}.intermediate.dense.weight']
        dest.y1w1.bias = weights[f'{src_name_base}.intermediate.dense.bias']
        dest.y2w2.weight = weights[f'{src_name_base}.output.dense.weight']
        dest.y2w2.bias = weights[f'{src_name_base}.output.dense.bias']
        dest.ln2.weight = weights[f'{src_name_base}.output.LayerNorm.weight']
        dest.ln2.bias = weights[f'{src_name_base}.output.LayerNorm.bias']

    for i in range(len(self.layers)):
        map_transformer_layer(self.layers[i], f'bert.encoder.layer.{i}')

model.Bert.load_from_weights = bert_load_from_weights

def bertsquad_load_from_weights(self : model.BertSquad, weights):
    self.bert.load_from_weights(weights)
    self.qa.weight = weights['qa_outputs.weight']
    self.qa.bias = weights['qa_outputs.bias']

model.BertSquad.load_from_weights = bertsquad_load_from_weights

def bertmodel_load_from_file(self : model.Bert, filename):
    weights = {
        name: make_torch_tensor(name, data)
        for name, data in np.load(filename).items()
    }
    self.load_from_weights(weights)

model.Bert.load_from_file = bertmodel_load_from_file
model.BertSquad.load_from_file = bertmodel_load_from_file
