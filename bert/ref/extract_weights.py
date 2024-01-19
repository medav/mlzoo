import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained(
    "bert-large-uncased-whole-word-masking-finetuned-squad")

for n, p in model.named_parameters(): print(n, p.shape)
for n, p in model.named_buffers(): print(n, p.shape)

params = {
    name: p.detach().numpy()
    for name, p in model.named_parameters()
}

params.update({
    name: b.detach().numpy()
    for name, b in model.named_buffers()
})

np.savez('bert-large-squad.npz', **params)
