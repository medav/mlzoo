import numpy
import torch
from .. import model

cfg = model.bert_base_conf(512)
net = model.BertSquad(cfg).half().cuda()

bs = 8

input_ids = torch.randint(0, cfg.vocab_size, (bs, 512)).cuda()
token_type_ids = torch.randint(0, 1, (bs, 512)).cuda()


with torch.cuda.profiler.profile():
    net(input_ids, token_type_ids)

torch.cuda.synchronize()
