import torch
from .. import model
from ..ref import load_weights

net = model.BertSquad(model.bert_large_conf(512))
net.load_from_file('./bert-large-squad.npz')

text = """
Depending on the degree of generality and programmability,
hardware accelerators can be designed differently and thus require
different mapping strategies. We divide the existing mapping flows
for spatial accelerators into two categories: hardware-aware and
ISA-aware. For specific domains, the hardware design and software
mapping can be coupled together by providing a domain-specific
hardware and software interface [28, 33, 47, 62]. More clearly, the
compiler/mapper is aware of the hardware architecture details
such as the number of processing elements (PEs) and their interconnection, and then the mapping can be formulated as optimization problems with respect to hardware constraints [24, 39, 66]. We
call this hardware-aware mapping. This approach achieves very
high energy efficiency for specific application domains but sacrifices flexibility. For ISA-aware mapping, the hardware accelerators
are programmable with ISA, which separates the algorithmic specification from hardware architectural details. These instructions are
often exposed as special intrinsics and using these intrinsics for
tensor computation is called tensorization [9]. We focus on the
ISA-aware mapping problem in this paper.
While intrinsics provide programmability, ISA-aware mapping
is still a challenging task mainly for two reasons. First, there are
different ways to compose a mapping using intrinsics. For example,
we find that there are 35 different ways to map the 7 loops of a 2D
convolution to the 3 dimensions of Tensor Core. The quality of the
mapping is apparently critical to the performance as different mappings vary substantially by affecting data locality and parallelism.
But existing compilers [9, 10, 52, 58, 67] heavily rely on manual programming with intrinsics to develop libraries or templates, which
may miss the optimal mapping choice. Second, different accelerators provide different intrinsics with intricate compute and memory
semantics. For example, Tensor Core uses different intrinsics to
describe matrix load/store, matrix multiplication, and initialization
semantics, while for Mali GPU, a single arm_dot intrinsic can work
without other explicit load/store intrinsics. Therefore, to support a
single algorithm on different accelerators, the programmers practically need to implement and tune the algorithm for each target
platform individually. Apparently, a desirable mapping solution
is to use a unified approach to expose a search space of feasible
mappings that can be explored automatically
"""

question = 'What is the focus of this paper?'
print(f'Question: {question}')

o = net.query(text, question)


print()
print('My best guess:')
print(o)
print()

