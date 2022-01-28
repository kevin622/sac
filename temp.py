from models import Policy as P1
from code_that_works.models_2 import Policy as P2
import torch

p1 = P1(3, 3)
p2 = P2(3, 3)
p2.load_state_dict(p1.state_dict())

print(p1.sample(torch.tensor([[1., 2., 3.]])))
print(p2.sample(torch.tensor([[1., 2., 3.]])))

import numpy as np
a = np.array([1,2])
