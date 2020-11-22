import sys

sys.path.append('/Users/user/Documents/PRA-SKRIPSI/skripsi')
import torch
from pruner.levelpruner import level_prune

tensor = torch.tensor([1.1, 2.3, 9.1, 3.3, 5.2, 6.1, 7.1, 8.3, 2.1, 4.3])
print('tensor: ', tensor)
mask = level_prune(param=tensor, sparsity=0.7)
print('mask:', mask)
print('masked tensor:', tensor)

