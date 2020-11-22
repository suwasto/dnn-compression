import sys

sys.path.append('/Users/user/Documents/PRA-SKRIPSI/skripsi')
import torch
from quantizer.quantizer import apply_weight_sharing

tensor = torch.Tensor([2.0, 5.3, 2.1, 3.2, 4.5, 3.2, 3.5, 3.2])
print('tensor: ', tensor.size())
codebook = None
codebook = apply_weight_sharing(tensor, 2, codebook)
print('codebook centroid :', codebook.cluster_centers_)
print('codebook labels:', codebook.labels_)
param_quantize = codebook.cluster_centers_[codebook.labels_]
param_quantize = torch.from_numpy(param_quantize).float().view(tensor.size())
print('param_quantize:', param_quantize)

numel = tensor.numel()
tensor = tensor.view(numel)
tensor_numpy = tensor.cpu().numpy()
print(tensor_numpy.reshape(numel, 1))
codebook.labels_ = codebook.predict(tensor_numpy.reshape(numel, 1).astype('float'))
print(codebook.labels_)