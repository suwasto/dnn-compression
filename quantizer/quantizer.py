"""
This module is used to Quantize model
Reference : https://github.com/synxlin/nn-compression
"""

import torch
import numpy as np
from sklearn.cluster import KMeans


def apply_weight_sharing(param, bits, codebook):
    """
    apply weight sharing to the parameter
    :param param: tensor, weight parameter of the model
    :param bits: int, number of bits for quantization
    :param codebook: dict, codebook for clustering param
    """
    num_el = param.numel()
    param_numpy = param.view(num_el).cpu().numpy()
    param_nz = param_numpy[param_numpy != 0]
    param_nz = param_nz.reshape(param_nz.size, 1)

    k = 2 ** bits
    if codebook is None:
        codebook = KMeans(n_clusters=k).fit(param_nz)
        centers = codebook.cluster_centers_
        centers = np.append(0.0, centers)
        codebook.cluster_centers_ = centers.reshape(centers.size, 1)
        codebook.labels_ = codebook.predict(param_numpy.reshape(num_el, 1).astype('float'))
    else:
        codebook.labels_ = codebook.predict(param_numpy.reshape(num_el, 1).astype('float'))

    return codebook


class Quantizer(object):

    def __init__(self, options):
        """
        Quantizer class for quantization
        :param options: param_name, bit_length list of tuple,
                            [(param_name(str),  bit_length(int))]
        """
        self.options = options
        self.codebooks = dict()

        print("=" * 90)
        print("Initializing Quantizer:")
        for opt in self.options:
            print(opt)

    def load_state_dict(self, state_dict):
        """
        Recover quantizer
        :param state_dict: dict, a dictionary containing a whole state of the Quantizer
        :return:
            Quantizer
        """
        self.options = state_dict['options']
        self.codebooks = dict()
        for name, codebook in state_dict['codebooks'].items():
            self.codebooks[name] = KMeans().set_params(**codebook['param'])
            self.codebooks[name].cluster_centers_ = codebook['centers']
            self.codebooks[name].labels_ = codebook['labels']
        print('=' * 90)
        print('Customizing Quantizer with options:')
        for r in self.options:
            print(r)
        print('=' * 90)

    def state_dict(self):
        """
        Returns a dictionary containing a whole state of the quantizer
        :return: dict, a dictionary containing a whole state of the Quantizer
        """
        state_dict = dict()
        state_dict['options'] = self.options
        codebooks = dict()
        for name, codebook in self.codebooks.items():
            if isinstance(codebook, KMeans):
                codebooks[name] = {
                    'param': codebook.get_params(),
                    'centers': codebook.cluster_centers_,
                    'labels': codebook.labels_
                }
        state_dict['codebooks'] = codebooks
        return state_dict

    def quantize(self, model):
        """
        quantize param
        :param model: torch.(cuda).tensor
        :return:
            dict, {'centers_': torch.tensor}
        """
        for param_name, param in model.named_parameters():
            opt_id = -1
            for idx, opt in enumerate(self.options):
                if opt[0] == param_name:
                    opt_id = idx
                    break

            if opt_id > -1 and param.dim > 1:
                bit_length = self.options[opt_id][1]
                codebook = self.codebooks.get(param_name)
                codebook = apply_weight_sharing(param, bit_length, codebook)
                param_quantize = codebook.cluster_centers_[codebook.labels_]
                param_quantize = torch.from_numpy(param_quantize).float().view(param.size())
                if not param.is_contiguous():
                    param_quantize = param_quantize.contiguous()
                param.set_(param_quantize)

                if codebook is not None:
                    self.codebooks[param_name] = codebook
            else:
                return None
