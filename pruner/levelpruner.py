"""
This module used to prune model by level of sparsities
reference : https://github.com/synxlin/nn-compression
"""

import math
import torch


def level_prune(param, sparsity):
    """
    element-wise pruning \n
    :param param: torch.(cuda.)Tensor, weight of conv/fc layer \n
    :param sparsity: float, pruning sparsity level, ex : 0.5 = 50% \n
    :return:
        torch.(cuda.).ByteTensor, mask for zeros
    """
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1:
        return torch.zeros_like(param).bool()
    num_el = param.numel()
    importance = param.abs()  # the absolute value of the weights
    num_stayed = num_el - int(math.ceil(num_el * sparsity))  # desired sparsity level
    topk, _ = torch.topk(importance.view(num_el), num_stayed, largest=True, sorted=False)
    thresh = topk[topk.argmin()]  # the smallest element in topk
    mask = torch.lt(importance.view(num_el), thresh).type(param.type()).bool()
    param.masked_fill_(mask, 0)
    return mask


class LevelPruner(object):
    def __init__(self, options):
        """
        Pruner class \n
        :param options: list of tuple, [(param_name(str), sparsity(float)]
        """
        self.options = options
        self.masks = dict()
        print("=" * 100)
        print("Initializing Pruner with options:")
        for r in self.options:
            print(r)

    def load_state_dict(self, state_dict, replace_options=True):
        """
        Load State Dict Pruner \n
        :param state_dict: dict, a dictionary containing a whole state of the pruner \n
        :param replace_options: bool, whether to use options settings in 'state_dict' \n
        :return: Pruner
        """
        if replace_options:
            self.options = state_dict['options']

        self.masks = state_dict['masks']
        print('=' * 90)
        print("Loading pruner with options:")
        for opt in self.options:
            print(opt)
        print("=" * 90)

    def state_dict(self):
        """
        returns a dictionary containing a whole state of the Pruner \n
        :return: dict, a dictionary containing a whole state of the Pruner
        """
        state_dict = dict()
        state_dict['options'] = [opt for opt in self.options]
        state_dict['masks'] = self.masks
        return state_dict

    def prune_param(self, param, param_name):
        """
        prune parameter \n
        :param param: torch.(cuda.)Tensor \n
        :param param_name: str, name of param \n
        :return:
            torch.(cuda.), mask for zeros
        """
        opt_index = -1
        for index, opt in enumerate(self.options):
            if opt[0] == param_name:
                opt_index = index
                break
        if opt_index > -1:
            sparsity = self.options[opt_index][1]
            mask = level_prune(param=param, sparsity=sparsity)
            return mask
        else:
            return None

    def prune(self, model, update_masks=False):
        """
        prune models \n
        :param model: torch.nn.Module \n
        :param update_masks: bool, whether update masks \n
        :return:
            void
        """
        update_masks = True if len(self.masks) == 0 or update_masks else False

        for param_name, param in model.named_parameters():
            if 'AuxLogits' in param_name or param.dim() <= 1:
                continue
            if update_masks:
                mask = self.prune_param(param=param.data, param_name=param_name)
                if mask is not None:
                    self.masks[param_name] = mask
            else:
                if param_name in self.masks:
                    mask = self.masks[param_name]
                    param.data.masked_fill_(mask, 0)
