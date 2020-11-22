"""
this module used to analyse optimal bits for quantization
"""

from quantizer.quantizer import apply_weight_sharing
from collections import OrderedDict
import torch
import csv
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def perform_bits_analysis(model, val_loader, bits, test_func, criterion):
    """
    :param model: torch.nn.Module, the model should be trained to maximum acuracy
    :param val_loader: dataloader, dataloader for the validation dataset
    :param bits: array, bit quantize
    :param test_func: function to execute the model on a test/validation dataset and return the accuracy and the loss 
                        value 
    :param criterion: loss function
    :return:
        dict, 'param_name':bit(float): (accuracy(float), loss(float))
    """
    sensitivities = OrderedDict()
    for _, (param_name, param) in enumerate(model.named_parameters()):
        if model.state_dict()[param_name].dim() not in [2, 4]:
            continue
        param_clone = param.data.clone()
        sensitivity = OrderedDict()
        codebook = dict()
        for bit_quantize in bits:
            cluster = apply_weight_sharing(param=param.data, bits=bit_quantize, codebook=codebook)
            param_quantize = cluster.cluster_centers_[cluster.labels_]
            param_quantize = torch.from_numpy(param_quantize).float().view(param.size())
            if not param.is_contiguous():
                param_quantize = param_quantize.contiguous()
            param.set_(param_quantize.to(param.device))
            acc, loss = test_func(val_loader, model, criterion)
            acc = acc.cpu().numpy()
            param.data.copy_(param_clone)
            sensitivity[bit_quantize] = (acc, loss)
            sensitivities[param_name] = sensitivity
    return sensitivities


def sensitivities_to_png(sensitivities, fname):
    """
    create plot for bit analysis results
    :param sensitivities: (dict), results from perform_bits_analysis
    :param fname: (str), path and filename for the output image
    :return:
        void
    """
    plt.figure(figsize=(15, 10))
    for param_name, sensitivity in sensitivities.items():
        sense = [values[0] for bit, values in sensitivity.items()]
        bits = [bit for bit, value in sensitivity.items()]
        plt.plot(bits, sense, label=param_name)
    plt.ylabel('accuracy')
    plt.xlabel('bit')
    plt.title('Quantize sensitivity')
    plt.legend(loc='lower center', ncol=2, mode='expand', borderaxespad=0.)
    plt.savefig(fname, format('png'))


def sensitivities_to_csv(sensitivities, fname):
    """
    Create a csv listing from the sensitivities dictionary.
    :param sensitivities: (dict), sensitivities dictionary produced by 'perform_sensitivity analysis' function,
    :param fname: (str), path and filename for the output csv
    """
    with open(fname, 'w') as csv_file:
        writer = csv.writer(csv_file)
        # write the header
        writer.writerow(['parameter', 'sparsity', 'accuracy', 'loss'])
        for param_name, bit in sensitivities.items():
            for bit, values in bit.items():
                writer.writerow([param_name] + [bit] + list(values))
