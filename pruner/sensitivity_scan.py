"""
This module used to perform sensitivity scan
include function to export sensitivity to png and csv
check for more detail :https://github.com/IntelLabs/distiller/
"""

from pruner.levelpruner import level_prune
from collections import OrderedDict
import csv
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


def sensitivity_to_png(sensitivities, fname):
    """
    create plot for sensitivities
    :param sensitivities: (dict), sensitivities dictionary produced by 'perform_sensitivity analysis' function
    :param fname: (str), path and filename for the output image
    """
    plt.figure(figsize=(15, 10))
    for param_name, sensitivity in sensitivities.items():
        sense = [values[0] for sparsity, values in sensitivity.items()]
        sparsities = [sparsity for sparsity, values in sensitivity.items()]
        plt.plot(sparsities, sense, label=param_name)
    plt.ylabel('accuracy')
    plt.xlabel('sparsity')
    plt.title('Pruning Sensitivity')
    plt.legend(loc='lower center', ncol=2, mode='expand', borderaxespad=0.)
    plt.savefig(fname, format='png')


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
        for param_name, sensitivity in sensitivities.items():
            for sparsity, values in sensitivity.items():
                writer.writerow([param_name] + [sparsity] + list(values))


def perform_sensitivity_analysis(model, val_loader, sparsities, test_func, criterion):
    """
    perform a sensitivity test for a model's weights parameters.
    :param model: torch.nn.Module, the model should be trained to maximum acuracy
    :param val_loader: dataloader, dataloader for the validation dataset
    :param sparsities: array, sparsity level
    :param test_func: function to execute the model on a test/validation dataset and return the accuracy and the loss value
    :param criterion: loss function
    :return:
        dict, {'param_name':{sparsity(float): (accuracy(float), loss(float))}}
    """
    sensitivities = OrderedDict()
    for _, (param_name, param) in enumerate(model.named_parameters()):
        if model.state_dict()[param_name].dim() not in [2, 4]:
            continue
        # make a copy of the model, because when we apply the zeros mask
        param_clone = param.data.clone()
        sensitivity = OrderedDict()

        for sparsity_level in sparsities:
            sparsity_level = float(sparsity_level)
            # Create the pruner
            # Element-wise sparsity
            level_prune(param=param.data, sparsity=sparsity_level)
            # Test and record the performance of the pruned model
            acc, loss = test_func(val_loader, model, criterion)
            acc = acc.cpu().numpy()
            param.data.copy_(param_clone)
            sensitivity[sparsity_level] = (acc, loss)
            sensitivities[param_name] = sensitivity
    return sensitivities
