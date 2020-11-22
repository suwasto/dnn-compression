""" A collection of useful utility functions to train and validate the models
"""

import time
import copy
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def validate(val_loader, model, criterion):
    """
    validate the model through validation dataset
    :param val_loader: DataLoader, validation dataset
    :param model: torch.nn.Module, trained model
    :param criterion: loss function
    """
    # switch to evaluate mode
    model.eval()

    running_loss = 0.0
    running_corrects = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            labels = labels.to(device)
            inputs = inputs.to(device)
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        total_loss = running_loss / len(val_loader.dataset)
        total_acc = running_corrects.double() / len(val_loader.dataset)
    return total_acc, total_loss


def train_model(model, train, val, criterion, optimizer, num_epochs=25, is_inception=False, pruner=None, quantizer=None,
                verbose=False):
    """
    Train and validate the model

    :param model: torch.nn.Module
    :param train: train dataloader
    :param val: val dataloader
    :param criterion:
    :param optimizer:
    :param num_epochs: int, number of epochs (default = 25)
    :param is_inception: bool, whether model is inception (default=False)
    :param pruner: pruner function, default = None
    :param quantizer: quantizer function, default = None
    :param verbose: bool, whether to print the pruner/quantizer
    :return:
        model : torch.nn.Module
        val_acc_history: []
    """

    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # set model to training mode
            else:
                model.eval()  # set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            dataloaders = train if phase == 'train' else val
            for inputs, labels in dataloaders:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in the training it has auxiliary output
                    # In train mode we calcuate the loss by summing the final output and the auxiliary output
                    # but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if pruner is not None:
                            pruner.prune(model=model, update_masks=False)
                        if quantizer is not None:
                            with torch.no_grad():
                                quantizer.quantize(model=model, update_labels=True, re_quantize=False)

                # statistic
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders.dataset)
            epoch_acc = running_corrects.double() / len(dataloaders.dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # loade best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def visualize_prediction(model, val, num_images=6):
    """
    function to visualize predictions
    :param model: torch.nn.Module, trained model
    :param val: validation dataloader
    :param num_images: int, number of images to plot (default=6)
    :return:
    """
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    fig.figsize(10, 8)
    with torch.no_grad():
        for _, (inputs, _) in enumerate(val):
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(preds[j]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def imshow(inp, title=None):
    """
    function to plot image
    :param inp: tensor input
    :param title: str, title of the plot
    """
    out = torchvision.utils.make_grid(inp)
    inp = out.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(10, 8))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


def density(tensor):
    """
    Computes the density(fraction of non-zero elements) of a tensor
    density 1.0 means that tensor has no non-zero elements
    :param tensor: tensor
    :return: density(float)
    """
    # nonzero = torch.nonzero(tensor)
    nonzero = tensor.abs().gt(0).sum()
    return float(nonzero.item()) / torch.numel(tensor)


def weights_sparsity_summary(model, param_dims=[2, 4]):
    df = pd.DataFrame(columns=[
        'Name', 'Shape', 'NNZ (dense)', 'NNZ (sparse)',
        'sparsity (%)', 'Std', 'Mean', 'Abs-Mean'
    ])
    pd.set_option('precision', 2)
    params_size = 0
    sparse_params_size = 0
    for name, param in model.named_parameters():
        if param.dim() in param_dims and any(n in name for n in ['weight', 'bias']):
            _density = density(param)
            params_size += param.numel()
            sparse_params_size += param.numel() * _density
            df.loc[len(df.index)] = ([
                name,
                param.size(),
                param.numel(),
                int(_density * param.numel()),
                (1 - _density) * 100,
                param.std().item(),
                param.mean().item(),
                param.abs().mean().item()
            ])
    total_sparsity = (1 - sparse_params_size / params_size) * 100
    df.loc[len(df.index)] = ([
        'Total sparsity:',
        '-',
        params_size,
        int(sparse_params_size),
        total_sparsity,
        0, 0, 0
    ])
    return df
