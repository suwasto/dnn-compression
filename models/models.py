"""
This module contains list of the model used for the experiment
    and also the code for training and validate the model
models : resnet, alexnet, vgg, squeezenet, densenet, inception
detailed explanation can be read at : https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
"""

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from torchvision import models

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_parameter_requires_grad(model, feature_extracting):
    """
    :param model : torch.nn.Module
    :param feature_extracting: bool, set True for finetuning and False for training from scratch
    :return:
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    """
    :param model_name: str, pick one from this list
                    [resnet, alexnet, vgg, squeezenet, densenet, inception]
    :param num_classes: int, num of classes in the dataset
    :param feature_extract: bool, set True for finetuning and False for training from scratch
    :param use_pretrained: bool, whether using pretrained model or not. default=True
    :return:
        model_ft : torch.nn.Module
        input_size: int
    """
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """
        Resnet
        the last layer is a fully connected layer as shown as below:
        (fc): Linear(in_features=512, out_features=1000, bias=True)
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """
        Alexnet
        the model output comes from 6th layer of the classifier
        (classifier): Sequential(
            ...
            (6):Linear(in_features=4096, out_features=1000, bias=True)
        )
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """
        VGG11_bn
        the output layer is similar to Alexnet
        (classifier): Sequential(
            ...
            (6):Linear(in_features=4096, out_features=1000, bias=True)
        )
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """
        Squeezenet1_0
        the output comes from a 1x1 convolutional layer which is the 1st layer
        of the classifier:
        (classifier): Sequential(
            (0): Dropout(p=0.5)
            (1): Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1))
            (2): ReLU(inplace)
            (3): AvgPool2d(kernel_size=13, stride=1, padding=0)
        )
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """
        Densenet-121
        The output layer is a linear layer with 1024 input features:
        (classifier): Linear(in_features=1024, out_features=1000, bias=True)
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """
        Inception v3
        Be careful, expects (299, 299) sized images and has auxilary output
        (AuxLogits): InceptionAux(
            ...
            (fc): Linear(in_features=768, out_features=1000, bias=True)
        )
        ...
        (fc): Linear(in_features=2048, out_features=1000, bias=True)
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size
