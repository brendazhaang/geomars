import torch
import torch.nn as nn
from torchvision.models import (
    alexnet,
    vgg16_bn,
    resnet18,
    resnet34,
    resnet50,
    densenet121,
    densenet161,
)
import pytorch_lightning as pl
from torch.nn import functional as F
from torchmetrics.functional import accuracy, precision
from torchmetrics.functional.classification import multiclass_recall


class MarsModel(pl.LightningModule):
    def __init__(self, hyper_param):
        super().__init__()
        self.momentum = hyper_param["momentum"]
        self.optimizer = hyper_param["optimizer"]
        self.lr = hyper_param["learning_rate"]
        self.num_classes = hyper_param["num_classes"]
        self.validation_step_outputs = []
        #self.validation_step_targets = []

        if hyper_param["model"] == "resnet18":
            """
            Resnet18
            """
            self.net = resnet18(pretrained=hyper_param["pretrained"])
            if hyper_param["transfer_learning"] is True:
                self.set_parameter_requires_grad(self.net)

            num_ftrs = self.net.fc.in_features
            self.net.fc = nn.Linear(num_ftrs, hyper_param["num_classes"])

        elif hyper_param["model"] == "resnet34":
            """
            Resnet34
            """
            self.net = resnet34(pretrained=hyper_param["pretrained"])
            if hyper_param["transfer_learning"] is True:
                self.set_parameter_requires_grad(self.net)
            num_ftrs = self.net.fc.in_features
            self.net.fc = nn.Linear(num_ftrs, hyper_param["num_classes"])

        elif hyper_param["model"] == "resnet50":
            """
            Resnet50
            """
            self.net = resnet50(pretrained=hyper_param["pretrained"])
            if hyper_param["transfer_learning"] is True:
                self.set_parameter_requires_grad(self.net)
            num_ftrs = self.net.fc.in_features
            self.net.fc = nn.Linear(num_ftrs, hyper_param["num_classes"])

        elif hyper_param["model"] == "alexnet":
            """
            Alexnet
            """
            self.net = alexnet(pretrained=hyper_param["pretrained"])
            if hyper_param["transfer_learning"] is True:
                self.set_parameter_requires_grad(self.net)
            num_ftrs = self.net.classifier[6].in_features
            self.net.classifier[6] = nn.Linear(num_ftrs, hyper_param["num_classes"])

        elif hyper_param["model"] == "vgg16":
            """
            VGG16_bn
            """
            self.net = vgg16_bn(pretrained=hyper_param["pretrained"])
            if hyper_param["transfer_learning"] is True:
                self.set_parameter_requires_grad(self.net)
            num_ftrs = self.net.classifier[6].in_features
            self.net.classifier[6] = nn.Linear(num_ftrs, hyper_param["num_classes"])

        elif hyper_param["model"] == "densenet121":
            """
            Densenet-121
            """
            self.net = densenet121(pretrained=hyper_param["pretrained"])
            if hyper_param["transfer_learning"] is True:
                self.set_parameter_requires_grad(self.net)
            num_ftrs = self.net.classifier.in_features
            self.net.classifier = nn.Linear(num_ftrs, hyper_param["num_classes"])

        elif hyper_param["model"] == "densenet161":
            """
            Densenet-161
            """
            self.net = densenet161(pretrained=hyper_param["pretrained"])
            if hyper_param["transfer_learning"] is True:
                self.set_parameter_requires_grad(self.net)
            num_ftrs = self.net.classifier.in_features
            self.net.classifier = nn.Linear(num_ftrs, hyper_param["num_classes"])

        else:
            print("Invalid model name, exiting...")
            exit()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(torch.argmax(y_hat, dim=1), y, num_classes=self.num_classes, task="multiclass")
        prec = precision(F.softmax(y_hat, dim=1), y, average='none', num_classes=self.num_classes, task="multiclass")
        recall = multiclass_recall(F.softmax(y_hat, dim=1), y, average='none', num_classes=self.num_classes)

        self.validation_step_outputs.append(loss)
        self.validation_step_outputs.append(acc)
        return {
            "val_loss": loss,
            "val_acc": acc,
            "val_prec": prec,
            "val_recall": recall,
        }

    #def validation_epoch_end(self, outputs):
       # avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
       # avg_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
       # return {
        #    "val_loss": avg_loss,
        #    "progress_bar": {"val_loss": avg_loss, "val_acc": avg_acc},
       # }
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.stack([x["val_acc"] for x in self.validation_step_targets]).mean()
        self.validation_step_outputs.clear()
        #self.validation_step_targets.clear()
        return {
            "val_loss": avg_loss,
            "progress_bar": {"val_loss": avg_loss, "val_acc": avg_acc},
        }
       
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        logs = {"test_loss": avg_loss}
        return {"test_loss": avg_loss, "log": logs}

    def configure_optimizers(self):
        params_to_update = []
        print("Params to learn:")
        for name, param in self.net.named_parameters():
            if param.requires_grad is True:
                params_to_update.append(param)
                print("\t", name)

        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(params_to_update, lr=self.lr)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                params_to_update, lr=self.lr, momentum=self.momentum
            )
        else:
            print("Invalid optimizer, exiting...")
            exit()

        return optimizer

    def set_parameter_requires_grad(model):
        for param in model.parameters():
            param.requires_grad = False
