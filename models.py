# Modeling
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision

from collections import OrderedDict

#ResNet
class ModifiedResNet(nn.Module):
    def __init__(self, output_classes, disentangle=False):
        super().__init__()
        backbone = torchvision.models.resnet50(pretrained=True)
        modules = list(backbone.children())[:-1]  # remove final layer
        self.backbone = nn.Sequential(*modules)
        self.fc = nn.Linear(2048, output_classes) # Magic number 2048 is from looking at resnet arch
        self.disentangle = disentangle
    def forward(self, x):
        rep = self.backbone(x)
        rep = torch.flatten(rep, 1)
        out = self.fc(rep)
        return out, rep


# SimCLR
class Flatten(nn.Module):
    def __init__(self, dim=-1):
        super(Flatten, self).__init__()
        self.dim = dim

    def forward(self, feat):
        return torch.flatten(feat, start_dim=self.dim)

class ResNetEncoder(models.resnet.ResNet):
    """Wrapper for TorchVison ResNet Model
    This was needed to remove the final FC Layer from the ResNet Model"""
    def __init__(self, block, layers):
        super().__init__(block, layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

class ResNet50(ResNetEncoder):
    def __init__(self, cifar_head=True, hparams=None):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3])

class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False

class SimCLRResNetEncoder(nn.Module):
  def __init__(self, disentangle):
        super().__init__()

        self.convnet = ResNet50()
        self.encoder_dim = 2048
        self.disentangle = disentangle

        self.proj_dim = 128
        projection_layers = [
            ('fc1', nn.Linear(self.encoder_dim, self.encoder_dim, bias=False)),
            ('bn1', nn.BatchNorm1d(self.encoder_dim)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.encoder_dim, 128, bias=False)),
            ('bn2', BatchNorm1dNoBias(128)),
        ]

        self.projection = nn.Sequential(OrderedDict(projection_layers))

  def forward(self, x):
        h = self.convnet(x)
        return self.projection(h), h

class SimCLRResNetClassifier(nn.Module):
  def __init__(self, encoder, output_dim, disentangle):
        super().__init__()
        self.disentangle = disentangle
        self.encoder = encoder
        for param in self.encoder.parameters():
          param.requires_grad = False
        self.encoder.projection_layers = nn.Identity()
        self.classification_layer = nn.Linear(2048, output_dim)


  def forward(self, x):
    proj, rep = self.encoder(x)
    out = self.classification_layer(rep)
    return out, rep