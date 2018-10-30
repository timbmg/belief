import torch.nn as nn
import torchvision
from .MLP import MLP


class FiLMWrapper(nn.Module):

    def __init__(self, resnet_version, mlp_hidden_size,
                 langugae_embedding_size, remove_last_layers=1):

        super().__init__()

        self.langugae_embedding_size = langugae_embedding_size

        self.resnet = self.get_resnet(resnet_version)(pretrained=True)

        self.mlp_hidden_size = mlp_hidden_size

        self.filmed_bn_layers = self.replace_batchnorm_layers()

        self.resnet = nn.Sequential(
            *list(self.resnet.children())[:-remove_last_layers])

        if remove_last_layers == 0:
            self.out_features = 1000
        elif remove_last_layers == 1:
            self.out_features = 2048
        elif remove_last_layers == 2:
            self.out_features = 2048 * 7 * 7

    # def to(self, device):
    #     super().to(device)
    #     for i in range(len(self.filmed_bn_layers)):
    #         self.filmed_bn_layers[i] = self.filmed_bn_layers[i].to(device)
    #
    #     return self

    def forward(self, image, language_embedding):

        for i in range(len(self.filmed_bn_layers)):
            self.filmed_bn_layers[i].set_input_encoding(language_embedding)

        return self.resnet(image).squeeze(2).squeeze(2)

    def replace_batchnorm_layers(self):

        filmed_bn_layers = list()

        resnet_stages = [self.resnet.layer1, self.resnet.layer2,
                         self.resnet.layer3, self.resnet.layer4]
        for si, stage in enumerate(resnet_stages):
            stage_layers = list()
            for bi, bottleneck in enumerate(stage.children()):
                bottleneck.bn1 = FiLMedBatchNorm2d(
                    self.langugae_embedding_size, self.mlp_hidden_size,
                    bottleneck.bn1)
                bottleneck.bn2 = FiLMedBatchNorm2d(
                    self.langugae_embedding_size, self.mlp_hidden_size,
                    bottleneck.bn2)
                bottleneck.bn3 = FiLMedBatchNorm2d(
                    self.langugae_embedding_size, self.mlp_hidden_size,
                    bottleneck.bn3)
                stage_layers.append(bottleneck)
                filmed_bn_layers += [bottleneck.bn1, bottleneck.bn2,
                                     bottleneck.bn3]

            if si == 0:
                self.resnet.layer1 = nn.Sequential(*stage_layers)
            elif si == 1:
                self.resnet.layer2 = nn.Sequential(*stage_layers)
            elif si == 2:
                self.resnet.layer3 = nn.Sequential(*stage_layers)
            elif si == 3:
                self.resnet.layer4 = nn.Sequential(*stage_layers)

        return filmed_bn_layers

    def film_parameters(self):
        parameters = list()
        for filmed_bn_layer in self.filmed_bn_layers:
            parameters += list(filmed_bn_layer.parameters())
        return parameters

    def get_resnet(self, version):
        if version == 18:
            return torchvision.models.resnet18
        elif version == 34:
            return torchvision.models.resnet34
        elif version == 50:
            return torchvision.models.resnet50
        elif version == 101:
            return torchvision.models.resnet101
        elif version == 152:
            return torchvision.models.resnet152
        else:
            raise ValueError(("Invalid ResNet version {}. "
                              + "Choose from 18, 34, 50, 101, 152.")
                             .format(version))


class FiLMedBatchNorm2d(nn.Module):

    def __init__(self, in_features, hidden_size, bn):

        super().__init__()

        self.bn = bn
        self.bn_weight = nn.Parameter(bn.weight, requires_grad=False)
        self.bn_bias = nn.Parameter(bn.bias, requires_grad=False)

        self.weight_mlp = MLP(
            [in_features, hidden_size, self.bn_weight.size(0)],
            activation='relu', bias=[True, False])
        self.bias_mlp = MLP(
            [in_features, hidden_size, self.bn_bias.size(0)],
            activation='relu', bias=[True, False])

        self.hidden_size = hidden_size

    def __repr__(self):
        return ("FiLMedBatchNorm2d({}, eps={}, momentum={}, affine={}, "
                + "track_running_stats={}, mlp_hidden_size={})").format(
                self.bn.num_features, self.bn.eps, self.bn.momentum,
                self.bn.affine, self.bn.track_running_stats, self.hidden_size)

    # def to(self, device):
    #     self.bn = self.bn.to(device)
    #     self.bn_weight = self.bn_weight.to(device)
    #     self.bn_bias = self.bn_bias.to(device)
    #
    #     return self

    def set_input_encoding(self, x):
        self.input_encoding = x

    def forward(self, x):

        batch_size, num_features, width, height = x.size()

        # if self.training:
        if False:
            mean = x.transpose(0, 1).contiguous()\
                .view(self.bn.num_features, -1).mean()
            var = x.transpose(0, 1).contiguous()\
                .view(self.bn.num_features, -1).var()
            self.update_running_stats(mean, var)
        else:
            mean = self.bn.running_mean\
                .view(1, -1, 1, 1)
            var = self.bn.running_var\
                .view(1, -1, 1, 1)

        delta_weight = self.weight_mlp(self.input_encoding)
        weight = delta_weight + self.bn_weight.view(1, -1)
        weight = weight.view(batch_size, -1, 1, 1)

        delta_bias = self.bias_mlp(self.input_encoding)
        bias = delta_bias + self.bn_bias.view(1, -1)
        bias = bias.view(batch_size, -1, 1, 1)

        return weight * (x-mean)/(var + self.bn.eps) + bias

    def update_running_stats(self, batch_mean, batch_var):
        self.running_mean = self.exp_running_avg(
            self.bn.running_mean, batch_mean, self.bn.momentum)
        self.running_var = self.exp_running_avg(
            self.bn.running_var, batch_var, self.bn.momentum)

    def exp_running_avg(self, running, new, momentum):
        return (1-momentum) * running + momentum * new
