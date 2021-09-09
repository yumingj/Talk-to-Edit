import torch
import torch.nn as nn


class FieldFunction(nn.Module):

    def __init__(
        self,
        num_layer=4,
        latent_dim=512,
        hidden_dim=512,
        leaky_relu_neg_slope=0.2,
    ):

        super(FieldFunction, self).__init__()

        layers = []

        # first layer
        linear_layer = LinearLayer(
            in_dim=latent_dim,
            out_dim=hidden_dim,
            activation=True,
            negative_slope=leaky_relu_neg_slope)
        layers.append(linear_layer)

        # hidden layers
        for i in range(num_layer - 2):
            linear_layer = LinearLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                activation=True,
                negative_slope=leaky_relu_neg_slope)
            layers.append(linear_layer)

        # final layers
        linear_layer = LinearLayer(
            in_dim=hidden_dim, out_dim=latent_dim, activation=False)
        layers.append(linear_layer)

        self.field = nn.Sequential(*layers)

    def forward(self, x):
        x = self.field(x)
        return x


class LinearLayer(nn.Module):

    def __init__(
        self,
        in_dim=512,
        out_dim=512,
        activation=True,
        negative_slope=0.2,
    ):

        super(LinearLayer, self).__init__()

        self.Linear = nn.Linear(
            in_features=in_dim, out_features=out_dim, bias=True)

        self.activation = activation
        if activation:
            self.leaky_relu = nn.LeakyReLU(
                negative_slope=negative_slope, inplace=False)

    def forward(self, x):
        x = self.Linear(x)
        if self.activation:
            x = self.leaky_relu(x)
        return x


class Normalization(nn.Module):

    def __init__(self, ):

        super(Normalization, self).__init__()

        self.mean = torch.tensor([0.485, 0.456, 0.406
                                  ]).unsqueeze(-1).unsqueeze(-1).to('cuda')
        print(self.mean.shape)
        self.std = torch.tensor([0.229, 0.224,
                                 0.225]).unsqueeze(-1).unsqueeze(-1).to('cuda')

    def forward(self, x):
        x = torch.sub(x, self.mean)
        x = torch.div(x, self.std)
        return x
