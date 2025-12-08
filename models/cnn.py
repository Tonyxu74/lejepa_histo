from torch import nn
import torchvision
import torch


class ResNetEncoder(nn.Module):
    """ResNet model for self-supervised learning, supports multicrops!"""

    def __init__(self, args):
        super(ResNetEncoder, self).__init__()

        self.model = torchvision.models.__dict__[args.resnet_name](pretrained=False)

        # match hidden dim of resnet size, FC layer from PAWS paper
        hidden_dim = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, args.embedding_size)
        )

    def forward(self, inputs):
        # forward function dealing with multicrops, which are different in dimension to large crops

        # operates on lists of inputs
        if not isinstance(inputs, list):
            inputs = [inputs]

        # deals with input shapes in list, each idx indexes a different input size to permit multicrops
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)

        # pass each input size through individually
        start_idx = 0
        for end_idx in idx_crops:
            _feat = self.model(torch.cat(inputs[start_idx:end_idx]))

            if start_idx == 0:
                feat = _feat
            else:
                feat = torch.cat((feat, _feat))
            start_idx = end_idx

        return feat
