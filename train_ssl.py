import torch
import wandb
from torchvision import transforms
from PIL import ImageFilter

from utils.base_trainer import Trainer
from utils.defaut_args import parser
from utils.ssl_losses import NTXent
from dataset.datasets import SimCLRDataset
from models.cnn import ResNetEncoder
from models.vit import ViTEncoder
from utils.ssl_optim import SGD, LARS


class GaussianBlur(object):
    """Gaussian blur transform used in SimCLR transformations"""
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if torch.bernoulli(torch.tensor(self.prob)) == 0:
            return img

        radius = self.radius_min + torch.rand(1) * (self.radius_max - self.radius_min)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class SelfSupervisedTrainer(Trainer):
    """Patch-based self-supervised CNN training"""

    def __init__(self, args):
        """
        Initialize the self-supervised trainer
        :param args: argparse arguments entered for experiment
        """
        self.run_task = 'Patch-level self-supervised CNN pretraining'
        print(f'Task={self.run_task}')
        super().__init__(args)

        self.args.drop_last = True
        print('Set drop_last=True for self-supervised training!')

        # loss func and optim and scheduler
        if self.args.ssl_method == 'simclr':
            self.loss_fn = NTXent(
                tau=self.args.temperature,
                actual_batch_size=self.args.batch_size,
                gather_grads=self.args.world_size > 1,
                world_size=self.args.world_size
            )
        else:
            raise ValueError('Unrecognized ssl method!')
        self.loss_fn = self.loss_fn.to(self.device)

        self.scheduler = None

        # definintions to create datasets, found in base Trainer class
        self.Dataset_Class = SimCLRDataset
        self.norm_mean = (0.5, 0.5, 0.5)
        self.norm_std = (0.5, 0.5, 0.5)
        self.init_transforms()

    def init_model_and_optimizer(self):
        """
        Initialize the model and optimizer for SSL training
        """

        # define model for ssl
        if 'resnet' in self.args.model_name:
            self.model = ResNetEncoder(self.args)
        elif 'vit' in self.args.model_name:
            self.model = ViTEncoder(self.args)
        else:
            raise NotImplementedError(f'Model {self.args.model_name} not implemented for SSL trainer!')
        self.model = self.model.to(self.device)

        if self.args.ssl_method == 'simclr':
            self.args.lr = 0.3 * (self.args.batch_size / 256)
            print('SimCLR lr automatically set to 0.3 * (batch_size / 256)')

        param_groups = [
            {'params': (p for n, p in self.model.named_parameters()
                        if ('bias' not in n) and ('bn' not in n))},
            {'params': (p for n, p in self.model.named_parameters()
                        if ('bias' in n) or ('bn' in n)),
             'LARS_exclude': True,
             'weight_decay': 0}
        ]
        base_optimizer = SGD(
            param_groups,
            weight_decay=self.args.weight_decay,
            momentum=0.9,
            nesterov=False,
            lr=self.args.lr
        )

        self.optimizer = LARS(base_optimizer, trust_coefficient=0.001)

    def init_transforms(self):
        """
        Initialize image transforms for SSL
        """
        # simclr augmentations from SSL in histopathology paper
        largecrop = transforms.RandomResizedCrop((224, 224), scale=(0.14, 1.0))
        simclr_augs = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomChoice([
                transforms.RandomRotation((0, 0)),
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((180, 180)),
                transforms.RandomRotation((270, 270))
            ]),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.4)], p=0.8),
            GaussianBlur(p=0.5)
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
        ])
        largecrop_transforms = transforms.Compose([largecrop, simclr_augs, normalize])

        # check if using multicrops, currently not used
        multicrop_transforms = None
        self.train_transforms = (largecrop_transforms, multicrop_transforms)
        # both val and train should use the SimCLR transforms!
        self.eval_transforms = self.train_transforms

    def train_one_epoch(self, dataloader):
        """
        Defines one epoch of self-supervised training loop
        :param dataloader: torch train dataloader object to iterate
        :return: a tuple of model outputs to compute metrics on
        """
        self.model.train()
        if self.args.ssl_method == 'simclr':
            return self.train_simclr(dataloader)
        else:
            raise NotImplementedError

    def train_simclr(self, dataloader):
        """
        Training loop for SimCLR
        :param dataloader: torch dataloader object to iterate through
        :return: average NTXent loss for epoch
        """
        total_loss = 0
        n_iters = 0
        for (img1, img2) in dataloader:

            # get representations of images, no multicrop for SimCLR
            img1, img2 = img1.to(self.device), img2.to(self.device)
            batch_size = img1.shape[0]
            features = self.model([img1, img2])
            features1, features2 = features[:batch_size], features[batch_size:]

            # NTXent loss
            loss = self.loss_fn(xi=features1, xj=features2)

            # propagate loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_iters += 1

        # return tuple for gather_distributed_outputs
        return total_loss / n_iters,

    def evaluate(self, dataloader, stage):
        """
        Defines one epoch of self-supervised evaluation
        :param dataloader: torch dataloader object to iterate through
        :param stage: either 'val' or 'test' based on the eval stage
        :return: a tuple of model outputs to compute metrics on
        """
        self.model.eval()
        with torch.no_grad():
            if self.args.ssl_method == 'simclr':
                return self.evaluate_simclr(dataloader, stage)
            else:
                raise NotImplementedError

    def evaluate_simclr(self, dataloader, stage):
        """
        Eval loop for SimCLR
        :param dataloader: torch dataloader object to iterate through
        :return: average NTXent loss for epoch
        """

        total_loss = 0
        n_iters = 0
        for (img1, img2) in dataloader:
            # get representations of images, no multicrop for SimCLR
            img1, img2 = img1.to(self.device), img2.to(self.device)
            features1, features2 = self.model(img1), self.model(img2)

            # NTXent loss
            loss = self.loss_fn(xi=features1, xj=features2)

            total_loss += loss.item()
            n_iters += 1

        # return tuple for gather_distributed_outputs
        return total_loss / n_iters,

    def compute_metrics(self, outputs, stage):
        """
        Compute relevant metrics for self-supervised learning
        :param outputs: tuple of model outputs to compute metrics on
        :param stage: stage of experiment
        :return: dict of relevant metrics and a key metric used to choose best validation model
        """
        if self.args.ssl_method == 'simclr':
            nt_xent_loss = outputs[0]
            metrics_dict = {f'{stage}_ntxent_loss': nt_xent_loss}
            key_metric = -metrics_dict[f'{stage}_ntxent_loss']
        else:
            raise NotImplementedError

        return metrics_dict, key_metric

    def configure_wandb_metrics(self):
        """
        Configures wandb metrics for self-supervised training
        """
        wandb.config.run_task = self.run_task
        for stage in ['train', 'val', 'test']:

            if self.args.ssl_method == 'simclr':
                wandb.define_metric(f"{stage}_ntxent_loss", summary="min")
            else:
                raise NotImplementedError


if __name__ == "__main__":
    # add new args here
    parser.add_argument('--pretrained_path', default=None, type=str,
                        help='path to pretrained model')
    parser.add_argument('--use_pretrained', default=0, type=int,
                        help='whether to use pretrained model or not')
    parser.add_argument('--model_name', default='resnet18', type=str,
                        help='name of model for training')
    parser.add_argument('--embedding_size', default=2048, type=int,
                        help='size of embedding for ResNet output')
    parser.add_argument('--ssl_method', default='simclr', type=str,
                        help='ssl method to use (simclr only currently)')
    parser.add_argument('--temperature', default=0.1, type=float,
                        help='temperature for sharpening softmax in NTXent loss')
    parser.add_argument('--multicrop', default=0, type=int,
                        help='number of multicrops to use')

    args = parser.parse_args()
    trainer = SelfSupervisedTrainer(args)
    trainer.run()
