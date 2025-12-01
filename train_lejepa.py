import torch
import wandb
from torchvision import transforms
from math import ceil
from PIL import ImageFilter
from torch.cuda.amp import GradScaler, autocast

from utils.base_trainer import Trainer
from utils.defaut_args import parser
from utils.ssl_losses import SIGReg
from dataset.datasets import SimCLRDataset
from models.cnn import ResNetEncoder
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


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
        self.run_task = 'Patch-level self-supervised CNN pretraining using LeJEPA'
        print(f'Task={self.run_task}')
        super().__init__(args)

        self.args.drop_last = True
        print('Set drop_last=True for self-supervised training!')

        # loss func and optim and scheduler
        self.loss_fn = SIGReg()
        self.loss_fn = self.loss_fn.to(self.device)

        self.scheduler = None
        self.probe = None
        self.scaler = GradScaler(enabled=True)

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
        self.model = ResNetEncoder(self.args)
        self.model = self.model.to(self.device)

        g1 = {"params": self.model.parameters(), "lr": self.args.lr, "weight_decay": 5e-2}

        self.optimizer = torch.optim.AdamW([g1])
        step_per_ep = ceil(len(self.train_data['data']) / self.args.batch_size)
        s1 = LinearLR(
            self.optimizer,
            start_factor=0.01,
            total_iters=self.args.warmup * step_per_ep
        )
        s2 = CosineAnnealingLR(
            self.optimizer,
            T_max=(self.args.num_epochs - self.args.warmup) * step_per_ep,
            eta_min=1e-3
        )
        self.scheduler = SequentialLR(self.optimizer, schedulers=[s1, s2], milestones=[self.args.warmup * step_per_ep])

    def init_transforms(self):
        """
        Initialize image transforms
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

        # check if using multicrops
        multicrop_transforms = None

        if self.args.multicrop > 0:
            multicrop = transforms.RandomResizedCrop((96, 96), scale=(0.05, 0.14))
            multicrop_transforms = transforms.Compose([multicrop, simclr_augs, normalize])

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
        total_loss = 0
        n_iters = 0
        for imgs in dataloader:

            with autocast(enabled=True):

                # get representations of images
                imgs = [img.to(self.device) for img in imgs]
                batch_size = imgs[0].shape[0]
                all_features = self.model(imgs)
                feat_size = all_features.shape[-1]

                # split to get global features
                g1_features, g2_features = all_features[:batch_size], all_features[batch_size: batch_size * 2]

                # lejepa loss
                inv_loss = ((g1_features + g2_features)/2 - all_features.view(-1, batch_size, feat_size)).square().mean()
                sigreg = self.loss_fn(all_features)
                loss = (1 - self.args.lambd) * inv_loss + self.args.lambd * sigreg

            # propagate loss
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

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
        raise NotImplementedError

    def compute_metrics(self, outputs, stage):
        """
        Compute relevant metrics for self-supervised learning
        :param outputs: tuple of model outputs to compute metrics on
        :param stage: stage of experiment
        :return: dict of relevant metrics and a key metric used to choose best validation model
        """

        lejepa_loss = outputs[0]
        metrics_dict = {f'{stage}_lejepa_loss': lejepa_loss}
        key_metric = -metrics_dict[f'{stage}_lejepa_loss']

        return metrics_dict, key_metric

    def configure_wandb_metrics(self):
        """
        Configures wandb metrics for self-supervised training
        """
        wandb.config.run_task = self.run_task
        for stage in ['train', 'val', 'test']:
            wandb.define_metric(f"{stage}_lejepa_loss", summary="min")


if __name__ == "__main__":
    # add new args here
    parser.add_argument('--pretrained_path', default=None, type=str,
                        help='path to pretrained model')
    parser.add_argument('--use_pretrained', default=0, type=int,
                        help='whether to use pretrained model or not')
    parser.add_argument('--resnet_name', default='resnet50', type=str,
                        help='name of resnet model for CNN')
    parser.add_argument('--embedding_size', default=512, type=int,
                        help='size of embedding for ResNet output')
    parser.add_argument('--multicrop', default=8, type=int,
                        help='number of multicrops to use')
    parser.add_argument('--warmup', default=10, type=int,
                        help='number of epochs to warmup')
    parser.add_argument('--lambd', default=0.05, type=float,
                        help='lambda for lejepa loss weighting')

    args = parser.parse_args()
    trainer = SelfSupervisedTrainer(args)
    trainer.run()
