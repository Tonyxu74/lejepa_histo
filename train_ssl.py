import torch
import wandb
from torchvision import transforms
import json
from math import ceil
from PIL import ImageFilter

from utils.base_trainer import Trainer
from utils.defaut_args import parser
from utils.ssl_losses import NTXent, PAWSLoss
from dataset.datasets import SimCLRDataset, PAWSLabelledDataset
from models.cnn import ResNetEncoder
from utils.ssl_optim import SGD, LARS, WarmupCosineSchedule


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
        elif self.args.ssl_method == 'paws':
            self.loss_fn = PAWSLoss(
                multicrop=self.args.multicrop,
                tau=self.args.temperature,
                T=self.args.target_sharpen,
                me_max=True
            )
        else:
            raise ValueError('Unrecognized ssl method!')
        self.loss_fn = self.loss_fn.to(self.device)

        self.scheduler = None

        # definintions to create datasets, found in base Trainer class, note PAWS also uses the SimCLR dataset!
        self.Dataset_Class = SimCLRDataset
        self.norm_mean = (0.5, 0.5, 0.5)
        self.norm_std = (0.5, 0.5, 0.5)
        self.init_transforms()
        if self.args.ssl_method == 'paws':
            # load support dataset
            with open(self.args.support_json, 'r') as fp:
                self.support_data = json.load(fp)

            # save it for logging purposes!
            if self.args.rank == 0:
                with open(f'{self.args.output_path}/support_datalist.json', 'w') as fp:
                    json.dump(self.support_data, fp, indent=4)
            self.support_dataset = PAWSLabelledDataset(self.args, self.support_data['data'], self.train_transforms[0])

    def init_model_and_optimizer(self):
        """
        Initialize the model and optimizer for SSL training
        """

        # define model for ssl
        self.model = ResNetEncoder(self.args)
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

        if self.args.ssl_method == 'paws':

            self.scheduler = WarmupCosineSchedule(
                base_optimizer,
                warmup_steps=self.args.warmup * ceil(len(self.train_data['data']) / self.args.batch_size),
                start_lr=self.args.start_lr,
                ref_lr=self.args.lr,
                final_lr=self.args.final_lr,
                T_max=self.args.num_epochs * ceil(len(self.train_data['data']) / self.args.batch_size)
            )

        self.optimizer = LARS(base_optimizer, trust_coefficient=0.001)

    def init_transforms(self):
        """
        Initialize image transforms for SimCLR and PAWS
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

        # check if using multicrops, only available for paws
        multicrop_transforms = None
        if self.args.ssl_method == 'paws':

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
        if self.args.ssl_method == 'simclr':
            return self.train_simclr(dataloader)
        elif self.args.ssl_method == 'paws':
            return self.train_paws(dataloader)
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

    def paws_iter(self, uimgs):
        """
        Run one iteration of PAWS (used because train and validate are very similar)
        :param uimgs: batch of unsupervised images
        :return: total_loss, paws_loss, and me_max calculated using loss_fn
        """
        # get support images and labels matrix
        simgs, labels_matrix = self.support_dataset.get_support_images()

        # concat images to pass into encoder simultaneously
        uimgs = [u.to(self.device) for u in uimgs]
        simgs = [s.to(self.device) for s in simgs]
        num_support = len(simgs)
        allimgs = simgs + uimgs

        features = self.model(allimgs)

        # get anchor and targets, do not propagate gradients thru targets
        anchor_supports = features[:num_support]
        anchor_views = features[num_support:]
        target_supports = features[:num_support].detach()
        target_views = features[num_support:].detach()

        # target_views includes multicrop features, have it only include largecrops
        # also image_1 is image_2 target and vice versa, so swap image_1 and image_2
        target_views = torch.cat([
            target_views[self.args.batch_size: 2*self.args.batch_size],
            target_views[:self.args.batch_size]
        ], dim=0)

        # compute loss
        paws_loss, me_max = self.loss_fn(
            anchor_views=anchor_views,
            anchor_supports=anchor_supports,
            anchor_support_labels=labels_matrix,
            target_views=target_views,
            target_supports=target_supports,
            target_support_labels=labels_matrix
        )
        loss = paws_loss + me_max
        return loss, paws_loss, me_max

    def train_paws(self, dataloader):
        """
        Train loop for PAWS
        :param dataloader: torch dataloader object to iterate through
        :return: average total, paws, and me_max loss for epoch
        """

        # change splits between processes every epoch
        self.support_dataset.next_epoch()

        total_loss = 0
        total_paws_loss = 0
        total_me_max_loss = 0
        n_iters = 0
        for uimgs in dataloader:

            loss, paws_loss, me_max = self.paws_iter(uimgs)

            # propagate loss and update scheduler
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_paws_loss += paws_loss.item()
            total_me_max_loss += me_max.item()
            total_loss += loss.item()
            n_iters += 1

        return total_loss / n_iters, total_paws_loss / n_iters, total_me_max_loss / n_iters

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
            elif self.args.ssl_method == 'paws':
                return self.evaluate_paws(dataloader, stage)
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

    def evaluate_paws(self, dataloader, stage):
        """
        Eval loop for PAWS
        :param dataloader: torch dataloader object to iterate through
        :return: average total, paws, and me_max loss for epoch
        """
        # change splits between processes every epoch
        self.support_dataset.next_epoch()

        total_loss = 0
        total_paws_loss = 0
        total_me_max_loss = 0
        n_iters = 0
        for uimgs in dataloader:
            loss, paws_loss, me_max = self.paws_iter(uimgs)

            total_paws_loss += paws_loss.item()
            total_me_max_loss += me_max.item()
            total_loss += loss.item()
            n_iters += 1

        return total_loss / n_iters, total_paws_loss / n_iters, total_me_max_loss / n_iters

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
        elif self.args.ssl_method == 'paws':
            total_loss, paws_loss, me_max = outputs
            metrics_dict = {
                f'{stage}_total_loss': total_loss,
                f'{stage}_paws_loss': paws_loss,
                f'{stage}_me_max_loss': me_max
            }
            key_metric = -metrics_dict[f'{stage}_total_loss']
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
            elif self.args.ssl_method == 'paws':
                wandb.define_metric(f"{stage}_total_loss", summary="min")
                wandb.define_metric(f"{stage}_paws_loss", summary="min")
                wandb.define_metric(f"{stage}_me_max_loss", summary="min")
            else:
                raise NotImplementedError


if __name__ == "__main__":
    # add new args here
    parser.add_argument('--pretrained_path', default=None, type=str,
                        help='path to pretrained model')
    parser.add_argument('--use_pretrained', default=0, type=int,
                        help='whether to use pretrained model or not')
    parser.add_argument('--resnet_name', default='resnet18', type=str,
                        help='name of resnet model for CNN')
    parser.add_argument('--support_batch_size', default=16, type=int,
                        help='number of support images per class to use in PAWS')
    parser.add_argument('--embedding_size', default=2048, type=int,
                        help='size of embedding for ResNet output')
    parser.add_argument('--ssl_method', default='simclr', type=str,
                        help='ssl method to use (simclr or paws)')
    parser.add_argument('--temperature', default=0.1, type=float,
                        help='temperature for sharpening softmax in NTXent/PAWS loss')
    parser.add_argument('--target_sharpen', default=0.25, type=float,
                        help='Target sharpening for PAWS loss')
    parser.add_argument('--support_json', default=None, type=str,
                        help='path to support json file for PAWS')
    parser.add_argument('--multicrop', default=0, type=int,
                        help='number of multicrops to use for PAWS')
    parser.add_argument('--warmup', default=10, type=int,
                        help='number of epochs to warmup for PAWS')
    parser.add_argument('--start_lr', default=0.3, type=float,
                        help='starting learning rate for PAWS scheduler')
    parser.add_argument('--final_lr', default=0.064, type=float,
                        help='ending learning rate for PAWS scheduler')

    args = parser.parse_args()
    trainer = SelfSupervisedTrainer(args)
    trainer.run()
