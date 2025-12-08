import torch
from torch import nn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from sklearn import metrics
import wandb
from torchvision import transforms

from utils.base_trainer import Trainer
from utils.defaut_args import parser
from utils.stain_aug import StainAug
from dataset.datasets import SupervisedDataset
from models.cnn import ResNetEncoder
from models.vit import ViTEncoder


class SupervisedTrainer(Trainer):
    """Patch-based supervised trainer"""

    def __init__(self, args):
        """
        Initialize the supervised trainer
        :param args: argparse arguments entered for experiment
        """
        self.run_task = 'Strongly supervised patch-level training'
        print(f'Task={self.run_task}')
        super().__init__(args)

        # loss func
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = self.loss_fn.to(self.device)

        # definintions to create datasets, found in base Trainer class
        self.Dataset_Class = SupervisedDataset
        self.norm_mean = (0.5, 0.5, 0.5)
        self.norm_std = (0.5, 0.5, 0.5)
        self.train_transforms = transforms.Compose([
            # resize/randomcrop?
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomChoice([
                transforms.RandomRotation((0, 0)),
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((180, 180)),
                transforms.RandomRotation((270, 270))
            ]),
            StainAug(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
        ])
        self.eval_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm_mean, std=self.norm_std)
        ])

    def init_model_and_optimizer(self):
        """
        Initialize model and optimizer for supervised training
        """

        # define model for supervised patch learning
        if 'resnet' in self.args.model_name:
            self.model = ResNetEncoder(self.args)
        elif 'vit' in self.args.model_name:
            self.model = ViTEncoder(self.args)
        else:
            raise NotImplementedError(f'Model {self.args.model_name} not implemented for supervised trainer!')
        if self.args.use_pretrained:
            self.load_pretrained(path=self.args.pretrained_path)

        self.model = self.model.to(self.device)
        # freeze feature model
        for param in self.model.parameters():
            param.requires_grad = False

        # randomly initialize and train final classification layer
        if 'resnet' in self.args.model_name:
            self.model.model.fc = nn.Linear(
                self.model.hidden_dim,
                self.args.num_classes
            ).to(self.device)
            params = self.model.model.fc.parameters()
        elif 'vit' in self.args.model_name:
            self.model.head = nn.Linear(
                self.model.hidden_dim,
                self.args.num_classes
            ).to(self.device)
            params = self.model.head.parameters()

        # optim definition
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(self.args.beta1, self.args.beta2)
        )

    def train_one_epoch(self, dataloader):
        """
        Defines one epoch of supervised training loop
        :param dataloader: torch train dataloader object to iterate
        :return: a tuple of model outputs to compute metrics on
        """
        self.model.train()
        y_pred = []
        y_probs = []
        y_true = []
        total_loss = 0
        n_iters = 0

        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)

            # predict and loss
            predictions = self.model(images)
            loss = self.loss_fn(predictions, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # get predicted class and labels for logging
            pred_probs = torch.softmax(predictions.detach(), dim=-1).cpu().numpy().tolist()
            pred_class = torch.argmax(predictions.detach(), dim=-1).cpu().numpy().tolist()
            labels = labels.detach().cpu().numpy().tolist()

            y_probs.extend(pred_probs)
            y_pred.extend(pred_class)
            y_true.extend(labels)
            total_loss += loss.item()
            n_iters += 1

        return y_true, y_pred, y_probs, total_loss / n_iters

    def evaluate(self, dataloader, stage):
        """
        Defines one epoch of supervised evaluation
        :param dataloader: torch dataloader object to iterate through
        :param stage: either 'val' or 'test' based on the eval stage
        :return: a tuple of model outputs to compute metrics on
        """
        self.model.eval()
        with torch.no_grad():
            y_pred = []
            y_probs = []
            y_true = []
            total_loss = 0
            n_iters = 0

            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                # predict and loss
                predictions = self.model(images)
                loss = self.loss_fn(predictions, labels)

                # get predicted class and labels for logging
                pred_probs = torch.softmax(predictions.detach(), dim=-1).cpu().numpy().tolist()
                pred_class = torch.argmax(predictions.detach(), dim=-1).cpu().numpy().tolist()
                labels = labels.detach().cpu().numpy().tolist()

                y_probs.extend(pred_probs)
                y_pred.extend(pred_class)
                y_true.extend(labels)
                total_loss += loss.item()
                n_iters += 1

        return y_true, y_pred, y_probs, total_loss / n_iters

    def compute_metrics(self, outputs, stage):
        """
        Compute relevant metrics for patch-based supervised learning to track progress
        :param outputs: tuple of model outputs to compute metrics on (y_true, y_pred, y_probs, avg_CE_loss)
        :param stage: stage of experiment denoted by a string
        :return: a dictionary of metrics to be tracked and a key metric used to choose best validation model
        """

        y_true, y_pred, y_probs, ce_loss = outputs

        # get confusion matrix metrics
        acc = metrics.accuracy_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred, average='macro')
        precision = metrics.precision_score(y_true, y_pred, average='macro')
        f1 = metrics.f1_score(y_true, y_pred, average='macro')

        # return metrics dict for wandb with confusion matrix + ROC
        metrics_dict = {
            f'{stage}_acc': acc,
            f'{stage}_CE_loss': ce_loss,
            f'{stage}_recall': recall,
            f'{stage}_precision': precision,
            f'{stage}_f1': f1,
            f'{stage}_conf_matrix': wandb.plot.confusion_matrix(
                y_true=y_true, preds=y_pred, title=f'{stage}_conf_matrix'),
            f'{stage}_ROC_curve': wandb.plot.roc_curve(y_true=y_true, y_probas=y_probs, classes_to_plot=[1])
        }

        # use this metric to choose best models
        key_metric = metrics_dict[f'{stage}_acc']

        return metrics_dict, key_metric

    def configure_wandb_metrics(self):
        """
        Configures wandb metrics for supervised training
        """
        wandb.config.run_task = self.run_task
        for stage in ['train', 'val', 'test']:
            wandb.define_metric(f"{stage}_loss", summary="min")
            wandb.define_metric(f"{stage}_acc", summary="max")
            wandb.define_metric(f"{stage}_recall", summary="max")
            wandb.define_metric(f"{stage}_precision", summary="max")
            wandb.define_metric(f"{stage}_f1", summary="max")

    def load_pretrained(self, path):
        """
        Loads pretrained model for training, with a special case dealing with simclr-trained weights
        :param path: path to pretrained model weights
        """
        if 'simclr_resnet50.ckpt' not in path:
            # typical model loading
            pretrained_dict = torch.load(path)
        else:
            # load SimCLR weights, needs extra processing
            pretrained_dict = torch.load(path)['state_dict']
            for key in list(pretrained_dict.keys()):
                pretrained_dict[key.replace('model.resnet.', 'model.')] = pretrained_dict.pop(key)

        model_dict = self.model.state_dict()
        #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # if pretrained_dict == {}:
        #     raise Warning('No model weights were loaded!')
        # else:
        #     print('Pretrained model weights were loaded!')
        # model_dict.update(pretrained_dict)
        res = self.model.load_state_dict(pretrained_dict)
        print(res)


if __name__ == "__main__":
    # add new args here
    parser.add_argument('--pretrained_path', default=None, type=str,
                        help='path to pretrained model')
    parser.add_argument('--use_pretrained', default=0, type=int,
                        help='whether to use pretrained model or not')
    parser.add_argument('--model_name', default='resnet18', type=str,
                        help='name of model for training')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classification classes')
    parser.add_argument('--embedding_size', default=512, type=int,
                        help='size of embedding for ResNet output')

    args = parser.parse_args()
    trainer = SupervisedTrainer(args)
    trainer.run()
