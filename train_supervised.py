import numpy as np
import torch
from torch import nn
from torch.utils import data
from sklearn import metrics
import wandb
from torchvision import transforms
import tqdm
import h5py
import os

from utils.base_trainer import Trainer
from utils.defaut_args import parser
from utils.stain_aug import StainAug
from dataset.datasets import SupervisedDataset, FeatureExtractionDataset
from models.cnn import CNN


class SupervisedTrainer(Trainer):
    """Patch-based CNN supervised trainer"""

    def __init__(self, args):
        """
        Initialize the supervised trainer
        :param args: argparse arguments entered for experiment
        """
        self.run_task = 'Strongly supervised patch-level CNN training'
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
        return_features = self.args.running_mode == 'deploy_ensemble'
        self.model = CNN(args, return_features=return_features)
        if self.args.use_pretrained:
            self.load_pretrained(path=self.args.pretrained_path)

        self.model = self.model.to(self.device)

        # optim definition
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(self.args.beta1, self.args.beta2)
        )

    def train_one_epoch(self, dataloader):
        """
        Defines one epoch of supervised CNN training loop
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
        Defines one epoch of supervised CNN evaluation
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
        Configures wandb metrics for supervised CNN training
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
        Loads pretrained model for CNN training, with a special case dealing with simclr-trained weights
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
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        if pretrained_dict == {}:
            raise Warning('No model weights were loaded!')
        else:
            print('Pretrained model weights were loaded!')
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def get_wsi_eval_iterator(self, use_all_datasets=False, return_dims=False):
        """
        Get a WSI-level eval iterator to extract features or obtain slide-level metrics
        :param use_all_datasets: if True, also return a dataloader for train+val WSIs for feature extraction
        :param return_dims: if True, dimensions of each slide will also be returned from dataloader
        :return: WSI-level loader for test datalist (optionally also train+val)
        """
        eval_ds = FeatureExtractionDataset(
            self.args,
            self.test_data['data'],
            self.eval_transforms,
            return_dims=return_dims
        )

        if use_all_datasets:
            trainval_ds = FeatureExtractionDataset(
                self.args,
                self.train_data['data'] + self.val_data['data'],
                self.eval_transforms,
                return_dims=return_dims
            )
            return eval_ds, trainval_ds

        return eval_ds

    def deploy_ensemble(self):
        """
        Run and average outputs from k-fold pretrained models and save heatmaps and features for each train/val/test set
        WSI for MIL methods (such as feature engineering and maxpooling on patch-level classification)
        """
        print(f'Using ensemble of models from base path: {self.args.ensemble_base_path}')
        assert self.args.world_size <= 1, 'Distributed training not supported with this eval function!'

        # make output dir
        if not os.path.exists(self.args.train_ensemble_output_path):
            os.makedirs(self.args.train_ensemble_output_path)
        if not os.path.exists(self.args.test_ensemble_output_path):
            os.makedirs(self.args.test_ensemble_output_path)

        with torch.no_grad():
            for fold in range(self.args.num_folds):
                print(f'======== Running on fold {fold} ========')

                # load model
                self.args.use_pretrained = True
                self.args.pretrained_path = f'{self.args.ensemble_base_path}/best_model_weights_fold_{fold}.pt'
                self.init_model_and_optimizer()
                self.model.eval()

                # test_dataloader, trainval_dataloader = self.get_wsi_eval_iterator(
                #     return_dims=True, use_all_datasets=True)
                test_dataloader = self.get_wsi_eval_iterator(return_dims=True)

                for dataloader, output_path in zip(
                    [test_dataloader],  # [test_dataloader, trainval_dataloader],
                    [self.args.test_ensemble_output_path],  # [self.args.test_ensemble_output_path, self.args.train_ensemble_output_path],
                ):
                    for wsi_patch_generator, wsi_id, wsi_level0_dims, wsi_level2_dims, (mpp_x, mpp_y), wsi_label, patient_id in tqdm.tqdm(dataloader):

                        print(f'Running on slide: {wsi_id}')
                        patch_probs = []
                        patch_locs = []
                        patch_feats = []
                        patch_tb_probs = []
                        for patches, locations in wsi_patch_generator:
                            # run patches thru model, append patch probabilities to list
                            patches = patches.to(self.device)
                            logits, features = self.model(patches)
                            probs = torch.softmax(logits, dim=1)[:, 1]  # get probs of tumor class
                            if self.args.num_classes == 3:
                                tb_probs = torch.softmax(logits, dim=1)[:, 2]  # get probs of tumor bed class
                                patch_tb_probs.extend(tb_probs.detach().cpu().numpy())

                            patch_probs.extend(probs.detach().cpu().numpy())
                            patch_feats.extend(features.detach().cpu().numpy())
                            patch_locs.extend(locations)

                        # save to h5 file
                        heatmap_output_path = f'{output_path}/{wsi_id}.h5'
                        mode = 'w' if fold == 0 else 'a'
                        with h5py.File(heatmap_output_path, mode) as f:
                            f.create_dataset(f'fold_{fold}_probs', data=patch_probs)
                            f.create_dataset(f'fold_{fold}_xy_coords', data=patch_locs)
                            f.create_dataset(f'fold_{fold}_features', data=patch_feats)
                            if self.args.num_classes == 3:
                                f.create_dataset(f'fold_{fold}_tb_probs', data=patch_tb_probs)

                            # save slide info for first fold only!
                            if fold == 0:
                                f.create_dataset('mpp_x', data=mpp_x)
                                f.create_dataset('mpp_y', data=mpp_y)
                                f.create_dataset('level0_dims', data=wsi_level0_dims)
                                f.create_dataset('level2_dims', data=wsi_level2_dims)
                                f.create_dataset('label', data=wsi_label)
                                f.create_dataset('patch_mpp', data=0.5)
                                f.create_dataset('patient_id', data=patient_id)

                            # adjust to align features so don't have to do it in MIL training
                            if fold == 4:
                                base_coords = None
                                all_features = []

                                for feat_fold in range(5):
                                    feats = f[f'fold_{feat_fold}_features'][:]
                                    coords = f[f'fold_{feat_fold}_xy_coords'][:].tolist()

                                    # if this is the first fold, set the base coords
                                    if base_coords is None:
                                        base_coords = coords
                                        all_features.append(feats)
                                    
                                    # otherwise, add to feats in the same order as base coords
                                    else:
                                        feats = np.array([feats[coords.index(coord)] for coord in base_coords])
                                        all_features.append(feats)
                                
                                f.create_dataset('base_coords', data=base_coords)
                                f.create_dataset('all_features', data=all_features)


if __name__ == "__main__":
    # add new args here
    parser.add_argument('--pretrained_path', default=None, type=str,
                        help='path to pretrained model')
    parser.add_argument('--use_pretrained', default=0, type=int,
                        help='whether to use pretrained model or not')
    parser.add_argument('--resnet_name', default='resnet18', type=str,
                        help='name of resnet model for CNN')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classification classes')
    parser.add_argument('--mask_base_path', default=None, type=str,
                        help='base path to WSI masks')
    parser.add_argument('--include_tb', default=False, type=lambda x: bool(int(x)),
                        help='whether to include tumor bed in training')

    parser.add_argument('--running_mode', default='run', type=str,
                        help='can do: run, deploy_ensemble')
    parser.add_argument('--ensemble_base_path', default=None, type=str,
                        help='base path to get model weights for ensembling')
    parser.add_argument('--train_ensemble_output_path', default=None, type=str,
                        help='base path to save ensemble output heatmaps for train/val set')
    parser.add_argument('--test_ensemble_output_path', default=None, type=str,
                        help='base path to save ensemble output heatmaps for test set')

    args = parser.parse_args()
    trainer = SupervisedTrainer(args)
    if args.running_mode == 'run':
        trainer.run()
    elif args.running_mode == 'deploy_ensemble':
        trainer.deploy_ensemble()
    else:
        raise NotImplementedError
