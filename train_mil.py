import numpy as np
import torch
from torch import nn
from sklearn import metrics
import wandb

from utils.base_trainer import Trainer
from utils.defaut_args import parser
from dataset.datasets import MILDataset
from models.mil_models import MILMaxpool, MILAttn, CLAMSingleBranch
from topk import SmoothTop1SVM


class MILTrainer(Trainer):
    """WSI-level multiple-instance learning (MIL) trainer"""

    def __init__(self, args):
        """
        Initialize the MIL trainer
        :param args: argparse arguments entered for experiment
        """
        self.run_task = 'Weakly supervised WSI-level MIL training'
        print(f'Task={self.run_task}')
        super().__init__(args)

        # 1 WSI at a time for MIL
        self.args.batch_size = 1

        # loss func for classification
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = self.loss_fn.to(self.device)

        # definintions to create datasets, found in base Trainer class
        self.Dataset_Class = MILDataset
        self.train_transforms = torch.from_numpy
        self.eval_transforms = torch.from_numpy

    def init_model_and_optimizer(self):
        """
        Initializes model and optimizer for MIL training
        """

        # instance clustering loss function
        inst_loss_fn = SmoothTop1SVM(n_classes=2)

        # they have a builtin cuda function!
        if self.device != 'cpu':
            inst_loss_fn = inst_loss_fn.cuda()

        # define model for MIL learning, code adapted from CLAM paper
        if self.args.model_name == 'mil_maxpool':
            self.model = MILMaxpool(
                n_classes=self.args.num_classes,
                top_k=self.args.maxpool_k,
                fold_num=len(self.args.fold_features.split(','))
            )
        elif self.args.model_name == 'mil_attn':
            self.model = MILAttn(
                gate=self.args.gated_attention,
                size_arg=self.args.model_size,
                dropout=self.args.use_dropout,
                n_classes=self.args.num_classes,
                fold_num=len(self.args.fold_features.split(','))
            )
        elif self.args.model_name == 'clam_sb':
            self.model = CLAMSingleBranch(
                gate=self.args.gated_attention,
                size_arg=self.args.model_size,
                dropout=self.args.use_dropout,
                n_classes=self.args.num_classes,
                instance_loss_fn=inst_loss_fn,
                fold_num=len(self.args.fold_features.split(','))
            )
        else:
            raise NotImplementedError

        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(self.args.beta1, self.args.beta2)
        )

    def train_one_epoch(self, dataloader):
        """
        Defines one epoch of weakly-supervised MIL training loop
        :param dataloader: torch train dataloader object to iterate
        :return: a tuple of model outputs to compute metrics on
        """
        self.model.train()

        # standard MIL training
        if self.args.model_name in ['mil_maxpool', 'mil_attn']:
            return self.train_standard_mil(dataloader)
        # CLAM training
        elif self.args.model_name == 'clam_sb':
            return self.train_clam(dataloader)
        else:
            raise NotImplementedError

    def train_standard_mil(self, dataloader):
        """
        Trains standard MIL methods (maxpool after FC layer, MIL with attention)
        :param dataloader: torch train dataloader object to iterate
        :return: a tuple of model outputs to compute metrics on
        """
        y_pred = []
        y_probs = []
        y_true = []
        total_loss = 0
        n_iters = 0

        for features, labels in dataloader:
            features, labels = features.to(self.device), labels.to(self.device)
            features = features.squeeze(0)

            # prediction is logits for each class for 1 WSI, pred_prob is probability for 1 WSI
            prediction, pred_prob, _, _ = self.model(features)
            loss = self.loss_fn(prediction, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # get predicted class and labels for logging
            pred_prob = pred_prob.detach().cpu().numpy().tolist()
            pred_class = torch.argmax(prediction.detach(), dim=-1).cpu().numpy().tolist()
            labels = labels.detach().cpu().numpy().tolist()

            y_probs.extend(pred_prob)
            y_pred.extend(pred_class)
            y_true.extend(labels)
            total_loss += loss.item()
            n_iters += 1

        return y_true, y_pred, y_probs, total_loss / n_iters

    def train_clam(self, dataloader):
        """
        Trains a Clustering-constrained Attention Multiple Instance Learning (CLAM) model
        :param dataloader: torch train dataloader object to iterate
        :return: a tuple of model outputs to compute metrics on
        """
        y_pred_wsi = []
        y_probs_wsi = []
        y_true_wsi = []
        y_pred_inst = []
        y_true_inst = []
        total_inst_loss = 0
        total_cls_loss = 0
        total_loss = 0
        n_iters = 0

        for features, labels in dataloader:
            features, labels = features.to(self.device), labels.to(self.device)
            features = features.squeeze(0)

            # prediction is logits for each class for 1 WSI, pred_prob is probability for 1 WSI
            # pred_class is argmax of prediction for WSI-level class prediction
            prediction, pred_prob, pred_class, _, instance_dict = self.model(features, label=labels, instance_eval=True)
            cls_loss = self.loss_fn(prediction, labels)
            instance_loss = instance_dict['instance_loss']

            bag_weight = torch.tensor(self.args.bag_weight).to(self.device)
            loss = bag_weight * cls_loss + (1-bag_weight) * instance_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # get WSI predicted class and labels for logging
            pred_prob_wsi = pred_prob.detach().cpu().numpy().tolist()
            pred_class_wsi = pred_class.squeeze(0).detach().cpu().numpy().tolist()
            labels_wsi = labels.detach().cpu().numpy().tolist()

            # get instance classification/clustering predicted class and labels
            pred_class_inst = instance_dict['inst_preds']
            labels_inst = instance_dict['inst_labels']

            y_probs_wsi.extend(pred_prob_wsi)
            y_pred_wsi.extend(pred_class_wsi)
            y_true_wsi.extend(labels_wsi)
            y_pred_inst.extend(pred_class_inst)
            y_true_inst.extend(labels_inst)
            total_loss += loss.item()
            total_inst_loss += instance_loss.item()
            total_cls_loss += cls_loss.item()
            n_iters += 1

        return y_true_wsi, y_pred_wsi, y_probs_wsi, total_loss / n_iters, \
            total_cls_loss / n_iters, total_inst_loss / n_iters, y_true_inst, y_pred_inst

    def evaluate(self, dataloader, stage):
        """
        Evaulate weakly-supervised MIL
        :param dataloader: torch dataloader object to iterate through
        :param stage: either 'val' or 'test' based on the eval stage
        :return: a tuple of model outputs to compute metrics on
        """
        self.model.eval()
        with torch.no_grad():

            # standard MIL eval
            if self.args.model_name in ['mil_maxpool', 'mil_attn']:
                return self.evaluate_standard_mil(dataloader, stage)
            # CLAM eval
            elif self.args.model_name == 'clam_sb':
                return self.evaluate_clam(dataloader, stage)
            else:
                raise NotImplementedError

    def evaluate_standard_mil(self, dataloader, stage):
        """
        Evaulate standard MIL methods
        :param dataloader: torch dataloader object to iterate through
        :param stage: either 'val' or 'test' based on the eval stage
        :return: a tuple of model outputs to compute metrics on
        """
        y_pred = []
        y_probs = []
        y_true = []
        total_loss = 0
        n_iters = 0

        for features, labels in dataloader:
            features, labels = features.to(self.device), labels.to(self.device)
            features = features.squeeze(0)

            # prediction is logits for each class for 1 WSI, pred_prob is probability for 1 WSI
            prediction, pred_prob, _, _ = self.model(features)
            loss = self.loss_fn(prediction, labels)

            # get predicted class and labels for logging
            pred_prob = pred_prob.detach().cpu().numpy().tolist()
            pred_class = torch.argmax(prediction.detach(), dim=-1).cpu().numpy().tolist()
            labels = labels.detach().cpu().numpy().tolist()

            y_probs.extend(pred_prob)
            y_pred.extend(pred_class)
            y_true.extend(labels)
            total_loss += loss.item()
            n_iters += 1

        return y_true, y_pred, y_probs, total_loss / n_iters

    def evaluate_clam(self, dataloader, stage):
        """
        Evaulate CLAM MIL model
        :param dataloader: torch dataloader object to iterate through
        :param stage: either 'val' or 'test' based on the eval stage
        :return: a tuple of model outputs to compute metrics on
        """
        y_pred_wsi = []
        y_probs_wsi = []
        y_true_wsi = []
        y_pred_inst = []
        y_true_inst = []
        total_inst_loss = 0
        total_cls_loss = 0
        n_iters = 0

        for features, labels in dataloader:
            features, labels = features.to(self.device), labels.to(self.device)
            features = features.squeeze(0)

            # prediction is logits for each class for 1 WSI, pred_prob is probability for 1 WSI
            # pred_class is argmax of prediction for WSI-level class prediction
            prediction, pred_prob, pred_class, _, instance_dict = self.model(features, label=labels, instance_eval=True)
            cls_loss = self.loss_fn(prediction, labels)
            instance_loss = instance_dict['instance_loss']

            # get WSI predicted class and labels for logging
            pred_prob_wsi = pred_prob.detach().cpu().numpy().tolist()
            pred_class_wsi = pred_class.squeeze(0).detach().cpu().numpy().tolist()
            labels_wsi = labels.detach().cpu().numpy().tolist()

            # get instance classification/clustering predicted class and labels
            pred_class_inst = instance_dict['inst_preds']
            labels_inst = instance_dict['inst_labels']

            y_probs_wsi.extend(pred_prob_wsi)
            y_pred_wsi.extend(pred_class_wsi)
            y_true_wsi.extend(labels_wsi)
            y_pred_inst.extend(pred_class_inst)
            y_true_inst.extend(labels_inst)
            total_inst_loss += instance_loss.item()
            total_cls_loss += cls_loss.item()
            n_iters += 1

        return y_true_wsi, y_pred_wsi, y_probs_wsi, \
            total_cls_loss / n_iters, total_inst_loss / n_iters, y_true_inst, y_pred_inst

    def compute_metrics(self, outputs, stage):
        """
        Compute relevant metrics for slide-based weakly supervised learning to track progress
        :param outputs: tuple of model outputs to compute metrics on
        :param stage: stage of experiment denoted by a string
        :return: a dictionary of metrics to be tracked and a key metric used to choose best validation model
        """
        metrics_dict = {}
        if self.args.model_name in ['mil_maxpool', 'mil_attn']:
            y_true_wsi, y_pred_wsi, y_probs_wsi, cls_loss = outputs

        elif self.args.model_name == 'clam_sb':
            if stage == 'train':
                y_true_wsi, y_pred_wsi, y_probs_wsi, total_loss, cls_loss, inst_loss, y_true_inst, y_pred_inst = outputs

                # metric specific to clam train
                metrics_dict.update({
                    f'{stage}_total_loss': total_loss
                })
            else:
                y_true_wsi, y_pred_wsi, y_probs_wsi, cls_loss, inst_loss, y_true_inst, y_pred_inst = outputs

            # metrics specific to clam
            acc_inst = metrics.accuracy_score(y_true_inst, y_pred_inst)
            metrics_dict.update({
                f'{stage}_inst_acc': acc_inst,
                f'{stage}_inst_loss': inst_loss,
                f'{stage}_inst_conf_matrix': wandb.plot.confusion_matrix(y_true=y_true_inst, preds=y_pred_inst)
            })

        else:
            raise NotImplementedError

        # get general confusion matrix metrics
        acc_wsi = metrics.accuracy_score(y_true_wsi, y_pred_wsi)
        recall_wsi = metrics.recall_score(y_true_wsi, y_pred_wsi)
        precision_wsi = metrics.precision_score(y_true_wsi, y_pred_wsi)
        f1_wsi = metrics.f1_score(y_true_wsi, y_pred_wsi)

        # return metrics dict for wandb with confusion matrix + ROC
        metrics_dict.update({
            f'{stage}_cls_acc': acc_wsi,
            f'{stage}_cls_loss': cls_loss,
            f'{stage}_cls_recall': recall_wsi,
            f'{stage}_cls_precision': precision_wsi,
            f'{stage}_cls_f1': f1_wsi,
            f'{stage}_cls_conf_matrix': wandb.plot.confusion_matrix(y_true=y_true_wsi, preds=y_pred_wsi),
            f'{stage}_cls_ROC_curve': wandb.plot.roc_curve(y_true=y_true_wsi, y_probas=y_probs_wsi, classes_to_plot=[1])
        })

        # CLAM and standard MIL key metric are all cls loss, **not combined cls and inst losses**
        key_metric = -metrics_dict[f'{stage}_cls_loss']
        return metrics_dict, key_metric

    def configure_wandb_metrics(self):
        """
        Configures wandb metrics for MIL training
        """
        wandb.config.run_task = self.run_task

        # metric only for clam train
        if self.args.model_name == 'clam_sb':
            wandb.define_metric("train_total_loss", summary="min")

        # other metrics
        for stage in ['train', 'val', 'test']:
            # metrics specific to clam
            if self.args.model_name == 'clam_sb':
                wandb.define_metric(f"{stage}_inst_acc",summary="max")
                wandb.define_metric(f"{stage}_inst_loss", summary="min")

            # metrics for all MIL
            wandb.define_metric(f"{stage}_cls_loss", summary="min")
            wandb.define_metric(f"{stage}_cls_acc", summary="max")
            wandb.define_metric(f"{stage}_cls_recall", summary="max")
            wandb.define_metric(f"{stage}_cls_precision", summary="max")
            wandb.define_metric(f"{stage}_cls_f1", summary="max")

    def deploy_ensemble_mil(self):
        """
        Deploys pretrained MIL model ensemble on a new test set
        """
        print(f'Deploy ensemble to test on new data: {self.args.eval_only_ensemble_path}')
        assert self.args.world_size <= 1, 'Distributed training not supported with this eval function!'

        self.init_model_and_optimizer()

        def get_and_print_metrics(y_true, y_probs, filenames):
            roc_auc = metrics.roc_auc_score(y_true, y_probs)
            y_pred = (y_probs > 0.5).astype(int)
            tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
            no_fn_thresh = min(y_probs[y_true == 1])
            ignorable_slides = (y_probs[y_true == 0] < no_fn_thresh).sum()

            fp_filenames = np.array(filenames)[(y_true == 0) & (y_pred == 1)]
            fn_filenames = np.array(filenames)[(y_true == 1) & (y_pred == 0)]

            print(f'ROC AUC: {roc_auc}')
            print(f'TN, FP, FN, TP: {tn, fp, fn, tp}')
            print(f'No FN threshold: {no_fn_thresh}')
            print(f'N ignorable slides: {ignorable_slides}')
            print(f'FP filenames: {fp_filenames}')
            print(f'FN filenames: {fn_filenames}')

        with torch.no_grad():

            all_probs, all_true = [], []

            # get all filenames for checking fp and fn slides
            test_filenames = [d['path'].split('/')[-1] for d in self.test_data['data']]

            for fold in range(self.args.num_folds):
                print(f'======== Running on fold {fold} ========')

                # load model
                pretrained_path = f'{self.args.eval_only_ensemble_path}/best_model_weights_fold_{fold}.pt'
                self.model.load_state_dict(torch.load(pretrained_path))
                self.model.eval()

                # run on test
                test_iter = self.get_eval_iterator(self.test_data['data'])
                outputs = self.test(test_iter)
                if self.args.model_name in ['mil_maxpool', 'mil_attn']:
                    y_true_wsi, y_pred_wsi, y_probs_wsi, cls_loss = outputs
                elif self.args.model_name == 'clam_sb':
                    y_true_wsi, y_pred_wsi, y_probs_wsi, cls_loss, inst_loss, y_true_inst, y_pred_inst = outputs

                y_true_wsi = np.array(y_true_wsi)
                y_probs_wsi = np.array(y_probs_wsi)[:, 1]

                # get metrics and append
                get_and_print_metrics(y_true_wsi, y_probs_wsi, test_filenames)
                all_probs.append(y_probs_wsi)
                all_true.append(y_true_wsi)

            # get ensemble metrics
            print('======== Ensemble metrics ========')
            all_probs = np.array(all_probs)
            assert all([np.array_equal(all_true[0], x) for x in all_true]), 'All true labels must be the same!'
            ensemble_probs = all_probs.mean(axis=0)
            get_and_print_metrics(all_true[0], ensemble_probs, test_filenames)


if __name__ == "__main__":
    # add new args here
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classification classes')
    parser.add_argument('--model_name', default='mil_maxpool', type=str,
                        help='name of MIL model to use')
    parser.add_argument('--fold_features', default='1,2,3,4,5', type=str,
                        help='which folds to get patch features from')
    parser.add_argument('--model_size', default='small', type=str,
                        help='size of MIL model to use')
    parser.add_argument('--use_dropout', default=1, type=int,
                        help='use dropout in training')
    parser.add_argument('--gated_attention', default=1, type=int,
                        help='use gated attention in training')
    parser.add_argument('--maxpool_k', default=1, type=int,
                        help='top k values to be used for maxpooling')
    parser.add_argument('--bag_weight', default=0.7, type=float,
                        help='weight for loss function doing classification over the entire bag of instances (WSI)')

    # args for evaluating on new data
    parser.add_argument('--eval_only', default=0, type=int,
                        help='set to 1 if looking to only eval on a new dataset with pretrained weights')
    parser.add_argument('--eval_only_ensemble_path', type=str,
                        help='path to ensemble model weights for only eval')

    args = parser.parse_args()
    trainer = MILTrainer(args)
    if not args.eval_only:
        trainer.run()
    else:
        trainer.deploy_ensemble_mil()


