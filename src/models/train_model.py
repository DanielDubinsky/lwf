import os
import random
from src.models.renset import ResNet

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from os import environ
from neptunecontrib.monitoring.sacred import NeptuneObserver
from src.models.predict_model import evaluate
from src.models.utils import MultiClassCrossEntropy, get_dataset
import copy


from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment('lwf')
storage_path = 'storage'
ex.observers.append(FileStorageObserver.create(storage_path))

print("using neptune")
ex.observers.append(NeptuneObserver(api_token=environ.get('NEPTUNE_API_TOKEN'),
                                    project_name='danield95/lwf'))

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@ex.config
def my_config():
    workers = 2             # Number of workers for dataloader
    seed = 1234
    batch_size = 128        # Batch size during training
    num_epochs = 5        # Number of training epochs
    lr = 0.1                # Learning rate for optimizers
    gpu = ''
    depth = 20
    num_classes = [10]
    stages = ['cifar10']


class Trainer:

    @ex.capture
    def __init__(self, gpu, seed, _run):
        # SACRED: we don't need any parameters here, they're in the config and the functions get a @ex.capture handle
        # later
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.device = torch.device("cuda:0" if (
            torch.cuda.is_available() and gpu != '') else "cpu")
        self.storage_path = os.path.join(storage_path, str(_run._id))
        self.loss_fn = nn.CrossEntropyLoss()
        self.make_dataset()
        self.make_dataloaders()
        self.make_model()

    # SACRED: The parameters input_size, hidden_size and num_classes come from our Sacred config file. Sacred finds
    # these because of the @ex.capture handle. Note that we did not have to add these parameters when we called this
    # method in the init.
    @ex.capture
    def make_model(self, depth, num_classes, gpu):
        total_classes = sum(num_classes)
        self.net = ResNet(
            depth=depth, num_classes=total_classes).to(self.device)
        self.classifier = self.net.classifier

        if (self.device.type == 'cuda') and (len(gpu.split(',')) > 1):
            # self.net = nn.DataParallel(self.net, list(range(len(gpu.split(',')))))
            self.net = nn.DataParallel(self.net, list(range(len(gpu.split(',')))))
            self.classifier = self.net.module.classifier
        self.classifier.set_trainable(list(range(total_classes)))

    # SACRED: The parameter learning_rate comes from our Sacred config file. Sacred finds this because of the
    # @ex.capture handle. Note that we did not have to add this parameter when we called this method in the init.
    @ex.capture
    def make_optimizer(self, lr, stage):
        # Setup Adam optimizers for both G and D
        # if stage == 0:
        #     params = self.net.parameters()
        # else:
        #     params = self.net.classifier.parameters()
        params = self.net.parameters()
        self.optimizer = torch.optim.SGD(
            params, lr=lr, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=0.001)

    # SACRED: Here we do not use any parameters from the config file and hence we do not need the @ex.capture handle.
    @ex.capture
    def make_dataset(self, stages):
        self.datasets = []
        self.eval_datasets = []
        for stage in stages:
            self.datasets.append(get_dataset(stage, train=True))
            self.eval_datasets.append(get_dataset(stage, train=False))

    # SACRED: The parameter batch_size comes from our Sacred config file. Sacred finds this because of the
    # @ex.capture handle. Note that we did not have to add this parameter when we called this method in the init.
    @ex.capture
    def make_dataloaders(self, batch_size, workers):
        # Create the dataloader
        self.dataloaders = [torch.utils.data.DataLoader(ds, batch_size=batch_size,
                                                        shuffle=True, num_workers=workers) for ds in self.datasets]

    # SACRED: The parameter num_epochs comes from our Sacred config file. Sacred finds this because of the
    # @ex.capture handle. Note that we did not have to add this parameter when we called this method.
    # _run is a special object you can pass to your function and it allows you to keep track of parameters (like we do).
    @ex.capture
    def train(self, num_epochs, stages, num_classes, _run):

        for stage_idx, stage in enumerate(stages):
            start_iter = 0

            dataloader = self.dataloaders[stage_idx]
            stage_cls_idx_start = sum(num_classes[:stage_idx])
            stage_num_classes = num_classes[stage_idx]
            best_stage_accuracy = 0.0
            best_epoch = 0

            prev_net = copy.deepcopy(self.net)

            self.make_optimizer(stage=stage_idx)

            # Freeze classifiers of other stages.
            self.classifier.set_trainable(
                list(range(stage_cls_idx_start, stage_cls_idx_start + stage_num_classes)))

            print(f"Training stage {stage_idx} {stage}")
            # For each epoch
            for epoch in range(start_iter, num_epochs):
                # For each batch in the dataloader
                running_loss = 0.0
                for i, data in enumerate(dataloader, 0):

                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs, features = self.net(inputs)
                    stage_outputs = outputs[:,
                                            stage_cls_idx_start: stage_cls_idx_start + stage_num_classes]

                    cls_loss = self.loss_fn(stage_outputs, labels)

                    # Distillation
                    dist_loss = torch.zeros_like(cls_loss)
                    # for j in range(stage_idx):
                    #     stage_j_cls_idx_start = sum(num_classes[:j])
                    #     stage_j_num_classes = num_classes[j]
                    #     stage_j_cls_indexes = list(range(stage_j_cls_idx_start, stage_j_cls_idx_start + stage_j_num_classes))
                    #     with torch.no_grad():
                    #         j_target, _ = prev_net(inputs)
                    #     j_target = j_target[:, stage_j_cls_indexes]
                    #     j_outputs = outputs[:, stage_j_cls_indexes]
                    #     dist_loss += MultiClassCrossEntropy(j_outputs, j_target, 2)

                    loss = cls_loss + dist_loss

                    loss.backward()
                    self.optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    if i % 50 == 49:    # print every 50 mini-batches
                        print('[%d, %d, %5d] loss: %.3f' %
                              (stage_idx, epoch + 1, i + 1, running_loss / 50))

                        _run.log_scalar(
                            f's{stage_idx}_loss', running_loss / 50)
                        running_loss = 0.0

                self.scheduler.step()

                # Evaluate model accuracy for each stage.
                if stage == 'cifar10&cifar100':
                    current_stage_accuracy = self.eval_joint(
                        stage_idx=stage_idx, epoch=epoch)
                else:
                    current_stage_accuracy = self.eval_past_stages(
                        stage_idx=stage_idx, epoch=epoch)

                if current_stage_accuracy > best_stage_accuracy:
                    best_stage_accuracy = current_stage_accuracy
                    best_epoch = epoch
                    if isinstance(self.net, nn.DataParallel):
                        state_dict = self.net.module.state_dict()
                    else:
                        state_dict = self.net.state_dict()
                    torch.save({'model': state_dict, 'optimizer': self.optimizer.state_dict(
                    ), 'config': _run.config}, f'{_run.experiment_info["name"]}_{_run._id}_s{stage_idx}.pth')
        print(
            f'Done Training - best epoch: {best_epoch} accuracy: {best_stage_accuracy}')

    @ex.capture
    def eval_stages(self, num_classes, _run, stage_idx, epoch):
        current_stage_accuracy = 0.0
        for i in range(stage_idx + 1):
            stage_i_cls_idx_start = sum(num_classes[:i])
            stage_i_num_classes = num_classes[i]
            stage_i_cls_indexes = list(
                range(stage_i_cls_idx_start, stage_i_cls_idx_start + stage_i_num_classes))
            stage_accuracy = evaluate(
                self.net, self.eval_datasets[i], stage_i_cls_indexes, self.device)
            if i == stage_idx:
                current_stage_accuracy = stage_accuracy
            print(f'[{stage_idx}, {epoch + 1}] accuracy stage {i}: {stage_accuracy}')
            _run.log_scalar(f's{stage_idx}_acc_s{i}', stage_accuracy)
        return current_stage_accuracy

    @ex.capture
    def eval_joint(self, num_classes, _run, stage_idx, epoch):
        cifar10, cifar100 = self.eval_datasets[stage_idx]
        acc_cifar10 = evaluate(self.net, cifar10, list(range(10)), self.device)
        acc_cifar100 = evaluate(
            self.net, cifar100, list(range(10, 110)), self.device)
        print(f'[{stage_idx}, {epoch + 1}] accuracy stage cifar10: {acc_cifar10}')
        _run.log_scalar(f's{stage_idx}_acc_cifar10', acc_cifar10)
        print(f'[{stage_idx}, {epoch + 1}] accuracy stage cifar100: {acc_cifar100}')
        _run.log_scalar(f's{stage_idx}_acc_cifar100', acc_cifar100)
        return (acc_cifar10 + acc_cifar100) / 2.0

    @ex.capture
    def run(self, _run):
        print(_run.config)
        self.train()


@ex.main
def main():
    """
    Sacred needs this main function, to start the experiment.
    If you want to import this experiment in another file (and use its configurations there, you can do that as follows:
    import train_nn
    ex = train_nn.ex
    Then you can use the 'ex' the same way we also do in this code.
    """

    trainer = Trainer()
    trainer.run()
    # SACRED: Everything you return here is stored as a result,
    # and will be shown as such on Sacredboard


if __name__ == '__main__':
    ex.run_commandline()  # SACRED: this allows you to run Sacred not only from your terminal,
    # (but for example in PyCharm)
