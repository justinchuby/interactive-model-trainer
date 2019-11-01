# @author Justin Chu 2019
import json
import os
import re
import pathlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from natsort import natsorted
import logging

from pytorch_interactive_trainer import Events

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


# TODO: make it an abstract class
class Handler:
    def __call__(self, estimator, event):
        self.handle(estimator, event)

    def handle(self, estimator, event):
        raise NotImplementedError


class ValidationHandler(Handler):
    def __init__(self, test_loader: DataLoader):
        self.test_loader = test_loader

    def handle(self, estimator, event):
        """
        Validate using the validation data generator
        """
        model = estimator.model
        criterion = estimator.criterion
        model.eval()
        epoch = estimator.state.epoch
        device = estimator.device

        val_loss, val_acc = test_classify(model, criterion, self.test_loader, device)
        logger.info(
            "Epoch: {}\tTrain Loss: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}".format(
                epoch, estimator.state.avg_loss, val_loss, val_acc
            )
        )
        model.train()


class ProgressBarHandler(Handler):
    """
    Display the loss in a progress bar
    """

    def __init__(self, pbar, batch_len, print_interval=10):
        self.pbar = pbar
        self.batch_len = batch_len
        self.print_interval = print_interval

    def handle(self, estimator, event):
        if event == Events.EPOCH_START:
            self.pbar.reset(total=self.batch_len)
            self.pbar.set_description("Epoch {}".format(estimator.state.epoch))

        batch_nb = estimator.state.batch + 1
        if batch_nb % self.print_interval == 0:
            self.pbar.set_postfix(
                loss="{:.2f}".format(estimator.state.avg_loss), refresh=False
            )
            self.pbar.update(self.print_interval)


def test_classify(model: nn.Module, criterion, test_loader: DataLoader, device):
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)

        _, y_hat = torch.max(F.softmax(outputs, dim=1), 1)
        y_hat = y_hat.view(-1)
        loss = criterion(outputs, labels)

        accuracy += torch.sum(torch.eq(y_hat, labels)).item()
        total += len(labels)
        # TODO: what is this?
        test_loss.extend([loss.item()] * feats.size()[0])
        del feats
        del labels

    return np.mean(test_loss), accuracy / total


class CheckpointHandler(Handler):
    """
    NOTE: Need to run this everytime a new experiment is run
    """

    def __init__(
        self,
        experiment_name: str,
        dirpath: str = None,
    ):
        # TODO: some of these should be handled when training starts.
        # TODO: test location first
        if dirpath is None:
            # Set default checkpoint directory
            dirpath = "experiments"

        version = self._next_version(os.path.join(dirpath, experiment_name))
        self.version = version
        current_version_path = "{experiment_name}/version_{version}".format(
            experiment_name=experiment_name, version=version
        )
        # checkpoint_path is a directory to contain the checkpoints
        self.checkpoint_path = os.path.join(
            dirpath, current_version_path, "checkpoints"
        )
        # Create the checkpoint folder
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        # Save the model summary
        self.model_summary_path = os.path.join(
            dirpath, current_version_path, "model_summary.txt"
        )

    def _next_version(self, dirpath: str) -> int:
        """
        Return the next version number in the given directory
        """
        try:
            version_re = re.compile(r"version_(\d+)")

            def is_valid_version(v: str):
                return version_re.search(v) is not None

            versions = tuple(filter(is_valid_version, os.listdir(dirpath)))
            if not versions:
                # No versions yet
                return 0
            current_version = natsorted(versions, reverse=True)[0]
            # Get the version number using the version pattern
            current_version = int(version_re.search(current_version).group(1))
            return current_version + 1
        except Exception as e:
            logger.warning("Starting from version 0 because of error: {}".format(e))
        return 0

    def handle(self, estimator, event):
        # If save the model summary if it doesn't exist
        if not pathlib.Path(self.model_summary_path).exists():
            with open(self.model_summary_path, "w") as outfile:
                outfile.write(str(estimator.model))

        # Save the optimizer and model params
        epoch = estimator.state.epoch
        epoch_path = os.path.join(self.checkpoint_path, "epoch_{}".format(epoch))
        model_param_path = os.path.join(epoch_path, "model_epoch_{}.pth".format(epoch))
        optimizer_param_path = os.path.join(
            epoch_path, "optimizer_epoch_{}.pth".format(epoch)
        )
        os.makedirs(epoch_path, exist_ok=True)
        torch.save(estimator.model.state_dict(), model_param_path)
        torch.save(estimator.optimizer.state_dict(), optimizer_param_path)
        # Save the training states
        estimator_state_path = os.path.join(epoch_path, "estimator_state.json")
        with open(estimator_state_path, "w") as outfile:
            json.dump(estimator.state.__dict__, outfile)

        logger.info("Saved model at epoch {} to {}".format(epoch, model_param_path))
