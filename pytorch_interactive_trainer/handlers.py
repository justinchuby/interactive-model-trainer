# @author Justin Chu 2019
import json
import os
import re
import pathlib
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from natsort import natsorted
import logging

from .training import Events

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
        self._estimator = None

    def handle(self, estimator, event):
        """
        Validate using the validation data generator
        """
        self._estimator = estimator
        model = estimator.model
        criterion = estimator.criterion
        model.eval()
        epoch = estimator.state.epoch
        device = estimator.device

        val_result = self.validate(model, criterion, self.test_loader, device)
        message = "\t".join([f"{key}: {value:.4f}" for key, value in val_result.items()])
        logger.info(
            f"Epoch: {epoch}\tTrain Loss: {estimator.state.avg_loss:.4f}" + message
        )
        model.train()

    def validate(self, model: nn.Module, criterion, test_loader: DataLoader, device):
        result = OrderedDict()

        for batch in test_loader:
            batch = [elem.to(device) for elem in batch]
            batch_result = self.validation_step(model, criterion, batch)

            for key, value in batch_result.items():
                result[key] = result.get(key, []).extend(value)

            # Clean up
            for elem in batch:
                del elem
            torch.cuda.empty_cache()

        for key, value in result.items():
            result[key] = np.mean(value)

        return result

    def validation_step(self, model, criterion, batch):
        x, y = batch
        outputs = model.forward(x)
        loss = criterion(outputs, y)
        result = OrderedDict()
        result["val_loss"] = [loss.item()] * batch[0].size()[0]

        _, y_hat = torch.max(F.softmax(outputs, dim=1), 1)
        y_hat = y_hat.view(-1)
        result["accuracy"] = torch.eq(y_hat, y).numpy()

        return result


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
            self.pbar.set_description(f"Epoch {estimator.state.epoch}")

        batch_nb = estimator.state.batch + 1
        if batch_nb % self.print_interval == 0:
            self.pbar.set_postfix(
                loss=f"{estimator.state.avg_loss:.2f}", refresh=False
            )
            self.pbar.update(self.print_interval)


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
        current_version_path = f"{experiment_name}/version_{version}"
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
            logger.warning(f"Starting from version 0 because of error: {e}")
        return 0

    def handle(self, estimator, event):
        # If save the model summary if it doesn't exist
        if not pathlib.Path(self.model_summary_path).exists():
            with open(self.model_summary_path, "w") as outfile:
                outfile.write(str(estimator.model))

        # Save the optimizer and model params
        epoch = estimator.state.epoch
        epoch_path = os.path.join(self.checkpoint_path, f"epoch_{epoch}")
        model_param_path = os.path.join(epoch_path, f"model_epoch_{epoch}.pth")
        optimizer_param_path = os.path.join(
            epoch_path, f"optimizer_epoch_{epoch}.pth"
        )
        os.makedirs(epoch_path, exist_ok=True)
        torch.save(estimator.model.state_dict(), model_param_path)
        torch.save(estimator.optimizer.state_dict(), optimizer_param_path)
        # Save the training states
        estimator_state_path = os.path.join(epoch_path, "estimator_state.json")
        with open(estimator_state_path, "w") as outfile:
            json.dump(estimator.state.__dict__, outfile)

        logger.info(f"Saved model at epoch {epoch} to {model_param_path}")
