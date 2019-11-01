# @author Justin Chu 2019
import traceback

import torch
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)


class Events(object):
    EPOCH_START = "EPOCH_START"
    EPOCH_END = "EPOCH_END"
    BATCH_START = "BATCH_START"
    BATCH_END = "BATCH_END"


class State(object):
    def __init__(self):
        self.epoch = 0
        self.max_epoch = 0
        self.batch = 0
        self.step = 0
        self.avg_loss = 0


class Estimator(object):
    """
    The base class for estimator that provides a train mathod, and calls
    hook functions. It can do automatic checkpointing and resumes
    to the most recent checkpoint.
    To do that, I need to have a file that stores the metadata of the model
    """

    def __init__(self, model, optimizer, criterion):
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._state = State()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def model(self):
        return self._model

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def criterion(self):
        return self._criterion

    @property
    def state(self):
        return self._state

    @property
    def device(self):
        return self._device

    def _call_handlers(self, handlers, event):
        if event in handlers:
            for hook in handlers[event]:
                try:
                    hook(self, event)
                except Exception:
                    logger.warning("Error calling hook during {}".format(event))
                    traceback.print_exc()

    def train(self, data_loader: DataLoader, max_epochs: int, handlers=None):
        """
        @param      handlers        The handlers in a dictionary
                                 {
                                    Events.EPOCH_START: []
                                    Events.EPOCH_END: []
                                    Events.BATCH_START: []
                                    Events.BATCH_END: []
                                 }
        """
        logger.info("Training started using {}.".format(self._device))
        logger.info(repr(self.model))

        if handlers is None:
            # TODO: apply default handlers
            handlers = {}
        self._state.max_epoch = max_epochs
        device = self._device
        self._model.train()
        self._model.to(device)

        for epoch in range(max_epochs):
            self._state.epoch = epoch
            accumulated_loss = 0.0
            # Call handlers in EPOCH_START
            self._call_handlers(handlers, Events.EPOCH_START)

            for batch_num, batch in enumerate(data_loader):
                self._state.step += len(batch[0])
                self._state.batch = batch_num
                self._call_handlers(handlers, Events.BATCH_START)

                self._model.train()
                # Transfer the features to device
                batch = [elem.to(device) for elem in batch]
                loss = self.training_step(batch)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                accumulated_loss += loss.item()
                self.state.avg_loss = accumulated_loss / (batch_num + 1)

                # Clean up cuda cash
                torch.cuda.empty_cache()

                self._call_handlers(handlers, Events.BATCH_END)

            self._call_handlers(handlers, Events.EPOCH_END)

    def training_step(self, batch):
        """
        Rewrite the training step to work with different data loaders
        """
        x, y = batch
        y_hat = self._model.forward(x)
        loss = self._criterion(y_hat, y)
        return loss
