# Interactive Model Trainer
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fjustinchuby%2Fpytorch-interactive-trainer.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2Fjustinchuby%2Fpytorch-interactive-trainer?ref=badge_shield)

Nothing important here, for now

## Motivation

Experimenting with neural network models requires a lot of repetitive setting up of the same things. For example, we will write a train method, a test method and make decisions on how to structure these methods. It would be nice to have a wrapper that defines the APIs so that we can provide a train method, evaluation criteria and just call train!

The wrapper should deal with taking snapshots and be robust to failure so that it can pick up from it left off if the system stops or disconnects.

It should also allow for pausing the training process, evaluate/export the model in the middle of training, and accessing/adjusting the hyperparameters (if they are accessible).

It is also common for us to monitor the training process when doing experiments so we can, say, stop the training can adjust the parameters when the loss goes wild. It would be nice it the wrapper oversees the training process and lets us know how the model is doing. To this end, the wrapper should keep track of the metrics in training and evaluation, provide a report and some description of the trend/statistics of the metrics.

The tool is going to be designed for Pytorch since it's what I am using right now. It might end up looking like Google's `Estimator` (but simpler) in Tensorflow since it does many things I was hoping for.

- Oh I also found `pytorch-lightning`. Looks pretty cool.


## License
[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fjustinchuby%2Fpytorch-interactive-trainer.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Fjustinchuby%2Fpytorch-interactive-trainer?ref=badge_large)