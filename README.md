# EideticNet

EideticNet is a PyTorch-based framework for training neural networks that can learn multiple tasks sequentially without [(catastrophic) forgetting](https://en.wikipedia.org/wiki/Catastrophic_interference). It accomplishes this through iterative pruning, selective deletion of synaptic connections on a task-specific basis, and parameter freezing. The available pruning methods are [Taylor expansion-based pruning](https://openaccess.thecvf.com/content_CVPR_2019/html/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.html), Lp-norm weight magnitude pruning, and random pruning.

See [Eidetic Learning: an Efficient and Provable Solution to Catastrophic Forgetting](https://arxiv.org/abs/2502.09500) for an example of research conducted by training `EideticNet`s.

# Features

- For each task on which a network is trained, accuracy is preserved perfectly.
- Functions for propagating sparsity induced by pruning to subsequent layers.
- Forward transfer is configurable. When forward transfer is enabled, the features learned during training of previous tasks are available to (the subsequent layers of) subsequent ones. When forward transfer is disabled, each task occupies its own disjoint subnetwork.
- Batch normalization parameters are preserved for each task.

# Installation

1. Install the package:
```bash
pip install -e .
```

2. Install required dependencies:
```bash
pip install torch torchmetrics torchvision
```
Note that the command for installing PyTorch may vary depending on your environment.

# Supported networks


- **MLP (Multi-Layer Perceptron)**
  ```python
  from eideticnet_training.networks import MLP

  mlp = MLP(
      in_features=784,  # Input dimension
      num_classes=[10, 10],  # List of output dimensions for each task
      num_layers=2,  # Number of hidden layers
      width=4096,  # Width of hidden layers
      dropout=0.0,  # Dropout probability
      bn=True  # Use batch normalization
  )
  ```

- **ConvNet**
  ```python
  from eideticnet_training.networks import ConvNet

  cnn = ConvNet(
      in_channels=3,  # Input channels
      num_classes=[10, 10],  # List of output dimensions for each task
      num_layers=2,  # Number of conv layers per block
      width=32,  # Base width of conv layers
      dropout=0.0,  # Dropout probability
      bn=True  # Use batch normalization
  )
  ```

- **ResNet (18/34/50/101)**
  ```python
  from eideticnet_training.networks import ResNet

  # ResNet-18
  resnet18 = ResNet(
      in_channels=3,
      num_classes=[10, 10],
      n_blocks=[2, 2, 2, 2],
      expansion=1
  )

  # ResNet-34
  resnet34 = ResNet(
      in_channels=3,
      num_classes=[10, 10],
      n_blocks=[3, 4, 6, 3],
      expansion=1
  )

  # ResNet-50
  resnet50 = ResNet(
      in_channels=3,
      num_classes=[10, 10],
      n_blocks=[3, 4, 6, 3],
      expansion=4
  )

  # ResNet-101
  resnet101 = ResNet(
      in_channels=3,
      num_classes=[10, 10],
      n_blocks=[3, 4, 23, 3],
      expansion=4
  )
  ```

# Basic Usage

1. Define your network by inheriting from `EideticNetwork`:

```python
from eideticnet_training.networks import EideticNetwork

class MyNetwork(EideticNetwork):
    def __init__(self):
        super().__init__()
        # Define network layers

    def forward(self, x):
        # Define forward pass

    def _bridge_prune(self, pct, pruning_type, score_threshold=None):
        # Define pruning connections between layers
```

2. Train sequential tasks:

```python
model = MyNetwork()
optimizer = torch.optim.Adam(model.parameters())

for task_id in range(num_tasks):
    # Prepare for new task
    model.prepare_for_task(task_id)

    # Train the task
    model.train_task(
        dataloader=train_loader,
        metric=accuracy_metric,
        optimizer=optimizer,
        test_batch_size=256,
        pruning_step=0.1,  # Prune 10% of parameters per iteration
        pruning_type="l2",  # Use L2 norm-based pruning
        validation_tasks=validation_datasets,
        validation_metrics=validation_accuracy_metrics,
        early_stopping_patience=5
    )
```

For working examples, see:

- Task-incremental learning for image classification: `experiments/sequential_classification.py`
- Cyclical task-incremental learning for image classification: `experiments/cyclical_classification.py`.
- Unit test: `tests/test_eidetic_network.py`

# Testing

Run the test suite:
```bash
pip install -e .'[test]'
pytest
```

# Contributing

See `CONTRIBUTING.md`.

# Citing

If you use this software in your research please cite:

```
@misc{dronen2025eideticlearningefficientprovable,
      title={Eidetic Learning: an Efficient and Provable Solution to Catastrophic Forgetting},
      author={Nicholas Dronen and Randall Balestriero},
      year={2025},
      eprint={2502.09500},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.09500},
}
```
