import copy
import math
from itertools import cycle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import wandb
from tqdm import tqdm

from .prune.hooks import register_eidetic_hooks
from .prune.masks import (
    UNASSIGNED_PARAMS,
    assign_1d_params_to_task,
    assign_Xd_params_to_task_forward_transfer,
    assign_Xd_params_to_task_last_layer,
    assign_Xd_params_to_task_no_forward_transfer,
    register_parameter_assignment_buffer,
    should_use_forward_transfer,
    unassigned_params_mask,
)
from .prune.taylor import (
    TaylorScorer,
    get_already_pruned_neurons,
    get_already_trained_neurons,
)
from .utils import count_neurons, validate_ignore_modules

SUPPORTED = [nn.Linear, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d]


class EideticNetwork(nn.Module):
    """
    The base class from which to inherit when implementing an eidetic network.

    Initialize an EideticNetwork.

    EideticNetwork is the base class for implementing neural networks capable
    of learning multiple tasks sequentially without catastrophic forgetting. It
    provides the structure and methods necessary for iterative pruning,
    reinitializing pruned parameters, and ensuring that trained tasks are not
    affected by the parameter updates performed when training subsequent tasks.

    A subclass is required to be implemented as follows:
        * Define a `_bridge_prune` method. This method uses existing pruning
          functions to ensure that pruning masks are propagated consistently
          across layers. The actual pruning is done by functions
          in `eideticnet.prune.bridge` (see `bridge_prune` and
          `bridge_prune_residual`). For example, when a batch normalization
          layer follows a linear layer, the linear layer's neurons are pruned
          according to the pruning type (L1, L2, Taylor, or random) and the
          induced sparsity pattern dictates which parameters of the subsequent
          batch normalization layer are pruned.
        * Define the classifier heads for the model as a `torch.nn.ModuleList`,
          one for each task. The name of the property should be `classifiers`.

    To use the implementation, for each of your tasks n in {0, 1, ..., N}:
        * Call the `prepare_for_task(n)` method.
        * Train the task by calling the `train_task` method.

    """

    def __init__(self):
        super().__init__()
        self.phase = None

    @property
    def device(self):
        return list(self.parameters())[0].device

    def set_phase(self, phase: int):
        """
        Set the phase of a task when training one in a sequence of tasks. This
        must be called before training any task. The phases of a sequence of
        tasks are a 0-based list of integers.

        Parameters
        ----------
        phase : int
            The training phase.
        """
        if self.phase is None and phase != 0:
            raise ValueError("Please start with phase=0.")

        wandb.log({"phase": phase})

        if self.phase is None:
            for m in self.modules():
                if (
                    hasattr(m, "weight")
                    and m.weight is not None
                    and m.weight.requires_grad
                ):
                    # The mask for unused capacity is initially -1 and is set
                    # to the task ID once it has been allocated to a task.
                    register_parameter_assignment_buffer(m, "weight", m.weight)
                    prune.identity(m, "weight")
                if (
                    hasattr(m, "bias")
                    and m.bias is not None
                    and m.bias.requires_grad
                ):
                    register_parameter_assignment_buffer(m, "bias", m.bias)
                    prune.identity(m, "bias")
            self.eidetic_handles = register_eidetic_hooks(self)

        self.phase = phase
        # FIXME This logic contradicts the requirements of class-incremental
        # learning (CIL), in which there's only a single classifier head.
        for i in range(len(self.classifiers)):
            self.classifiers[i].requires_grad_(i == phase)

    def reset_capacity_buffers(self):
        for buffer_name, buffer in self.named_buffers():
            if "excess_capacity" in buffer_name:
                buffer.fill_(UNASSIGNED_PARAMS)

    @torch.no_grad
    def _activate_excess_capacity(model, phase, forward_transfer=True):
        # Update the buffers that track which parameters have been assigned to
        # that task and will be frozen during training of subsequent tasks.
        if model.phase is None:
            raise ValueError(
                "You can only call _activate_excess_capacity after one phase "
                "of training."
            )
        tasks_assigned = {}
        for module_name, module in model.named_modules():
            if hasattr(module, "weight_mask"):
                # The zero elements of the weight mask correspond to weights
                # that have been pruned.
                pruned_mask = module.weight_mask == 0
                # Refill the weight mask elements to 1, because we'll start
                # pruning again from scratch in the next task.
                module.weight_mask.fill_(1)
                module.weight_orig[pruned_mask] = 0
                tasks_assigned[
                    f"tasks_assigned/{module_name}.weight"
                ] = wandb.Histogram(module.weight_excess_capacity.cpu())
                if module.weight.ndim > 1:
                    # FIXME merge task assignment functions/logic into a single
                    # generic function.
                    if hasattr(module, "last_layer"):
                        # FIXME We need a more flexible way to handle
                        # classifier heads. In some cases, we may want to
                        # propagate a classifier head's masks across all
                        # classifier heads of downstream tasks. Since
                        # subclasses of EideticNetwork (as implemented today)
                        # only prune the classifier head of the current task,
                        # task assignment across multiple classifier heads is
                        # inconsistent.
                        assign_Xd_params_to_task_last_layer(
                            module.weight_excess_capacity, pruned_mask, phase
                        )
                    elif should_use_forward_transfer(module, forward_transfer):
                        assign_Xd_params_to_task_forward_transfer(
                            module.weight_excess_capacity,
                            pruned_mask,
                            phase,
                        )
                    else:
                        assign_Xd_params_to_task_no_forward_transfer(
                            module.weight_excess_capacity,
                            pruned_mask,
                            phase,
                        )
                else:
                    assign_1d_params_to_task(
                        module.weight_excess_capacity, pruned_mask, phase
                    )
                if (
                    hasattr(module, "bias")
                    and module.bias is not None
                    # Don't prune the biases of the last layer.
                    and not hasattr(module, "last_layer")
                ):
                    pruned_mask = module.bias_mask == 0
                    module.bias_mask.fill_(1)
                    module.bias_orig[pruned_mask] = 0
                    assign_1d_params_to_task(
                        module.bias_excess_capacity, pruned_mask, phase
                    )
                    tasks_assigned[
                        f"tasks_assigned/{module_name}.bias"
                    ] = wandb.Histogram(module.bias_excess_capacity.cpu())
        wandb.log(tasks_assigned)

    @torch.no_grad
    def _random_initialize_excess_capacity(model):
        # Re-initialize the parameters that have not yet been assigned to any
        # task.
        for module in model.modules():
            if hasattr(module, "weight_excess_capacity"):
                if not module.weight_orig.requires_grad:
                    continue
                placeholder = module.weight.clone()
                if module.weight.ndim > 1:
                    torch.nn.init.kaiming_uniform_(placeholder, a=math.sqrt(5))
                    module.weight_orig[
                        unassigned_params_mask(module.weight_excess_capacity)
                    ] = placeholder[
                        unassigned_params_mask(module.weight_excess_capacity)
                    ]
                else:
                    module.weight_orig[
                        unassigned_params_mask(module.weight_excess_capacity)
                    ] = 1
            if hasattr(module, "bias_excess_capacity"):
                placeholder = module.bias.clone()
                if module.weight.ndim > 1:
                    fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                        module.weight
                    )
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                else:
                    bound = 0
                torch.nn.init.uniform_(placeholder, -bound, bound)
                module.bias_orig[
                    unassigned_params_mask(module.bias_excess_capacity)
                ] = placeholder[
                    unassigned_params_mask(module.bias_excess_capacity)
                ]
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.running_mean[
                    unassigned_params_mask(module.weight_excess_capacity)
                ] = 0
                module.running_var[
                    unassigned_params_mask(module.weight_excess_capacity)
                ] = 1

    def prepare_for_task(self, t: int, forward_transfer=True):
        """
        Prepares the network to train a subsequent task without forgetting the
        previous one by updating the buffers that track which parameters have
        been assigned to previous tasks and re-initialized the parameters that
        have not yet been assigned to any task.

        Parameters
        ----------
        t : int
            The task ID that will be trained next.
        forward_transfer : bool
            Share features learned by previous tasks with subsequent tasks.
            When false, per-task subnetworks are disjoint and can be trained
            independently.
        """
        previous_phase = self.phase
        self.set_phase(t)
        if t > 0:
            # Some task was trained before the one we're preparing to train.
            # When calling _activate_excess_capacity (which assigns neurons
            # that were *not* pruned during the previous task to that task and
            # freezes them), we pass in the previous phase, to keep track of
            # which parameters are associated with which task.
            self._activate_excess_capacity(
                phase=previous_phase, forward_transfer=forward_transfer
            )
            self._random_initialize_excess_capacity()

    def _bridge_prune(self, pct, pruning_type=2, score_threshold=None):
        raise NotImplementedError(
            "Implement your custom bridge pruning method"
        )

    def bridge_prune_taylor(
        self,
        pct,
        dataloader,
        loss_function,
        pruning_step_size_is_constant=False,
    ):
        if not hasattr(self, "scorer"):
            ###########################################################
            # Setup Taylor pruning for the network the first time through.
            ###########################################################
            def is_supported(module):
                return isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d))

            # Score the neurons of supported layers excluding the output layer.
            modules = [
                module
                for module_name, module in self.named_modules()
                if is_supported(module) and "classifiers." not in module_name
            ]
            self.scorer = TaylorScorer(modules)

        self.scorer.register_hooks()

        ###################################################################
        # We will run a training pass (in eval mode) to obtain scores. Before
        # doing that, verify that we won't end up pruning too many neurons.
        ###################################################################

        ###################################################################
        # Taylor pruning determines what to prune on a global scale. There's a
        # single score threshold across all layers such that any neuron with an
        # importance score below that threshold is pruned, if it hasn't been
        # pruned already. Here we find the score threshold.
        ###################################################################
        num_neurons = sum(
            count_neurons(module) for module in self.scorer.modules
        )

        def num_pruned_or_trained():
            already_pruned = 0
            already_trained = 0
            for module in self.scorer.modules:
                if hasattr(module, "weight_mask"):
                    already_pruned_neurons = get_already_pruned_neurons(
                        module.weight_mask
                    )
                    already_pruned += already_pruned_neurons.sum()
                if hasattr(module, "weight_excess_capacity"):
                    already_trained_neurons = get_already_trained_neurons(
                        module.weight_excess_capacity
                    )
                    already_trained += already_trained_neurons.sum()
            return already_pruned + already_trained

        ###################################################################
        # TaylorScorer ensures that neurons that have already been pruned or
        # been frozen have a score of infinity, so every pruning iteration we
        # only need to prune the fixed percentage of the number of neurons.
        ###################################################################
        num_pruned = num_pruned_or_trained()
        num_remaining = num_neurons - num_pruned

        if pruning_step_size_is_constant:
            num_to_prune = math.floor(num_neurons * pct)
        else:
            num_to_prune = math.floor(num_remaining * pct)

        if num_to_prune == 0:
            raise StopIteration("Number of neurons to prune is 0.")

        if num_to_prune > (num_neurons - num_pruned):
            raise StopIteration(
                f"Number of neurons to prune {num_to_prune} is greater than "
                "the number of neurons that can be pruned "
                f"({num_neurons - num_pruned})"
            )

        ###############################################################
        # Run either the entire training set, or a random subset of the
        # training set of user-specified size, through the network to
        # accumulate scores.
        ###############################################################
        self.eval()
        self.zero_grad()
        for x, y in tqdm(
            dataloader, total=len(dataloader), desc="Taylor scoring"
        ):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            loss = loss_function(self(x)[self.phase], y)
            loss.backward()

        # Since networks currently specify the modules to prune, and their
        # relationshoips, in their own _bridge_prune method, there is no
        # need to specify any modules to ignore here.
        self.scorer.partition_scores()
        self.scorer.normalize_scores()
        normalized_scores = tuple(
            [
                module.normalized_scores
                for module in self.scorer.modules
                if hasattr(module, "normalized_scores")
            ]
        )
        all_scores = torch.cat(normalized_scores).flatten()
        assert len(all_scores) == num_neurons, (
            "all_scores",
            len(all_scores),
            "num_neurons",
            num_neurons,
        )
        self.scorer.unregister_hooks()

        # Scores are globally normalized across layers, so to find the next
        # neurons to score that will minimally affect task performance, we
        # just choose the ones with the lowest scores.
        kth_lowest_value, _ = torch.kthvalue(all_scores, num_to_prune)
        score_threshold = kth_lowest_value.item()

        wandb.log(
            {
                "taylor/num_to_prune": num_to_prune,
                "taylor/pct": pct,
                "taylor/score_threshold": score_threshold,
                "taylor/min_score": all_scores.min().item(),
                "taylor/max_score": all_scores.max().item(),
                "taylor/mean_score": all_scores.mean().item(),
            }
        )

        return score_threshold

    def bridge_prune(
        self,
        pct,
        dataloader,
        pruning_type,
        loss_function=F.cross_entropy,
        pruning_step_size_is_constant=False,
    ):
        is_in_training = self.training
        score_threshold = None

        if pruning_type == "taylor":
            score_threshold = self.bridge_prune_taylor(
                pct,
                dataloader,
                loss_function,
                pruning_step_size_is_constant=pruning_step_size_is_constant,
            )

        # Call the subclass's implementation of bridge pruning, which wires up
        # the layers such that they are pruned consistenty.
        self._bridge_prune(
            pct, pruning_type=pruning_type, score_threshold=score_threshold
        )
        if is_in_training:
            self.train()

    @torch.no_grad
    def test(self, dataset, metric, class_metric, phase, test_batch_size):
        self.eval()
        test_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=8,
            shuffle=False,
            drop_last=False,
            batch_size=test_batch_size,
        )
        metric.reset()
        class_metric.reset()
        for data, target in test_loader:
            target = target.to(self.device, non_blocking=True)
            data = data.to(self.device, non_blocking=True)
            output = self(data)
            preds = output[phase]
            metric(preds, target)
            class_metric(preds, target)
        wandb.log(
            {
                f"eval/epoch_metric_task{phase:02d}": metric.compute().item(),
                f"eval/epoch_class_metric_task{phase:02d}": class_metric.compute().tolist(),
            }
        )

    def dummy_test(self, phase):
        wandb.log(
            {
                f"eval/epoch_metric_task{phase:02d}": 0,
                f"eval/epoch_class_metric_task{phase:02d}": [],
            }
        )

    @torch.no_grad
    def predict(self, dataset, batch_size):
        self.eval()
        test_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=8,
            shuffle=False,
            drop_last=False,
            batch_size=batch_size,
        )
        for data, target in test_loader:
            target = target.to(self.device, non_blocking=True)
            data = data.to(self.device, non_blocking=True)
            output = self(data)
            yield output, target

    def train_one_epoch(self, loader, metric, optimizer):
        self.train()
        metric.reset()
        for batch_idx, (data, target) in enumerate(loader):
            target = target.to(self.device, non_blocking=True)
            data = data.to(self.device, non_blocking=True)
            optimizer.zero_grad()
            outputs = self(data)
            norms = {}
            for i, c in enumerate(self.classifiers):
                if hasattr(c, "weight"):
                    norms[f"norms/{i:03d}"] = c.weight.square().sum().item()
                else:
                    norms[f"norms/{i:03d}"] = sum(
                        [
                            _c.weight.square().sum().item()
                            for _c in c.children()
                            if hasattr(_c, "weight")
                        ]
                    )

            loss = F.cross_entropy(outputs[self.phase], target)
            loss.backward()
            m = metric(outputs[self.phase], target)
            wandb.log(
                {"train/loss": loss.item(), "train/batch_metric": m, **norms}
            )
            optimizer.step()
        return metric.compute()

    def report(self, ignore_modules=None):
        ignore_modules = validate_ignore_modules(ignore_modules)
        total_params = 1
        total_excess = 0
        total_masked = 0
        for i, (module_name, m) in enumerate(self.named_modules()):
            if module_name in ignore_modules:
                continue
            if hasattr(m, "weight_orig"):
                params = torch.numel(m.weight)
                total_params += params

                if hasattr(m, "weight_mask"):
                    masked = torch.count_nonzero(m.weight_mask == 0).item()
                    total_masked += masked

                # Count the number of parameters that have not yet been
                # assigned to a task.
                excess = unassigned_params_mask(m.weight_excess_capacity).sum()
                total_excess += excess

                module_class = type(m).__name__
                weight_shape = tuple(m.weight.shape)

                masked_format = (
                    "masked/layers/layer{i:03d}-{module_class}-{weight_shape}"
                )
                masked_key = masked_format.format(
                    i=i, module_class=module_class, weight_shape=weight_shape
                )
                excess_format = (
                    "excess/layers/layer{i:03d}-{module_class}-{weight_shape}"
                )
                excess_key = excess_format.format(
                    i=i, module_class=module_class, weight_shape=weight_shape
                )
                wandb.log(
                    {masked_key: masked / params, excess_key: excess / params}
                )

        wandb.log(
            {
                "masked": total_masked / total_params,
                "excess": total_excess / total_params,
            }
        )

        return total_masked / total_params, total_excess / total_params

    def test_all(
        self,
        validation_tasks,
        validation_metrics,
        validation_class_metrics,
        test_batch_size,
        validation_task_nums=set(),
    ):
        args_provided = all(
            (
                validation_tasks,
                validation_metrics,
                validation_class_metrics,
                test_batch_size,
            )
        )
        if not args_provided:
            raise ValueError("Some arguments to test_all are None")
        if not validation_task_nums:
            validation_task_nums = set(
                [i for i in range(len(validation_tasks))]
            )
        for i, t in enumerate(validation_tasks):
            if i in validation_task_nums:
                self.test(
                    t,
                    validation_metrics[i],
                    validation_class_metrics[i],
                    i,
                    test_batch_size,
                )
            else:
                self.dummy_test(i)

    def train_and_maybe_test(
        self,
        dataloader,
        metric,
        optimizer,
        best,
        validation_tasks=None,
        validation_metrics=None,
        validation_class_metrics=None,
        validation_task_nums=set(),
        test_batch_size=None,
        do_test=True,
    ):
        m = self.train_one_epoch(dataloader, metric, optimizer)
        wandb.log({"train/best_epoch_metric": best, "train/epoch_metric": m})
        self.report()
        if do_test:
            self.test_all(
                validation_tasks,
                validation_metrics,
                validation_class_metrics,
                test_batch_size,
                validation_task_nums=validation_task_nums,
            )
        return m

    def train_task(
        self,
        dataloader,
        metric,
        optimizer,
        test_batch_size,
        max_interpruning_epochs=2,
        pruning_step=0.02,
        pruning_step_size_is_constant=False,
        tolerance=0.03,
        pruning_type="l2",
        validation_tasks=[],
        validation_metrics=[],
        validation_class_metrics=[],
        validation_task_nums=set(),
        last_task=False,
        max_masking_ratio=0.95,
        early_stopping_patience=5,
        reduce_learning_rate_before_pruning=0.0,
        test_frequently=False,
        lower_bound=False,
        max_epochs=math.inf,
    ):
        """
        Train the model on a specific task and, after early stopping patience
        is exceeded, perform iterative pruning.

        Parameters
        ----------
            dataloader (DataLoader): The data loader for the training set.
            metric (callable): The metric used to evaluate model performance.
            optimizer (Optimizer): The optimizer used for training.
            test_batch_size (int): Batch size for testing and validation.
            max_interpruning_epochs (int, optional): Maximum number of epochs
                between pruning iterations. Defaults to 2.
            pruning_step (float or list, optional): Step size for pruning. Can
                be a single value or a list of values. Defaults to 0.02.
            pruning_step_size_is_constant (bool, optional): If True, pruning
                step size remains constant. Defaults to False.
            tolerance (float, optional): Tolerance for performance degradation
                during pruning. Defaults to 0.03.
            pruning_type (str or list, optional): Type of pruning to use. Can
                be a single value or a list of values. Defaults to "l2".
            validation_tasks (list, optional): List of validation tasks.
                Defaults to [].
            validation_metrics (list, optional): List of validation metrics.
                Defaults to [].
            validation_class_metrics (list, optional): List of validation class
                metrics. Defaults to [].
            validation_task_nums (set, optional): Set of validation task
                numbers. Defaults to set().
            last_task (bool, optional): If True, this is the last task in the
                sequence. Defaults to False.
            max_masking_ratio (float, optional): Maximum ratio of parameters
                that can be masked. Defaults to 0.95.
            early_stopping_patience (int, optional): Number of epochs to wait
                before early stopping. Defaults to 5.
            reduce_learning_rate_before_pruning (float, optional): Factor by
                which to reduce learning rate before pruning. Defaults to 0.0.
            test_frequently (bool, optional): If True, test the model
                frequently during training, at the cost of a longer training
                time. Defaults to False.
            lower_bound (bool, optional): If True, return after training and do
                not prune. This is for estimating the accuracy of a baseline.
        """

        if len(validation_tasks):
            assert len(validation_tasks) == len(self.classifiers)

        def reduce_learning_rate(factor):
            for group in optimizer.param_groups:
                group["lr"] = group["lr"] / reduce_learning_rate_before_pruning

        def stop_training(m, epoch, best, best_epoch):
            patience_exceeded = epoch - best_epoch > early_stopping_patience
            # Train if we've memorized the training set or exceeded patience.
            return (
                epoch >= max_epochs or m == 1 or m < best and patience_exceeded
            )

        if isinstance(pruning_type, (tuple, list)):
            pruning_type = cycle(pruning_type)
        else:
            pruning_type = cycle([pruning_type])

        if isinstance(pruning_step, (tuple, list)):
            pruning_step = cycle(pruning_step)
        else:
            pruning_step = cycle([pruning_step])

        best = 0
        best_epoch = 0
        do_stop = False
        train_epoch = 0
        pruning_iteration = 0
        recovery_epoch = 0

        ###################################################################
        # Train the current task.
        ###################################################################
        while True:
            wandb.log(
                {
                    "progress/train_epoch": train_epoch,
                    "progress/pruning_iteration": pruning_iteration,
                    "progress/recovery_epoch": recovery_epoch,
                }
            )

            train_epoch += 1

            m = self.train_and_maybe_test(
                dataloader,
                metric,
                optimizer,
                best,
                validation_tasks,
                validation_metrics,
                validation_class_metrics,
                validation_task_nums,
                test_batch_size,
                do_test=test_frequently,
            )

            if do_stop:
                # Before this epoch, we detected that we'd reached a plateau in
                # training set accuracy, so stop training.
                break

            do_stop = stop_training(m, train_epoch, best, best_epoch)

            if do_stop and reduce_learning_rate_before_pruning > 0:
                # Reduce the learning rate and train for one more epoch.
                reduce_learning_rate(
                    factor=reduce_learning_rate_before_pruning
                )

            if m > best:
                best_epoch = train_epoch

            best = max(best, m)

        # Run all validation sets after ordinary training.
        self.test_all(
            validation_tasks,
            validation_metrics,
            validation_class_metrics,
            test_batch_size,
            validation_task_nums=validation_task_nums,
        )

        if tolerance == -1 or last_task:
            return

        prev_state_dict = copy.deepcopy(self.state_dict())
        masked = 0
        # Ensure that we run at least one pruning iteration, even if the
        # training set accuracy dropped below the best one in the final
        # pre-training epoch.
        m = best

        wandb.log(
            {
                "progress/train_epoch": 0,
                "progress/pruning_iteration": pruning_iteration,
                "progress/recovery_epoch": recovery_epoch,
            }
        )

        ###################################################################
        # Iteratively prune the current task until reaching a stopping
        # condition.
        ###################################################################

        def is_accuracy_within_tolerance(current, best):
            return current >= best * (1 - tolerance)

        while True and not lower_bound:
            try:
                wandb.log(
                    {
                        "progress/train_epoch": 0,
                        "progress/pruning_iteration": pruning_iteration,
                        "progress/recovery_epoch": recovery_epoch,
                    }
                )
                pruning_iteration += 1

                # StopIteration can be raised here.
                if not is_accuracy_within_tolerance(m, best):
                    raise StopIteration(
                        f"Metric {m} fell below threshold "
                        f"{best * (1 - tolerance)}"
                    )
                # Here.
                if masked > max_masking_ratio:
                    raise StopIteration(
                        f"Percent of masked parameters ({masked}) exceeds "
                        f"maximum allowed ({max_masking_ratio})"
                    )

                prev_state_dict = copy.deepcopy(self.state_dict())

                this_pruning_type = next(pruning_type)
                this_pruning_step = next(pruning_step)

                # Or, when doing Taylor pruning, here.
                self.bridge_prune(
                    this_pruning_step,
                    dataloader,
                    pruning_type=this_pruning_type,
                    pruning_step_size_is_constant=pruning_step_size_is_constant,
                )

                masked, _ = self.report()
                wandb.log({"prune": this_pruning_step})

                for recovery_epoch in range(max_interpruning_epochs):
                    wandb.log(
                        {
                            "progress/train_epoch": 0,
                            "progress/pruning_iteration": pruning_iteration,
                            "progress/recovery_epoch": recovery_epoch,
                        }
                    )
                    m = self.train_and_maybe_test(
                        dataloader,
                        metric,
                        optimizer,
                        best,
                        validation_tasks,
                        validation_metrics,
                        validation_class_metrics,
                        validation_task_nums,
                        test_batch_size,
                        do_test=test_frequently,
                    )
                    if is_accuracy_within_tolerance(m, best):
                        best = max(best, m)
                        break
                    else:
                        best = max(best, m)
            except StopIteration as se:
                # Accuracy has dropped below the threshold, the fraction of
                # masked neurons has reached the limit, or -- when doing Taylor
                # pruning -- we requested to prune more neurons than remain.
                print(f"Stopping iterative pruning: {se}")

                self.load_state_dict(prev_state_dict, strict=False)

                if hasattr(self, "scorer"):
                    self.scorer.zero_buffers()

                break

        # Run all validation sets after iterative pruning.
        self.test_all(
            validation_tasks,
            validation_metrics,
            validation_class_metrics,
            test_batch_size,
            validation_task_nums=validation_task_nums,
        )

    def set_input_layer(self, module):
        module.first_layer = True

    def set_output_layer(self, module):
        module.last_layer = True
