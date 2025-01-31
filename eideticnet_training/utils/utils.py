import torch


def update_class_predictions_metrics(
    per_task_predictions,
    task_prediction,
    targets,
    args,
    class_metric,
    oracle_class_metric,
):

    nc = args.num_classes_per_task
    for i in range(len(targets)):
        offset = task_prediction[[i]] * args.num_classes_per_task
        local_predictions = per_task_predictions[task_prediction[i]][i]
        class_prediction = local_predictions[
            task_prediction[i] * nc : (task_prediction[i] + 1) * nc
        ].argmax()
        class_metric.update(class_prediction + offset, targets[[i]])

        oracle = targets[[i]] // nc
        local_predictions = per_task_predictions[oracle[0]][i]
        offset = oracle * args.num_classes_per_task
        class_prediction = local_predictions[
            oracle[0] * nc : (oracle[0] + 1) * nc
        ].argmax()
        oracle_class_metric.update(class_prediction + offset, targets[[i]])


def validate_ignore_modules(ignore_modules):
    if isinstance(ignore_modules, str):
        ignore_modules = set([ignore_modules])
    else:
        ignore_modules = set() if not ignore_modules else ignore_modules
    return ignore_modules


def count_neurons(model, ignore_modules=None):
    """
    Count the number of neurons in a model. Each output filter of a
    convolutional layer is a neuron.
    """
    ignore_modules = validate_ignore_modules(ignore_modules)
    num_neurons = 0
    for module_name, module in model.named_modules():
        if module_name in ignore_modules:
            continue
        if isinstance(module, torch.nn.Linear):
            num_neurons += module.weight.shape[0]
        elif isinstance(module, torch.nn.Conv1d):
            num_neurons += module.weight.shape[0]
        elif isinstance(module, torch.nn.Conv2d):
            num_neurons += module.weight.shape[0]
        elif isinstance(module, torch.nn.Conv3d):
            num_neurons += module.weight.shape[0]
    return num_neurons


def get_module_by_name(module, module_name):
    submodule_names = module_name.split(".")
    next_module = module
    try:
        for submodule_name in submodule_names:
            next_module = getattr(next_module, submodule_name)
        return next_module
    except Exception:
        raise ValueError(f"Did not find {module_name} in {module}")
