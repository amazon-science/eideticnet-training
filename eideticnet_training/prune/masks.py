import torch

UNASSIGNED_PARAMS = -1


def unassigned_params_mask(tensor):
    return tensor == UNASSIGNED_PARAMS


def assigned_params_mask(tensor):
    return tensor > UNASSIGNED_PARAMS


def register_parameter_assignment_buffer(m, name, params):
    name = f"{name}_excess_capacity"
    m.register_buffer(name, UNASSIGNED_PARAMS * torch.ones_like(params))


def _get_repeats(pruned_mask):
    repeats = [1, pruned_mask.size(1)] + [
        1 for _ in range(pruned_mask.ndim - 2)
    ]
    return repeats


def should_use_forward_transfer(module, forward_transfer):
    """
    Returns true if forward transfer is enabled for this module or globally.
    """
    if hasattr(module, "forward_transfer"):
        return module.forward_transfer
    elif hasattr(module, "first_layer"):
        # Since the first layer has no inputs, no input dimensions were pruned
        # in the previous layer, so even if forward transfer is disabled by the
        # global configuration, we implicitly enable forward transfer for the
        # first layer. If we don't do this, the consequences are unfavorable:
        # all input dimensions of the first layer will be assigned to the task
        # that just finished training, those input dimensions will be frozen,
        # and it'll be difficult for the network to provide high accuracy on
        # any subsequent tasks.
        return True
    else:
        # Use the global configuration.
        return forward_transfer


def assign_Xd_params_to_task_last_layer(
    weight_excess_capacity, pruned_mask, task_id
):
    """
    Update the assignment of parameters to the given task such that (1) the
    previously-assigned parameters are unchanged and (2) the parameters that
    were not pruned during training of this task are assigned. Since the layer
    under consideration here is the last layer, no output neurons will have
    been pruned.
    """

    def should_skip_assignment(expanded_pruned_mask):
        """
        For last-layer classifier heads, we don't want to assign parameters to
        a task when (1) no parameters have been assigned and (2) no parameters
        have been pruned. Subclasses of EideticNet currently prune only the
        classifier head of the current task, which means that the classifier
        heads of subsequent tasks are not pruned at all. Because, however,
        Eideticnet._activate_excess_capacity iterates of all the modules of a
        network, this function here can be called on non-active classifier
        heads that haven't been trained at all yet. For those classifier heads,
        check for the special case of (1) and (2) above and return without
        doing anything.

        Note: an edge condition for this special case is when it is not
        possible for feature layers to be pruned any further during training
        of a task the first time it is trained.
        """
        return torch.all(
            weight_excess_capacity == UNASSIGNED_PARAMS
        ) and torch.all(~expanded_pruned_mask)

    previously_assigned_mask = assigned_params_mask(weight_excess_capacity)
    repeats = [pruned_mask.size(0), 1]
    expanded_pruned_mask = pruned_mask.all(0).unsqueeze(0).repeat(*repeats)
    newly_assigned_mask = ~previously_assigned_mask & ~expanded_pruned_mask
    if should_skip_assignment(expanded_pruned_mask):
        return
    weight_excess_capacity[newly_assigned_mask] = task_id


# FIXME merge assign_Xd_params_to_task_forward_transfer and
# assign_Xd_params_to_task_no_forward_transfer into a single function.
def assign_Xd_params_to_task_forward_transfer(
    weight_excess_capacity, pruned_mask, task_id
):
    """
    Update the assignment of parameters to the given task such that (1) the
    previously-assigned parameters are unchanged and (2) the parameters that
    were not pruned during training of this task are assigned.
    """
    previously_assigned_mask = assigned_params_mask(weight_excess_capacity)
    repeats = _get_repeats(pruned_mask)
    expanded_pruned_mask = pruned_mask.all(1).unsqueeze(1).repeat(*repeats)
    newly_assigned_mask = ~previously_assigned_mask & ~expanded_pruned_mask
    weight_excess_capacity[newly_assigned_mask] = task_id


def assign_Xd_params_to_task_no_forward_transfer(
    weight_excess_capacity, pruned_mask, task_id
):
    """
    Update the assignment of parameters to the given task such that (1) the
    previously-assigned parameters are unchanged and (2) the parameters that
    were not pruned during training of this task are assigned.
    """
    previously_assigned_mask = assigned_params_mask(weight_excess_capacity)
    pruned_mask_neurons = pruned_mask.all(0).unsqueeze(0)
    pruned_mask_input_dims = pruned_mask.all(1).unsqueeze(1)
    pruned_mask_no_forward_transfer = (
        pruned_mask_neurons & pruned_mask_input_dims
    )
    newly_assigned_mask = (
        ~previously_assigned_mask & ~pruned_mask_no_forward_transfer
    )
    weight_excess_capacity[newly_assigned_mask] = task_id


def assign_1d_params_to_task(weight_excess_capacity, pruned_mask, task_id):
    """
    Update the assignment of parameters to the given task such that (1) the
    previously-assigned parameters are unchanged and (2) the parameters that
    were not pruned during training of this task are assigned.
    """
    previously_assigned_mask = assigned_params_mask(weight_excess_capacity)
    newly_assigned_mask = ~previously_assigned_mask & ~pruned_mask
    weight_excess_capacity[newly_assigned_mask] = task_id
