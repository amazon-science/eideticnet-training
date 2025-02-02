import torch
import wandb
from neural_collapse.accumulate import (
    CovarAccumulator,
    DecAccumulator,
    MeanAccumulator,
    VarNormAccumulator,
)
from neural_collapse.kernels import kernel_stats, log_kernel
from neural_collapse.measure import (
    clf_ncc_agreement,
    covariance_ratio,
    self_duality_error,
    similarities,
    simplex_etf_error,
    variability_cdnv,
)


# For monitoring neural collapse.
class HiddenStates:
    pass


def log_neural_collapse_metrics(
    *, model, classifier, train_loader, validation_tasks, test_batch_size
):
    classifier = model.classifiers[model.phase]
    test_loader = torch.utils.data.DataLoader(
        validation_tasks[model.phase],
        num_workers=8,
        shuffle=False,
        drop_last=False,
        batch_size=test_batch_size,
    )

    with torch.no_grad():
        model.eval()
        weights = classifier.weight
        device = weights.device
        num_classes, hidden_state_dim = weights.shape

        # NC collections
        mean_accum = MeanAccumulator(
            n_classes=num_classes, d_vectors=hidden_state_dim, device=device
        )
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            model(images)
            mean_accum.accumulate(HiddenStates.value, labels)
        means, mG = mean_accum.compute()

        var_norms_accum = VarNormAccumulator(
            n_classes=num_classes,
            d_vectors=hidden_state_dim,
            device=device,
            M=means,
        )
        covar_accum = CovarAccumulator(
            n_classes=num_classes,
            d_vectors=hidden_state_dim,
            device=device,
            M=means,
        )
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            model(images)
            var_norms_accum.accumulate(HiddenStates.value, labels, means)
            covar_accum.accumulate(HiddenStates.value, labels, means)
        var_norms, _ = var_norms_accum.compute()
        covar_within = covar_accum.compute()

        dec_accum = DecAccumulator(
            n_classes=num_classes,
            d_vectors=hidden_state_dim,
            device=device,
            M=means,
            W=weights,
        )
        dec_accum.create_index(means)  # optionally use FAISS index for NCC
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            model(images)

            # mean embeddings (only) necessary again if not using FAISS index
            if dec_accum.index is None:
                dec_accum.accumulate(
                    HiddenStates.value, labels, weights, means
                )
            else:
                dec_accum.accumulate(HiddenStates.value, labels, weights)

        # NC measurements
        results = {
            "nc/nc1_pinv": covariance_ratio(covar_within, means, mG),
            "nc/nc1_svd": covariance_ratio(covar_within, means, mG, "svd"),
            "nc/nc1_quot": covariance_ratio(
                covar_within, means, mG, "quotient"
            ),
            "nc/nc1_cdnv": variability_cdnv(var_norms, means, tile_size=64),
            "nc/nc2_etf_err": simplex_etf_error(means, mG),
            "nc/nc2g_dist": kernel_stats(means, mG, tile_size=64)[1],
            "nc/nc2g_log": kernel_stats(
                means, mG, kernel=log_kernel, tile_size=64
            )[1],
            "nc/nc3_dual_err": self_duality_error(weights, means, mG),
            "nc/nc3u_uni_dual": similarities(weights, means, mG).var().item(),
            "nc/nc4_agree": clf_ncc_agreement(dec_accum),
        }

        wandb.log(results)
