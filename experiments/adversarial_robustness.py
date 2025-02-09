import foolbox as fb
import torch
from typing import Dict


def test_adversarial_robustness(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    epsilon: float = 0.3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    """
    Test a PyTorch model's robustness against common adversarial attacks on
    MNIST.

    Usage:
    >> results = test_adversarial_robustness(model, test_loader)
    >> for attack_name, accuracy in results.items():
    >>     print(f"{attack_name} Attack - Accuracy: {accuracy:.4f}")

    Args:
        model: PyTorch model to test
        epsilon: Maximum perturbation size (L∞ norm)
        device: Device to run the attacks on
        test_loader: The test set data loader

    Returns:
        Dictionary containing accuracy under different attacks
    """
    model.eval()

    # Create Foolbox model
    bounds = (0, 1)
    preprocessing = dict(mean=[0.], std=[1.])
    fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)

    attacks = {
        'FGSM': fb.attacks.FGSM(),
        'PGD': fb.attacks.PGD(steps=40),
        'CW': fb.attacks.L2CarliniWagnerAttack(steps=100),
    }

    results = {'Clean': 0.0}
    total_samples = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        total_samples += len(images)

        clean_acc = (fmodel(images).argmax(axis=-1) == labels).float().mean()
        results['Clean'] += clean_acc.item() * len(images)

        for attack_name, attack in attacks.items():
            if attack_name not in results:
                results[attack_name] = 0.0

            _, advs, success = attack(fmodel, images, labels, epsilons=epsilon)
            if advs is not None:
                accuracy = 1 - success.float().mean()
                results[attack_name] += accuracy.item() * len(images)

    results = {k: v/total_samples for k, v in results.items()}

    return results
