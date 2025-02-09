import pytest
import torch
import torch.distributions as dist

from eideticnet_training.losses import (
    double_soft_orthogonality_loss,
    gini_loss,
    kurtosis_loss,
    maximize_minimal_angle_loss,
    mutual_coherence_loss,
    soft_orthogonality_loss,
)

N = 10000


class TestGini:
    """Test the Gini coefficient.

    Verify the correctness of the Gini coefficient loss output and its
    gradients.

    In the tests of the output, the expectations (returned by `get_expected`)
    are based on the table in [1], which provides analytical values of the Gini
    coefficient for different distributions. Only a subset of the distributions
    in the table are tested.

    [1] https://en.wikipedia.org/wiki/Gini_coefficient#Continuous_probability_distribution  # noqa: E501
    """

    def test_output_with_uniform_input(self):
        """Test the Gini coefficient of U(a, b)."""
        x = torch.rand(N)

        def get_expected():
            a = 0
            b = 1
            expected = (b - a) / (3 * (b + a))
            return torch.tensor(expected)

        actual = gini_loss(x)
        assert torch.allclose(actual, get_expected(), atol=1e-1)

    def test_output_with_shifted_uniform_input(self):
        """Test that Gini of U(a-2, b-2) equals Gini of U(a, b).

        They should be equal because gini_loss shifts the input such that the
        minimum is 0.
        """
        x = torch.rand(N)

        def get_expected():
            a = 0
            b = 1
            expected = (b - a) / (3 * (b + a))
            return torch.tensor(expected)

        warning = "negative inputs"
        with pytest.warns(UserWarning, match=warning):
            actual = gini_loss(x - 2, use_abs=False)
        assert torch.allclose(actual, get_expected(), atol=1e-1)

    def test_output_with_dirac_input(self):
        """Test the Gini coefficient of the maximally equal Dirac delta."""
        x = torch.ones(N)

        def get_expected():
            return torch.zeros(1)

        actual = gini_loss(x)
        assert torch.allclose(actual, get_expected())

    def test_output_with_exponential_input(self):
        """Test the Gini coefficient with exponential distribution."""
        rate = torch.rand(1)
        sampler = dist.Exponential(rate)
        sample_shape = torch.tensor((N,))
        x = sampler.sample(sample_shape).flatten()

        def get_expected():
            return torch.tensor(1 / 2)

        actual = gini_loss(x)
        assert torch.allclose(actual, get_expected(), atol=1e-2)

    def test_output_log_normal_input(self):
        """Test the Gini coefficient with log normal distribution."""
        loc = torch.zeros(1)
        scale = torch.ones(1)
        sampler = dist.LogNormal(loc, scale)
        sample_shape = torch.tensor((N,))
        x = sampler.sample(sample_shape).flatten()

        def get_expected():
            return torch.erf(scale / 2)

        actual = gini_loss(x)
        assert torch.allclose(actual, get_expected(), atol=1e-1)

    def test_output_with_uniform_matrix_input(self):
        """Test the Gini coefficient with 2D input."""
        x = torch.rand(3, N)

        def get_expected():
            a = 0
            b = 1
            expected_val = (b - a) / (3 * (b + a))
            expected = [expected_val] * 3
            return torch.tensor(expected)

        actual = gini_loss(x)
        assert torch.allclose(actual, get_expected(), atol=1e-1)

    def test_output_with_gaussian_input(self):
        """Test the Gini coefficient of N(0, 1)."""
        x = torch.randn(N)

        def get_expected():
            xshift = x - x.min()
            return gini_loss(xshift, use_abs=False)

        warning = "negative inputs"
        with pytest.warns(UserWarning, match=warning):
            actual = gini_loss(x, use_abs=False)
        assert torch.allclose(actual, get_expected(), atol=1e-1)

    def test_output_when_truncating_gaussian_input(self):
        """Test the Gini coefficient of N(42, 1) with varying truncation.

        Repeated truncation should cause the Gini coefficient to become
        smaller, because the tails of the Gaussian are extremes (rich
        individuals) and repeated truncation causes the distribution to
        increasingly represent a Dirac delta distribution, which has a Gini
        coefficient value of 0.
        """

        def weakly_decreasing(x):
            diffs = [x[i + 1] <= x[i] for i in range(len(x) - 1)]
            return all(diffs)

        x = 42 + torch.randn(N)
        delta = (x.max() - x.min()) / 10
        losses = []
        for i in range(10):
            x[x < (x.min() + delta)] = x.min() + delta
            x[x > (x.max() - delta)] = x.max() - delta
            losses.append(gini_loss(x).item())
        losses = torch.tensor(losses)
        assert weakly_decreasing(losses)

    def test_backward(self):
        """Test that the gradients go in the right direction.

        Minimizing the Gini coefficient loss should reduce inequality.
        """

        def make_high_gini_vector(n):
            x = torch.zeros(n)
            x[0] = 1.0
            x = x.clone().detach().requires_grad_(True)
            assert torch.allclose(gini_loss(x), torch.tensor(1.0), atol=1e-1)
            return x

        x = make_high_gini_vector(N)
        optimizer = torch.optim.SGD([x], lr=1)
        optimizer.zero_grad()  # Not strictly necessary here.
        y = gini_loss(x)
        y.backward()
        optimizer.step()
        # With a high learning rate the Gini coefficient will be low after one
        # update step.
        assert torch.allclose(gini_loss(x), torch.zeros(1))
        # And none of the population will have zero wealth.
        assert (x == 0.0).sum() == 0


@pytest.mark.parametrize(
    "reduction, normalize",
    (
        (None, False),
        (None, True),
        ("mean", False),
        ("mean", True),
        ("sum", False),
        ("sum", True),
    ),
)
def test_soft_orthogonality_forward(reduction, normalize):
    x = torch.eye(3)
    y = soft_orthogonality_loss(x, reduction=reduction, normalize=normalize)
    expected = {
        (None, False): -x,
        (None, True): -x,
        ("mean", False): 0.0,
        ("mean", True): 0.0,
        ("sum", False): 0.0,
        ("sum", True): 0.0,
    }[(reduction, normalize)]
    assert torch.all(y == expected)


@pytest.mark.parametrize(
    "reduction, normalize",
    (
        (None, False),
        (None, True),
        ("mean", False),
        ("mean", True),
        ("sum", False),
        ("sum", True),
    ),
)
def test_double_soft_orthogonality_forward(reduction, normalize):
    x = torch.eye(3)
    y = double_soft_orthogonality_loss(
        x, reduction=reduction, normalize=normalize
    )
    expected = {
        (None, False): -2 * x,
        (None, True): -2 * x,
        ("mean", False): 0.0,
        ("mean", True): 0.0,
        ("sum", False): 0.0,
        ("sum", True): 0.0,
    }[(reduction, normalize)]
    assert torch.all(y == expected)


@pytest.mark.parametrize("normalize", (False, True))
def test_mutual_coherence_loss(normalize):
    x = torch.eye(3)
    y = mutual_coherence_loss(x, normalize=normalize)
    assert y == 0.0


@pytest.mark.parametrize("normalize", (False, True))
def test_maximize_minimal_angle_loss(normalize):
    x = torch.eye(3)
    y = maximize_minimal_angle_loss(x, normalize=normalize)
    # FIXME: verify the correctness of this.
    assert torch.isclose(y, torch.tensor(-1.5708))


def test_smoke_kurtosis_loss():
    x = torch.randn(10, 30)
    kurtosis_loss(x)


@pytest.mark.parametrize(
    "distribution, excess_kurtosis", (("gaussian", 3.0), ("uniform", 1.8))
)
def test_kurtosis_loss_value(distribution, excess_kurtosis):
    if distribution == "gaussian":
        x = torch.randn(10000)
    elif distribution == "uniform":
        x = torch.rand(10000)
    else:
        raise ValueError(f"Unexpected type of distribution {distribution}.")
    loss = kurtosis_loss(x, excess_kurtosis=excess_kurtosis)
    assert torch.allclose(loss, torch.zeros(1), atol=2e-01)
