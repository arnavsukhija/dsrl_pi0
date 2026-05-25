from typing import Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks import MLP
from jaxrl2.networks.constants import default_init

class LearnedStdNormalPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)

        log_stds = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
        return distribution

class TanhMultivariateNormalDiag(distrax.Transformed):

    def __init__(self,
                 loc: jnp.ndarray,
                 scale_diag: jnp.ndarray,
                 low: Optional[jnp.ndarray] = None,
                 high: Optional[jnp.ndarray] = None):
        distribution = distrax.MultivariateNormalDiag(loc=loc,
                                                      scale_diag=scale_diag)

        layers = []

        if not (low is None or high is None):

            def rescale_from_tanh(x):
                x = (x + 1) / 2  # (-1, 1) => (0, 1)
                return x * (high - low) + low

            def forward_log_det_jacobian(x):
                high_ = jnp.broadcast_to(high, x.shape)
                low_ = jnp.broadcast_to(low, x.shape)
                return jnp.sum(jnp.log(0.5 * (high_ - low_)), -1)

            layers.append(
                distrax.Lambda(
                    rescale_from_tanh,
                    forward_log_det_jacobian=forward_log_det_jacobian,
                    event_ndims_in=1,
                    event_ndims_out=1))

        layers.append(distrax.Block(distrax.Tanh(), 1))

        bijector = distrax.Chain(layers)

        super().__init__(distribution=distribution, bijector=bijector)

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

def _last_dim_bias_init(bias_value: float):
    """Init the Dense bias to zeros except the LAST dim, set to `bias_value`.

    Used to bias the actor's pseudo_time output toward +1 (= K_max post-tanh),
    preventing the K-attractor collapse at min_K=1."""
    def init_fn(key, shape, dtype):
        b = jnp.zeros(shape, dtype)
        return b.at[-1].set(bias_value)
    return init_fn


class LearnedStdTanhNormalPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    low: Optional[float] = None
    high: Optional[float] = None
    pseudo_time_init_bias: float = 0.0

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        if self.pseudo_time_init_bias != 0.0:
            means = nn.Dense(self.action_dim,
                             kernel_init=default_init(1e-2),
                             bias_init=_last_dim_bias_init(self.pseudo_time_init_bias))(outputs)
        else:
            means = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)

        log_stds = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = TanhMultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds), low=self.low, high=self.high)
        return distribution