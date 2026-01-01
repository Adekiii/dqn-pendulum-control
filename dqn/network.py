import flax.linen as nn
import jax.numpy as jnp


class QNetwork(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x):
        x = (x - jnp.array([0.0, 0.0, 0.0])) / jnp.array([1.0, 1.0, 8.0])
        x = nn.Dense(128, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        
        return nn.Dense(self.n_actions)(x)