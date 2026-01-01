import jax.numpy as jnp
import gymnax


def make_env(n_actions):
    env, env_params = gymnax.make("Pendulum-v1")
    actions = jnp.linspace(-2.0, 2.0, n_actions)

    return env, env_params, actions