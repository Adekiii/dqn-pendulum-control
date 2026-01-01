import jax.random as jr
import jax.numpy as jnp
import jax
from dqn.policy import get_action
from dqn.buffer import Transition


def rollout(key_input, env, q_net, actions, policy_params, env_params, steps_in_episode, epsilon):
    """
    Rollout a jitted gymnax episode with lax.scan.
    Adapted from: https://github.com/RobertTLange/gymnax/blob/main/examples/00_getting_started.ipynb
    """
    # Reset the environment
    key_reset, key_episode = jr.split(key_input)
    obs, state = env.reset(key_reset, env_params)

    def policy_step(carry, _):
        """Step transition in jax env."""
        obs, state, key = carry
        key, key_step, key_net = jr.split(key, 3)

        action, key = get_action(q_net, actions, policy_params, obs, key_net, epsilon)
        action = jnp.array([action])

        next_obs, next_state, reward, done, _ = env.step(
            key_step, state, action, env_params
        )

        carry = (next_obs, next_state, key)
        return carry, (obs, action, reward, next_obs, done)

    # Scan over episode step loop
    _, scan_out = jax.lax.scan(
        policy_step, (obs, state, key_episode), None, steps_in_episode
    )

    return scan_out

def collect_rollout(key, jit_rollout, env, q_net, params, env_params, actions, replay_buffer, epsilon):
    traj = jit_rollout(key, env, q_net, actions, params, env_params, 200, epsilon)
    obs, action, reward, next_obs, done = traj

    for i in range(obs.shape[0]):
        action_idx = jnp.argmin(jnp.abs(actions - action[i][0]))
        replay_buffer.add(
            Transition(
                state=obs[i],
                action=int(action_idx),
                reward=float(reward[i]) * 0.1,
                next_state=next_obs[i],
                done=bool(done[i]),
            )
        )