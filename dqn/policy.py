import jax.random as jr
import jax.numpy as jnp
import jax


def epsilon_greedy(q_values, key, epsilon):
    key, key_explore, key_random = jr.split(key, 3)

    def explore(_):
        return jr.randint(key_explore, shape=(), minval=0, maxval=q_values.shape[0])
        
    def exploit(_):
        return jnp.argmax(q_values)
        
    action = jax.lax.cond(
        jr.uniform(key_random) < epsilon,
        explore,
        exploit,
        operand=None
    )
    
    return action, key

def get_action(q_net, actions, policy_params, obs, key, epsilon=0.1):
    q_values = q_net.apply(policy_params, obs)
    action_idx, key = epsilon_greedy(q_values, key, epsilon)
    action = actions[action_idx]
    
    return action, key