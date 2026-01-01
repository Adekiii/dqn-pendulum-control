import jax.random as jr
import jax.numpy as jnp
import optax
import jax

from environment.pendulum import make_env
from environment.rollout import rollout, collect_rollout
from dqn.network import QNetwork
from dqn.buffer import ReplayBuffer
from utils.visualize import visualize_trajectory

n_actions = 11
state_dim = 3
gamma = 0.99
batch_size = 128
epsilon = 1.0
epsilon_decay = 0.997
min_epsilon = 0.1
train_episodes = 1_000
learning_rate = 3e-4
buffer_size = 50_000

env, env_params, actions = make_env(n_actions)
replay_buffer = ReplayBuffer(buffer_size=buffer_size)
jit_rollout = jax.jit(rollout, static_argnums=(1, 2, 6))

key = jr.PRNGKey(42)

key, key_network = jr.split(key)
q_net = QNetwork(n_actions)
params = q_net.init(key, jnp.ones((state_dim,)))
target_params = params

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=learning_rate)
)
opt_state = optimizer.init(params)


def dqn_loss(params, target_params, batch, gamma):
    q_values = q_net.apply(params, batch["state"])
    q_action = jnp.take_along_axis(q_values, batch["action"][..., None], axis=1).squeeze()

    next_q = q_net.apply(target_params, batch["next_state"])
    next_q_max = jnp.max(next_q, axis=1)

    target = batch["reward"] + gamma * next_q_max * (1.0 - batch["done"])
    #loss = jnp.mean((q_action - target) ** 2)
    loss = jnp.mean(optax.huber_loss(q_action, target, delta=1.0))

    return loss

def update_target(params, target_params, tau=1.0):
    return jax.tree_util.tree_map(
        lambda p, tp: tau * p + (1 - tau) * tp,
        params,
        target_params,
    )

# Visualize before training
# key, key_eval = jr.split(key)
# traj_post = jit_rollout(key_eval, env, q_net, actions, params, env_params, 200, epsilon=1.0)
# visualize_trajectory(traj_post)

@jax.jit
def train_step(params, target_params, opt_state, batch, gamma):
    loss, grads = jax.value_and_grad(dqn_loss)(params, target_params, batch, gamma)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss


for episode in range(train_episodes):
    key, key_rollout = jr.split(key)
    collect_rollout(key_rollout, jit_rollout, env, q_net, params, env_params, actions, replay_buffer, epsilon)

    if len(replay_buffer) > batch_size:
        for _ in range(min(4, len(replay_buffer) // batch_size)):
            batch = replay_buffer.sample(batch_size)
            params, opt_state, loss = train_step(
                params, target_params, opt_state, batch, gamma
            )
    
    if episode % 100 == 0:
        target_params = params

    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if episode % 20 == 0:
        print(f"episode: {episode}, loss: {loss:.4f}")

# Visualize after training
key, key_eval = jr.split(key)
traj_post = jit_rollout(key_eval, env, q_net, actions, params, env_params, 200, epsilon=0.0)
visualize_trajectory(traj_post)