from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import jax.numpy as jnp


def visualize_trajectory(traj, discrete_actions=None):
    obs, actions, rewards, next_obs, done = traj
    T = obs.shape[0]
    
    cos_theta = obs[:, 0]
    sin_theta = obs[:, 1]
    theta_dot = obs[:, 2]
    u = actions.squeeze()
    r = rewards
    
    t = jnp.arange(T)

    fig, axes = plt.subplots(5, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(t, cos_theta, color="C0")
    axes[0].set_title(r'$\cos(\theta)$')

    axes[1].plot(t, sin_theta, color="C0")
    axes[1].set_title(r'$\sin(\theta)$')
    
    axes[2].plot(t, theta_dot, color="C0")
    axes[2].set_title(r'$\dot{\theta}$')
    
    axes[3].plot(t, u, color="C1")
    axes[3].set_title(r"$u(t)$")
    if discrete_actions is not None:
        axes[3].set_yticks(discrete_actions)
    
    axes[4].plot(t, r, color="C2")
    axes[4].set_title(r"$r(t)$")

    fig.tight_layout()
    #plt.show()

    folder_path = Path(__file__).resolve().parent.parent / "figures"
    now = datetime.now()
    filename = "result_" + now.strftime("%d-%m, %H %M %S") + ".png"
    fig.savefig(folder_path / filename)