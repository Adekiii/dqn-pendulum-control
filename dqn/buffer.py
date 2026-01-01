from typing import NamedTuple
from collections import deque
import random
import jax.numpy as jnp


class Transition(NamedTuple):
    state: list
    action: int
    reward: float
    next_state: list
    done: bool


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
    
    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = Transition(*zip(*random.sample(self.buffer, batch_size)))

        state = jnp.array(batch.state, dtype=jnp.float32)
        action = jnp.array(batch.action, dtype=jnp.int32)
        reward = jnp.array(batch.reward, dtype=jnp.float32)
        next_state = jnp.array(batch.next_state, dtype=jnp.float32)
        done = jnp.array(batch.done, dtype=jnp.float32)

        return {
            "state": state, 
            "action": action, 
            "reward": reward, 
            "next_state": next_state, 
            "done": done
        }

    def __len__(self):
        return len(self.buffer)