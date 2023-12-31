from dataclasses import dataclass


@dataclass
class Config:
    gamma: float
    gamma_min: float
    gamma_max: float
    beta_min: float
    beta_max: float
    adaptive_c: float
    batch_size: int
    lr: float
    initial_exploration: int
    log_interval: int
    update_target: int
    replay_memory_capacity: int
    device: str
    sequence_length: int
    burn_in_length: int
    eta: float
    local_mini_batch: int
    n_step: int
    over_lapping_length: int
    epsilon_decay: float
    random_seed: int
    enable_ngu: bool
    hidden_size: int
    enable_rnd: bool
    enable_adaptive_discount: bool


config = Config(
    gamma=0.99,
    gamma_min=0.9,
    gamma_max=0.997,
    beta_min=0.,
    beta_max=1e-4,
    adaptive_c=1e-3,
    batch_size=32,
    lr=0.001,
    initial_exploration=1000,
    log_interval=10,
    update_target=1000,
    replay_memory_capacity=1000,
    device="cpu",
    sequence_length=32,
    burn_in_length=4,
    eta=0.9,
    local_mini_batch=8,
    n_step=2,
    over_lapping_length=16,
    epsilon_decay=0.00001,
    random_seed=45, # was 42
    enable_ngu=True,
    hidden_size=16,
    enable_rnd=True,
    enable_adaptive_discount=True,
)
