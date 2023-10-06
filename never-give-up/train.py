import numpy as np
import torch
import torch.optim as optim
import wandb
from gym import Wrapper
from gym_maze.envs.maze_env import MazeEnvSample10x10, MazeEnvRandom20x20Plus, MazeEnvRandom10x10Plus, MazeEnvRandom30x30Plus
from torch.nn import init

from config import config
from embedding_model import EmbeddingModel# , compute_intrinsic_reward
from memory import Memory, LocalBuffer
from model import R2D2
from model_rnd import R2D2_RND


def get_action(state, target_net, epsilon, env, hidden):
    action, hidden = target_net.get_action(state, hidden)

    if np.random.rand() <= epsilon:
        return env.action_space.sample(), hidden
    else:
        return action, hidden


def update_target_model(online_net, target_net):
    target_net.load_state_dict(online_net.state_dict())


class Maze(Wrapper):
    def step(self, action: int):
        obs, rew, done, info = super().step(["N", "E", "S", "W"][action])
        self.set.add((obs[0], obs[1]))
        if rew > 0:
            rew = 10
        return obs / 10, rew, done, info

    def reset(self):
        self.set = set()
        return super().reset()


def main():
    env = Maze(MazeEnvRandom20x20Plus())#Maze(MazeEnvSample10x10())

    torch.manual_seed(config.random_seed)
    env.seed(config.random_seed)
    np.random.seed(config.random_seed)
    env.action_space.seed(config.random_seed)

    wandb.init(project="ngu-maze", config=config.__dict__)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print("state size:", num_inputs)
    print("action size:", num_actions)

    online_net = R2D2(num_inputs, num_actions)
    target_net = R2D2(num_inputs, num_actions)
    update_target_model(online_net, target_net)
    embedding_model = EmbeddingModel(obs_size=num_inputs, num_outputs=num_actions, use_rnd=config.enable_rnd)
    embedding_loss = 0
    embedding_model.to(config.device)

    optimizer = optim.Adam(online_net.parameters(), lr=config.lr)

    online_net.to(config.device)
    target_net.to(config.device)
    online_net.train()
    target_net.train()

    ################# New for AERL ####################
    if config.enable_adaptive_discount:
        online_net_2 = R2D2(num_inputs, num_actions)
        target_net_2 = R2D2(num_inputs, num_actions)
        update_target_model(online_net_2, target_net)
        
        optimizer_2 = optim.Adam(online_net_2.parameters(), lr=config.lr)
        
        online_net_2.to(config.device)
        target_net_2.to(config.device)
        online_net_2.train()
        target_net_2.train()
    ###################################################



    memory = Memory(config.replay_memory_capacity)
    epsilon = 1.0
    steps = 0
    loss = 0
    local_buffer = LocalBuffer()
    sum_reward = 0
    sum_augmented_reward = 0
    sum_obs_set = 0

    if config.enable_adaptive_discount:
        gamma_1 = config.gamma_min
        gamma_2 = config.gamma_max
        delta_tm1 = 0
    beta = 0.0001
    ################# New for AERL ####################
    if config.enable_rnd:
        if not config.enable_adaptive_discount:
            beta = 0.00005
        else:
            beta = config.beta_max
    ###################################################


    for episode in range(2500):
        done = False
        state = env.reset()
        state = torch.Tensor(state).to(config.device)

        hidden = (
            torch.Tensor().new_zeros(1, 1, config.hidden_size).to(config.device),
            torch.Tensor().new_zeros(1, 1, config.hidden_size).to(config.device),
        )

        episodic_memory = [embedding_model.embedding(state).to(config.device)]
        td_error_memory = []

        episode_steps = 0
        horizon = 100
        while not done:
            steps += 1
            episode_steps += 1

            action, new_hidden = get_action(state, target_net, epsilon, env, hidden)

            next_state, env_reward, done, _ = env.step(action)
            next_state = torch.Tensor(next_state).to(config.device)

            augmented_reward = env_reward
            if config.enable_ngu:
                next_state_emb = embedding_model.embedding(next_state)
                # print("embedding dimensions: ", next_state_emb.shape)
                intrinsic_reward, lifelong_intrinsic_reward = embedding_model.compute_intrinsic_reward(episodic_memory, next_state_emb)
                episodic_memory.append(next_state_emb)
                
                # TODO: scale intrinsic reward by alpha (lifelong factor)
                augmented_reward = env_reward + beta * intrinsic_reward

            mask = 0 if done else 1

            # May want to do local push using custom gamma
            local_buffer.push(state, next_state, action, augmented_reward, mask, hidden, gamma_1 if config.enable_adaptive_discount else None)
            hidden = new_hidden
            if len(local_buffer.memory) == config.local_mini_batch:
                batch, lengths = local_buffer.sample()
                ################# New for AERL ####################
                # if adaptive gamma is enabled, compute the TD error using the two gammas and decide which way to adjust
                if config.enable_adaptive_discount:
                    # print("Adjusting discount factors")
                    td_error_1 = R2D2.get_td_error(online_net, target_net, batch, lengths, gamma_1)
                    td_error_2 = R2D2.get_td_error(online_net_2, target_net_2, batch, lengths, gamma_2)
                    delta_t = td_error_1.pow(2).sum().item()
                    sigma_t = 1 + abs(delta_tm1 - delta_t) / delta_t
                    # IMPORTANT: get_td_error() is 
                    # predicted v(s) - actual rewards/values G_t
                    # NOT the other way around. That means we need to flip the signs on the
                    # comparisons, since the expression in Kim et al. is Q(s, a) - V(s)
                    td_error_1_sum = td_error_1.sum().item()
                    td_error_2_sum = td_error_2.sum().item()
                    if gamma_1 >= gamma_2:
                        pass
                    elif td_error_1_sum > td_error_2_sum:
                        delta_gamma = config.adaptive_c * (1 / sigma_t)
                        gamma_1 += delta_gamma
                        beta -= (delta_gamma / (config.gamma_max - config.gamma_min)) * (config.beta_max - config.beta_min)
                    elif td_error_1_sum < td_error_2_sum:
                        gamma_2 -= config.adaptive_c * (1 / sigma_t)
                    
                    delta_tm1 = delta_t
                    td_error = td_error_1
                    
                else:
                    td_error = R2D2.get_td_error(online_net, target_net, batch, lengths)
                memory.push(td_error, batch, lengths)
                ###################################################

            sum_reward += env_reward
            state = next_state
            sum_augmented_reward += augmented_reward

            if steps > config.initial_exploration and len(memory) > config.batch_size:
                epsilon -= config.epsilon_decay
                epsilon = max(epsilon, 0.1)

                batch, indexes, lengths = memory.sample(config.batch_size)
                loss, td_error = R2D2.train_model(online_net, target_net, optimizer, batch, lengths, gamma_1 if config.enable_adaptive_discount else None)
                ################# New for AERL ####################
                if config.enable_adaptive_discount:
                    # We don't need to save the result of the loss/td error, just 
                    # need to update the online and target networks
                    R2D2.train_model(online_net_2, target_net_2, optimizer_2, batch, lengths, gamma_2) 
                if config.enable_ngu:
                    embedding_loss = embedding_model.train_model(batch)
                ###################################################
                memory.update_priority(indexes, td_error, lengths)

                if steps % config.update_target == 0:
                    update_target_model(online_net, target_net)
                    if config.enable_adaptive_discount:
                        update_target_model(online_net_2, target_net_2)

            if episode_steps >= horizon or done:
                sum_obs_set += len(env.set)
                break

        if episode > 0 and episode % config.log_interval == 0:
            mean_reward = sum_reward / config.log_interval
            mean_augmented_reward = sum_augmented_reward / config.log_interval
            metrics = {
                "episode": episode,
                "mean_reward": mean_reward,
                "epsilon": epsilon,
                "embedding_loss": embedding_loss,
                "loss": loss,
                "mean_augmented_reward": mean_augmented_reward,
                "steps": steps,
                "sum_obs_set": sum_obs_set / config.log_interval,
                # "rnd_loss": lifelong_intrinsic_reward,
                "gamma_1": gamma_1,
                "gamma_2": gamma_2,
                "td_error_1 (sum)": td_error_1_sum,
                "td_error_2 (sum)": td_error_2_sum,
                "delta_t": delta_t,
                "sigma_t": sigma_t,
                "beta": beta,
            }
            print(metrics)
            wandb.log(metrics)

            sum_reward = 0
            sum_augmented_reward = 0
            sum_obs_set = 0


if __name__ == "__main__":
    main()
