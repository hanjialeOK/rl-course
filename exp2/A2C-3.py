import argparse
import datetime
import os
import random
import time
import gym
import tensorflow as tf
import torch
import torch.nn as nn


def preprocess(image):
    """ Pre-process 210x160x3 uint8 frame into 6400 (80x80) 1D float vector. """

    image = torch.Tensor(image)

    # Crop, downsample by factor of 2, and turn to grayscale by keeping only red channel
    image = image[35:195]
    image = image[::2,::2, 0]

    image[image == 144] = 0 # erase background (background type 1)
    image[image == 109] = 0 # erase background (background type 2)
    image[image != 0] = 1 # everything else (paddles, ball) just set to 1

    return image.flatten().float()


def calc_discounted_future_rewards(rewards, discount_factor):
    r"""
    Calculate the discounted future reward at each timestep.

    discounted_future_reward[t] = \sum_{k=1} discount_factor^k * reward[t+k]

    """

    discounted_future_rewards = torch.empty(len(rewards))

    # Compute discounted_future_reward for each timestep by iterating backwards
    # from end of episode to beginning
    discounted_future_reward = 0
    for t in range(len(rewards) - 1, -1, -1):
        # If rewards[t] != 0, we are at game boundary (win or loss) so we
        # reset discounted_future_reward to 0 (this is pong specific!)
        if rewards[t] != 0:
            discounted_future_reward = 0

        ### TODO: calculated discounted_future_reward at each timestep
        discounted_future_reward = rewards[t] + discount_factor * discounted_future_reward
        discounted_future_rewards[t] = discounted_future_reward

    return discounted_future_rewards



class ActorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.actor(x))


class CriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.critic(x)


def run_episode(actor_model, critic_model, env, discount_factor, render=False):
    UP = 2
    DOWN = 3

    observation = env.reset()
    prev_x = preprocess(observation)

    action_chosen_log_probs = []
    values = []
    rewards = []

    done = False

    while not done:
        if render:
            # Render game window at 30fps
            time.sleep(1 / 30)
            env.render()

        # Preprocess the observation, set input to network to be difference
        # image between frames
        cur_x = preprocess(observation)
        x = cur_x - prev_x
        prev_x = cur_x

        # Run the policy network and sample action from the returned probability
        prob_up = actor_model(x)

        ### Sample an action and then calculate the log probability of sampling
        ### the action that ended up being chosen. Then append to `action_chosen_log_probs`.
        action = UP if random.random() < prob_up else DOWN
        action_chosen_log_probs.append(torch.log(prob_up if action == UP else (1 - prob_up)))

        # Run the value network
        values.append(critic_model(x))

        # Step the environment, get new measurements, and updated discounted_reward
        observation, reward, done, _ = env.step(action)
        rewards.append(torch.Tensor([reward]))
    
    action_chosen_log_probs = torch.cat(action_chosen_log_probs).to(device)
    rewards = torch.cat(rewards).to(device)
    values = torch.cat(values).to(device)

    # Calculate the discounted future reward at each timestep
    discounted_future_rewards = calc_discounted_future_rewards(rewards, discount_factor).to(device)

    # Standardize the rewards to have mean 0, std. deviation 1 (helps control the gradient estimator variance).
    # It causes roughly half of the actions to be encouraged and half to be discouraged, which
    # is helpful especially in beginning when +1 reward signals are rare.
    discounted_future_rewards = (discounted_future_rewards - discounted_future_rewards.detach().mean()) \
                                    / discounted_future_rewards.detach().std()
    advantage = discounted_future_rewards - values
    # value loss
    critic_loss = advantage.pow(2).sum()
    # policy loss
    actor_loss = -(advantage.detach() * action_chosen_log_probs).sum()

    return critic_loss, actor_loss, rewards.sum()


def train(args, render=False):
    # Hyperparameters
    input_size = 80 * 80 # input dimensionality: 80x80 grid
    hidden_size = 200 # number of hidden layer neurons
    learning_rate = 7e-4
    discount_factor = 0.99 # discount factor for reward

    batch_size = 4
    save_every_batches = 5

    # Create policy network
    actor_model = ActorNetwork(input_size, hidden_size)
    # Create value network
    critic_model = CriticNetwork(input_size, hidden_size)

    # Load model weights and metadata from checkpoint if exists
    if os.path.exists('./checkpoint/{}-checkpoint.pth'.format(args.file_name)):
        print('Loading from checkpoint...')
        save_dict = torch.load('./checkpoint/{}-checkpoint.pth'.format(args.file_name))
        actor_model.load_state_dict(save_dict['actor_model_weights'])
        critic_model.load_state_dict(save_dict['critic_model_weights'])
        start_time = save_dict['start_time']
        last_batch = save_dict['last_batch']
    else:
        start_time = datetime.datetime.now().strftime("%H.%M.%S-%m.%d.%Y")
        last_batch = -1

    actor_optimizer = torch.optim.Adam(actor_model.parameters(), lr=learning_rate)
    critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=learning_rate)
    # Set up tensorboard logging
    tf_writer = tf.summary.create_file_writer(
        os.path.join('tensorboard_logs', ''.join(('{}-'.format(args.file_name), start_time))))
    tf_writer.set_as_default()

    # Create pong environment (PongDeterministic versions run faster)
    # Episodes consist of a series of games until one player has won 20 times.
    # A game ending in a win yields +1 reward and a game ending in a loss gives -1 reward.
    #
    # The RL agent (green paddle) plays against a simple AI (tan paddle) that
    # just tries to track the y-coordinate of the ball.
    env = gym.make("PongDeterministic-v4")

    # Pick up at the batch number we left off at to make tensorboard plots nicer
    batch = last_batch + 1
    while True:
        
        mean_batch_critic_loss = 0
        mean_batch_actor_loss = 0
        mean_batch_reward = 0

        for batch_episode in range(batch_size):

            # Run one episode
            critic_loss, actor_loss,episode_reward = run_episode(actor_model, critic_model, env, discount_factor, render)
            mean_batch_critic_loss += critic_loss / batch_size
            mean_batch_actor_loss += actor_loss / batch_size
            mean_batch_reward += episode_reward / batch_size

            # Boring book-keeping
            print(f'Episode reward total was {episode_reward}')

        # Backprop after `batch_size` episodes
        critic_optimizer.zero_grad()
        mean_batch_critic_loss.backward()
        critic_optimizer.step()

        actor_optimizer.zero_grad()
        mean_batch_actor_loss.backward()
        actor_optimizer.step()

        # Batch metrics and tensorboard logging
        print(f'Batch: {batch}, mean loss: {mean_batch_actor_loss:.2f}, '
              f'mean reward: {mean_batch_reward:.2f}')
        tf.summary.scalar('mean loss', mean_batch_actor_loss.detach().item(), step=batch)
        tf.summary.scalar('mean reward', mean_batch_reward.detach().item(), step=batch)

        if batch % save_every_batches == 0:
            print('Saving checkpoint...')
            save_dict = {
                'actor_model_weights': actor_model.state_dict(),
                'critic_model_weights': critic_model.state_dict(),
                'start_time': start_time,
                'last_batch': batch
            }
            torch.save(save_dict, './checkpoint/{}-checkpoint.pth'.format(args.file_name))

        batch += 1

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def main():
    # By default, doesn't render game screen, but can invoke with `--render` flag on CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--file_name', type=str, default='A2C-3')
    if not os.path.exists('./checkpoint'):
        os.mkdir("./checkpoint")
    args = parser.parse_args()

    train(args, render=args.render)


if __name__ == '__main__':
    main()
