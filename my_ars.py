#%%
""" 
An implementation for Augmented Radom Search algorithm solivng 2d open ai gym enviroments - tested for Box2d envs.
Part of Move37 course - chapter 5-2
Requires:
openai gym (pip install gym)
swig.exe : http://www.swig.org/  (add path)
ffmpeg : https://www.ffmpeg.org/download.html (add path)
box2d (pip install box2d-py)
written in python 3.6.7 using a jupyter server in VS Code

"""

import os
import numpy as np
import gym
from gym import wrappers

class Hp():
    """ 
    Class forstoring all hyper parameters
    """
    def __init__(self,
                 nb_steps=1000,
                 episode_length=2000,
                 learning_rate=0.02,
                 num_deltas=32,
                 num_best_deltas=8,
                 noise=0.03,
                 seed=1,
                 env_name='BipedalWalker-v2',
                 record_every=50):

        self.nb_steps = nb_steps #how many episodes
        self.episode_length = episode_length # how many steps are in an episode
        self.learning_rate = learning_rate #alpha, the learning rate
        self.num_deltas = num_deltas # count of parallel deltas
        self.num_best_deltas = num_best_deltas # count of how many of the best results form the different deltas to use
        assert self.num_best_deltas <= self.num_deltas
        self.noise = noise # the noise magnitude
        self.seed = seed # random seed for reproduceability
        self.env_name = env_name
        self.record_every = record_every

class Normalizer():
    """ 
    Class for Normalization actions on the states
    """
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std

class Policy():
    """ 
    This class will turn inputs into actions. 
    Methods:
        evaluate : return the dot product of input and weights (theta), possibly with noise (delta)
        sample_delta : get the right amount of deltas from a normal distribution 
        update : 

    """

    def __init__(self, input_size, output_size, hp):
        self.theta = np.zeros((output_size, input_size))
        self.hp = hp

    def evaluate(self, input, delta = None, direction = None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == "+":
            return (self.theta + self.hp.noise * delta).dot(input)
        elif direction == "-":
            return (self.theta - self.hp.noise * delta).dot(input)

    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(self.hp.num_deltas)]

    def update(self, rollouts, sigma_rewards):
        """ 
        Update the weights based on our rollouts
        Args: 
            rollouts: touple with reward_pos, reward_neg, delta
            sigma_rewards: stddev of the rewards. Needed to de-normalize
        Return: None
        """
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, delta in rollouts:
          step += (r_pos - r_neg) * delta
          #de-normalize
          self.theta += self.hp.learning_rate / (self.hp.num_best_deltas * sigma_rewards) * step




class ArsTrainer():
    def __init__(self,
                    hp=None,
                    input_size = None,
                    output_size = None,
                    normalizer = None,
                    policy = None,
                    monitor_dir = None):
        self.hp = hp or Hp()
        np.random.seed(self.hp.seed)
        self.env = gym.make(self.hp.env_name)
        if monitor_dir is not None:
            should_record = lambda i: self.record_video
            self.env = wrappers.Monitor(self.env, monitor_dir, video_callable=should_record, force=True)
        self.hp.episode_length = self.env.spec.timestep_limit or self.hp.episode_length
        self.input_size = input_size or self.env.observation_space.shape[0]
        self.output_size = output_size or self.env.action_space.shape[0]
        self.normalizer = normalizer or Normalizer(self.input_size)
        self.policy = policy or Policy(self.input_size, self.output_size, self.hp)
        self.record_video = False

    def explore(self,direction = None, delta = None):
        #test a policy on a specific direction over one episode
        state = self.env.reset()
        done = False
        num_plays = 0.0
        sum_rewads = 0.0
        while not done and num_plays < self.hp.episode_length:
            self.normalizer.observe(state)
            state = self.normalizer.normalize(state) #norm the state
            action = self.policy.evaluate(state,delta,direction) #get action
            state,reward,done,_ = self.env.step(action)#make a step and get new state,reward
            reward = max(min(reward,1),-1)
            sum_rewads += reward #the final score
            num_plays += 1
        return sum_rewads

    def train(self):
        for step in range(self.hp.nb_steps):
            #make random noise deltas and empty lists for pos and neg rewards
            deltas = self.policy.sample_deltas()
            rewards_pos = [0] * self.hp.num_deltas
            rewards_neg = [0] * self.hp.num_deltas
            
            #play an empisode each with positive and negative deltas
            for i in range(self.hp.num_deltas):
                rewards_pos[i] = self.explore(direction='+',delta=deltas[i])
                rewards_neg[i] = self.explore(direction='-',delta=deltas[i])
            
            #compute the stddev for all rewards
            rewards_stddev = np.array(rewards_pos + rewards_neg).std()

            #sort the rollouts by the max(reward_pos,reward_neg) and select the deltas with the best rewards
            #use sorted but store position using a dict
            scores = {i:max(ipos,ineg) for i,(ipos,ineg) in enumerate(zip(rewards_pos,rewards_neg))}
            sorted_and_cut_indexes = sorted(scores.keys(),key = lambda i:scores[i],reverse = True)[:self.hp.num_best_deltas]
            rollouts = [(rewards_pos[i],rewards_neg[i],deltas[i]) for i in sorted_and_cut_indexes]

            #update policy
            self.policy.update(rollouts,rewards_stddev)

            # Only record video during evaluation, every n steps
            if step % self.hp.record_every == 0:
                self.record_video = True
            
            # Play an episode with the new weights and print the score
            reward_evaluation = self.explore()
            print(f'Step: {step} Reward: {reward_evaluation}')
            self.record_video = False

    

def mkdir(base,name):
    path = os.path.join(base,name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


#main. if run not in a notebook use: if __name__ == __main__:
ENV_NAME = 'BipedalWalker-v2'
videos_dir = mkdir('.','videos')
moritor_dir = mkdir(videos_dir,ENV_NAME)
hp =  Hp(seed=1,env_name = ENV_NAME)
trainer = ArsTrainer(hp=hp, monitor_dir=moritor_dir)
trainer.train()

