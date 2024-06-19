import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.sac import CnnPolicy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.evaluation import evaluate_policy
from carla_env import CarlaEnv
from gymnasium.spaces import Discrete
import sys
import os
import argparse
import torch
import glob
import sys
#from stable_baselines3.common.buffers import PrioritizedReplayBuffer

def main(model_name, load_model, town, fps, im_width, im_height, repeat_action, start_transform_type, sensors, 
         enable_preview, enable_spectator, steps_per_episode, seed=7, action_type='fix_throttle'):

    env = CarlaEnv(town, fps, im_width, im_height, repeat_action, start_transform_type, sensors,
                   action_type, enable_preview, enable_spectator, steps_per_episode, playing=False, enable_trailer=True)
    # test_env = CarlaEnv(town, fps, im_width, im_height, repeat_action, start_transform_type, sensors,
    #                action_type, enable_preview=True, enable_spectator=True, steps_per_episode=steps_per_episode, playing=True)

    # try:
    if load_model:
        device = torch.device(f"cuda:{torch.cuda.device_count()-1}") if torch.cuda.is_available() else torch.device("cpu")
        #buffer = PrioritizedReplayBuffer(buffer_size=500000, alpha=0.6) #prioritized exp replay buffer, uncomment buffer_size if this is commented
        model = SAC.load(
            model_name, 
            env,
            #action_noise=NormalActionNoise(mean=np.array([-0.1]), sigma=np.array([0.2])), #throttle max=0.5, brake max = -0.7                               
            action_noise=NormalActionNoise(mean=np.array([0]), sigma=np.array([0.2])), #only steering, sigma = 0.1 before
            #action_noise=NormalActionNoise(mean=np.array([0.3, 0.0]), sigma=np.array([0.5, 0.1])),
            device=device,
            buffer_size=500000,
            #replay_buffer= buffer,
            batch_size=256
            ) #defaul batch size,buffer size if not
    else:
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device(f"cuda:{torch.cuda.device_count()-1}") if torch.cuda.is_available() else torch.device("cpu")
        model = SAC(
            CnnPolicy, 
            env, 
            verbose=2,
            seed=seed, 
            device=device, 
            tensorboard_log='./logs/sem_sac',
            action_noise=NormalActionNoise(mean=np.array([0]), sigma=np.array([0.2])), #only steering
            #action_noise=NormalActionNoise(mean=np.array([0.3, 0]), sigma=np.array([0.5, 0.1])),
            buffer_size=500000,
            batch_size=256
            )
            
            

    print(model.__dict__)
    evaluate = False
    try:
        model.learn(
            total_timesteps=20000, 
            log_interval=4, 
            tb_log_name=model_name)
        
        if evaluate:
            mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=10) 
                #eval_env=test_env, 
                #eval_freq=1000, 
                #n_eval_episodes=10
                #)
            model_path = os.path.join('models/', model_name)
            model.save(model_path) #change this location to the location with train_sac.py
    finally:
        env.close()
        if evaluate:
            test_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-name', help='name of model when saving')
    parser.add_argument('--load', type=bool, help='whether to load existing model')
    parser.add_argument('--map', type=str, default='Town04', help='name of carla map')
    parser.add_argument('--fps', type=int, default=10, help='fps of carla env')
    parser.add_argument('--width', type=int, help='width of camera observations')
    parser.add_argument('--height', type=int, help='height of camera observations')
    parser.add_argument('--repeat-action', type=int, help='number of steps to repeat each action')
    parser.add_argument('--start-location', type=str, help='start location type: [random, highway] for Town04')
    parser.add_argument('--sensor', action='append', type=str, help='type of sensor (can be multiple): [rgb, semantic]')
    parser.add_argument('--preview', action='store_true', help='whether to enable preview camera')
    parser.add_argument('--spectator', action='store_true', help='whether to enable spectator camera')
    parser.add_argument('--episode-length', type=int, help='maximum number of steps per episode')
    parser.add_argument('--seed', type=int, default=7, help='random seed for initialization')
    parser.add_argument('--action-type', type=str, choices=['fixed_throttle', 'lateral_purepursuit', 'continuous', 'discrete'], default='fixed_throttle', help='type of action space')
    
    args = parser.parse_args()
    model_name = args.model_name
    load_model = args.load
    town = args.map
    fps = args.fps
    im_width = args.width
    im_height = args.height
    repeat_action = args.repeat_action
    start_transform_type = args.start_location
    sensors = args.sensor
    enable_preview = args.preview
    enable_spectator = args.spectator
    steps_per_episode = args.episode_length
    seed = args.seed
    action_type = args.action_type

    main(model_name, load_model, town, fps, im_width, im_height, repeat_action, start_transform_type, sensors, enable_preview, enable_spectator, steps_per_episode, seed,
         action_type=action_type)