import atexit
import os
import signal
import sys
import carla
import gymnasium as gym
import time
import random
import numpy as np
import math
from queue import Queue
# from misc import dist_to_roadline, exist_intersection
from gymnasium import spaces
from gymnasium.spaces import Discrete
from setup import setup
from absl import logging
import graphics
import pygame
import subprocess
import glob

logging.set_verbosity(logging.DEBUG)

# Carla environment
class CarlaEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, town, fps, im_width, im_height, repeat_action, start_transform_type, sensors,
                 action_type, enable_preview, enable_spectator, steps_per_episode, playing=False, timeout=10,
                 enable_trailer=True, soft_lane_invasion=True):

        #self.client, self.world, self.frame, self.server = setup(town=town, fps=fps, client_timeout=timeout)
        self.client, self.world, self.frame, self.original_sim_settings = setup(town=town, fps=fps, client_timeout=timeout, num_max_restarts=5)
        self.client.set_timeout(5.0)
        self.map = self.world.get_map()
        blueprint_library = self.world.get_blueprint_library()
        # self.vehicle_bp = blueprint_library.filter('vehicle.carlamotors.firetruck')[0] #vehicle set here
        # self.vehicle_bp = blueprint_library.filter('vehicle.dodge.charger_2020')[0]
        self.vehicle_bp = blueprint_library.filter('vehicle.daf.dafxf')[0] #vehicle set here
        self.trailer_bp = blueprint_library.filter('vehicle.trailer.trailer')[0] if enable_trailer else None
        self.im_width = im_width
        self.im_height = im_height
        self.repeat_action = repeat_action
        self.action_type = action_type
        self.start_transform_type = start_transform_type
        self.sensors = sensors
        self.actor_list = []
        self.preview_camera = None        
        self.steps_per_episode = steps_per_episode
        self.playing = playing
        self.preview_camera_enabled = enable_preview
        self.spectator_view = enable_spectator #added1
        self.spectator = None # self.world.get_spectator() if enable_spectator else None
        self.traffic_manager = self.client.get_trafficmanager() #added for pure pursuit
        self.traffic_manager.global_percentage_speed_difference(60.0)  # Vehicles move at 40% of their top speed
        
        self.soft_lane_invasion = soft_lane_invasion


        # self.episode = 0
        #self.spawn_traffic() #comment for no traffic

        #comment for render-mode on
        # config_script_path = os.path.join(carla.__path__[0].split('PythonAPI')[0], 'PythonAPI/util/config.py') # "/home/aku8wk/Carla/CARLA_0.9.15/PythonAPI/util/config.py"
        # try:
        #     logging.debug("Running config script to disable rendering: {}".format(config_script_path))
        #     subprocess.run([config_script_path, "--no-rendering"], check=True)
        # except subprocess.CalledProcessError as e:
        #     logging.error("Failed to run config script: {}".format(e))
        #     sys.exit(1)
        settings = self.world.get_settings()
        settings.no_rendering_mode = not enable_spectator
        self.world.apply_settings(settings)



    @property
    def observation_space(self, *args, **kwargs):
        """Returns the observation spec of the sensor."""
        return gym.spaces.Box(low=0.0, high=255.0, shape=(self.im_height, self.im_width, 3), dtype=np.uint8)

    @property
    def action_space(self):
        """Returns the expected action passed to the `step` method."""
        if self.action_type == 'continuous': #element 0-throttle/brake, 1- steering
            #return gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]))
            return gym.spaces.Box(low=np.array([0.2, -0.3]), high=np.array([0.25, 0.3])) # sac_car_rgb4 model
        elif self.action_type == 'fix_throttle':
            #return gym.spaces.Box(low=np.array([-0.3]), high=np.array([0.3])) #only steering. sac_car_steer3
            return gym.spaces.Box(low=np.array([-0.6]), high=np.array([0.6])) #only steering. sac_car_steer6
        elif self.action_type == 'lateral_purepursuit':
            #return gym.spaces.Box(low=np.array([-0.7]), high=np.array([0.7])) #throttle and brake. sac_car_purep7
            return gym.spaces.Box(low=np.array([-0.7]), high=np.array([0.5])) #throttle and brake. sac_car_purep5
        elif self.action_type == 'discrete':
            return gym.spaces.MultiDiscrete([9, 9])
        else:
            raise NotImplementedError()
        

    def seed(self, seed):
        if not seed:
            seed = 7
        random.seed(seed)
        self._np_random = np.random.RandomState(seed) 
        return seed

    # Resets environment for new episode
    def reset(self, seed=None):
        logging.debug("Resetting environment ... ")
        self._destroy_agents()
        self.actor_list = []

        if self.action_type == 'lateral_purepursuit': #traffic spawning for throttle/brake learning
            num_veh = 30
            for _ in range(0,num_veh):
                traffic_loc = random.choice(self.map.get_spawn_points())
                bp_traffic = random.choice(self.world.get_blueprint_library().filter('vehicle'))
                npc = self.world.try_spawn_actor(bp_traffic, traffic_loc)
                if npc is not None:
                    self.actor_list.append(npc)
                    npc.set_autopilot(True, self.traffic_manager.get_port()) # Control by traffic manager
                    self.traffic_manager.ignore_lights_percentage(npc, 0)  # Follow traffic lights
                    self.traffic_manager.ignore_signs_percentage(npc, 0)  # Follow traffic signs
                    self.traffic_manager.auto_lane_change(npc, True)  # Allow lane changing
                    self.traffic_manager.distance_to_leading_vehicle(npc, 3) # Maintain distance
                    self.traffic_manager.random_left_lanechange_percentage(npc,0)
                    self.traffic_manager.random_right_lanechange_percentage(npc,0)
                    self.traffic_manager.set_desired_speed(npc,20)
                    self.traffic_manager.set_global_distance_to_leading_vehicle(3)
                    self.traffic_manager.set_hybrid_physics_mode(False)
                    #self.traffic_manager.set_hybrid_physics_radius(70.0)
        logging.debug("Traffic spawned")

        # self.collision_hist = []
        # self.lane_invasion_hist = []
        
        self.frame_step = 0
        self.out_of_loop = 0
        self.dist_from_start = 0
        # self.total_reward = 0

        self.front_image_Queue = Queue()
        self.preview_image_Queue = Queue()

        # self.episode += 1

        # When Carla breaks (stops working) or spawn point is already occupied, spawning a car throws an exception
        # allow it to try for 3 seconds then forgive
        spawn_start = time.time()
        # while True:
        
        ego_spawn_failure_cnt = 0
        max_ego_spawn_failure = 1
        while ego_spawn_failure_cnt < max_ego_spawn_failure:
            try:
                # try:
                # Get random spot from a list from predefined spots and try to spawn a car there
                self.start_transform = self._get_start_transform()
                #print(self.start_transform) #printing spawn location, can comment
                self.curr_loc = self.start_transform.location
                self.vehicle = self.world.spawn_actor(self.vehicle_bp, self.start_transform)
                # Append actor to a list of spawned actors, need to remove them after episode ends
                self.actor_list.append(self.vehicle)

                if self.trailer_bp is not None:
                    forward_vector = self.start_transform.get_forward_vector() * 5.2
                    trailer_start_transform = carla.Transform(self.start_transform.location - forward_vector, self.start_transform.rotation)
                    self.trailer = self.world.spawn_actor(self.trailer_bp, trailer_start_transform)
                    self.actor_list.append(self.trailer)
                else:
                    self.trailer = None
                
                break
            except RuntimeError as e:
                ego_spawn_failure_cnt += 1
                logging.warn(f'Error spawning ego truck: {e}. Trial {ego_spawn_failure_cnt}/{max_ego_spawn_failure}')
    
        # break
        # except Exception as e:
        #     logging.error('Error carla 141 {}'.format(str(e)))
        #     time.sleep(0.01)
        logging.debug("Ego truck spawned")

        # If that can't be done in 3 seconds - forgive (and allow main process to handle for this problem)
        # if time.time() > spawn_start + 3:
        #     raise Exception('Can\'t spawn a car')

        bound_x = self.vehicle.bounding_box.extent.x #to set the boundaries of agent- half of width & length
        bound_y = self.vehicle.bounding_box.extent.y
        bound_z = self.vehicle.bounding_box.extent.z
                

        if 'rgb' in self.sensors:
            self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
        elif 'semantic' in self.sensors:
            self.rgb_cam = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        else:
            raise NotImplementedError('unknown sensor type')

        self.rgb_cam.set_attribute('image_size_x', f'{self.im_width}')
        self.rgb_cam.set_attribute('image_size_y', f'{self.im_height}')
        #self.rgb_cam.set_attribute('fov', '90')
        self.rgb_cam.set_attribute('fov', '100')


        transform_front = carla.Transform(carla.Location(x=bound_x*1.5, y=0, z=bound_z*0.5), carla.Rotation(pitch=0)) #camera sensor position
        #transform_front = carla.Transform(carla.Location(x=bound_x*1.5, y=0, z=bound_z*0.5))
        self.sensor_front = self.world.spawn_actor(self.rgb_cam, transform_front, attach_to=self.vehicle)
        self.sensor_front.listen(self.front_image_Queue.put)
        self.actor_list.extend([self.sensor_front])

        # Preview ("above the car") camera
        if self.preview_camera_enabled:
            
            self.preview_cam = self.world.get_blueprint_library().find('sensor.camera.rgb')
            self.preview_cam.set_attribute('image_size_x', '400')
            self.preview_cam.set_attribute('image_size_y', '400')
            self.preview_cam.set_attribute('fov', '100')
            transform = carla.Transform(carla.Location(x=-5*bound_y, z=3*bound_z), carla.Rotation(pitch=6.0))
            self.preview_sensor = self.world.spawn_actor(self.preview_cam, transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.SpringArm)
            self.preview_sensor.listen(self.preview_image_Queue.put)
            self.actor_list.append(self.preview_sensor)
        
        logging.debug("Cameras spawned")
        
        if self.spectator_view:
            self.spectator = self.world.get_spectator()
            self.spectator.set_transform(carla.Transform(self.curr_loc + carla.Location(z=50), carla.Rotation(pitch=-90.0)))
            logging.debug("Spectator camera spawned")
        

        #some workarounds
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=0.0))
        work_around_frames = 80
        for _ in range(work_around_frames):
            self.world.tick()
        # time.sleep(4)

        # Collision history is a list callback is going to append to (we brake simulation on a collision)
        self.collision_hist = []
        self.lane_invasion_hist = []

        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        lanesensor = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.colsensor = self.world.spawn_actor(colsensor, carla.Transform(carla.Location(x=5.0)), attach_to=self.vehicle)
        self.vehicle.set_collisions(True)
        self.colsensor.set_collisions(True)
        self.trailer_colsensor = self.world.spawn_actor(colsensor, carla.Transform(), attach_to=self.trailer) if self.trailer is not None else None
        self.lanesensor = self.world.spawn_actor(lanesensor, carla.Transform(), attach_to=self.vehicle)
        self.colsensor.listen(lambda event: self._collision_data(event, ignore_actor_ids=set([self.trailer.id])))
        self.lanesensor.listen(lambda event: self._lane_invasion_data(event))
        if self.trailer_colsensor is not None:
            self.trailer.set_collisions(True)
            self.trailer_colsensor.set_collisions(True)
            self.trailer_colsensor.listen(lambda event: self._collision_data(event, ignore_actor_ids=set([self.vehicle.id])))
        self.actor_list.append(self.colsensor)
        if self.trailer_colsensor is not None:
            self.actor_list.append(self.trailer_colsensor)
        self.actor_list.append(self.lanesensor)
        logging.debug("Collision and lane invasion sensors spawned")

        self.world.tick()

        while self.front_image_Queue.empty():
            logging.debug("waiting for camera to be ready")
            time.sleep(0.01)
            self.world.tick()

        # Disengage brakes
        self.vehicle.apply_control(carla.VehicleControl(brake=0.0))

        image = self.front_image_Queue.get()
        image = np.array(image.raw_data)
        image = image.reshape((self.im_height, self.im_width, -1))
        image = image[:, :, :3]

        logging.debug("Environment reset done.")

        return image, {}

    def step(self, action): #executing single action, _step is defined below
        total_reward = 0
        for _ in range(self.repeat_action):
            obs, rew, terminated, truncted, info = self._step(action)
            total_reward += rew
            if terminated or truncted:
                break
        return obs, total_reward, terminated, truncted, info

    # Steps environment
    def _step(self, action):
        if self.spectator is not None:
            curr_loc = self.vehicle.get_location()
            self.spectator.set_transform(carla.Transform(curr_loc + carla.Location(z=50), carla.Rotation(pitch=-90.0)))

        self.world.tick()
        #self.render()
            
        self.frame_step += 1

        # Apply control to the vehicle based on an action
        if self.action_type == 'continuous':
            if action[0] > 0: #accelerating, set brake to 0
                action = carla.VehicleControl(throttle=float(action[0]), steer=float(action[1]), brake=0)
            else: #decelerating, set throttle to 0, take negative of first element as brake
                action = carla.VehicleControl(throttle=0, steer=float(action[1]), brake= -float(action[0]))
        
        elif self.action_type == 'fix_throttle':
            fixed_throttle = 0.1
            steering_action = action[0]
            action = carla.VehicleControl(throttle=fixed_throttle, steer=float(steering_action), brake=0)

        elif self.action_type == 'lateral_purepursuit':
            #give throttle here
            action[0] = max(0.01, action[0])
            if action[0] > 0: # Accelerating, set brake to 0
                action = carla.VehicleControl(throttle=float(action[0]), steer = self.pure_pursuit(), brake=0)
            else: # Decelerating, set throttle to 0, take negative of first element as brake
                action = carla.VehicleControl(throttle=0, steer = self.pure_pursuit(), brake=-float(action[0]))
            # action.throttle = 1.0
            # action.brake = 0.0

        elif self.action_type == 'discrete':
            #if action[0] == 0:
            #    action = carla.VehicleControl(throttle=0, steer=float((action[1] - 4)/4), brake=1)
            #else:
            #    action = carla.VehicleControl(throttle=float((action[0])/3), steer=float((action[1] - 4)/4), brake=0)
            throttle_mapping = {0: -1.0, 1: -0.75, 2: -0.5, 3: -0.25, 4: 0.0, 5: 0.25, 6: 0.5, 7: 0.75, 8: 1.0}
            steering_mapping = {0: -1.0, 1: -0.75, 2: -0.5, 3: -0.25, 4: 0.0, 5: 0.25, 6: 0.5, 7: 0.75, 8: 1.0}
            throttle_val = throttle_mapping[action[0]]
            steer_val = steering_mapping[action[1]]
            if throttle_val >0:
                action = carla.VehicleControl(throttle=float(throttle_val), steer=float(steer_val), brake=0)
            else:
                action = carla.VehicleControl(throttle=0, steer=float(steer_val), brake=float(-throttle_val))
            #brake_val = 0 if throttle_val > 0 else -throttle_val
            #action = carla.VehicleControl(throttle=throttle_val, steer=steer_val, brake=brake_val) #incorrect
        else:
            raise NotImplementedError()
        logging.debug('{}, {}, {}'.format(action.throttle, action.steer, action.brake))
        self.vehicle.apply_control(action)

        # Calculate speed in km/h from car's velocity
        v = self.vehicle.get_velocity()
        kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

        loc = self.vehicle.get_location()
        new_dist_from_start = loc.distance(self.start_transform.location)
        square_dist_diff = new_dist_from_start ** 2 - self.dist_from_start ** 2
        self.dist_from_start = new_dist_from_start

        image = self.front_image_Queue.get()
        image = np.array(image.raw_data)
        image = image.reshape((self.im_height, self.im_width, -1))

        
        if 'rgb' in self.sensors:
            image = image[:, :, :3]
        if 'semantic' in self.sensors:
            image = image[:, :, 2]
            image = (np.arange(13) == image[..., None]) #one hot encoded representation of pixels into classes
            image = np.concatenate((image[:, :, 2:3], image[:, :, 6:8]), axis=2) #select   relevant classes and concatenate
            image = image * 255 #converts image back to pxel value range
            # logging.debug('{}'.format(image.shape))
            # assert image.shape[0] == self.im_height
            # assert image.shape[1] == self.im_width
            # assert image.shape[2] == 3

        # dis_to_left, dis_to_right, sin_diff, cos_diff = dist_to_roadline(self.map, self.vehicle)

        terminated = False
        truncated = False
        reward = 0
        info = dict()

        # # If car collided - end and episode and send back a penalty
        if len(self.collision_hist) != 0:
            terminated = True
            truncated = False
            reward += -100
            self.collision_hist = [] # TODO: these should be in reset
            self.lane_invasion_hist = []
        
        if len(self.lane_invasion_hist) != 0:
            if self.soft_lane_invasion:
                reward += self._soft_lane_invasion_penalty()
            elif not self.action_type == 'lateral_purepursuit': #pure pursuit may lead to slight lane violation
                terminated = True
                truncated = False
                reward += -100
                self.collision_hist = []
                self.lane_invasion_hist = []
        
        verbose = True
        if verbose:
            # self.world.debug.draw_box(carla.BoundingBox(self.colsensor.bounding_box), self.colsensor.get_transform(), 0, carla.Color(255,0,0,0), 0.1)
            self.world.debug.draw_box(self.colsensor.bounding_box, self.colsensor.bounding_box.rotation, 0.2, carla.Color(255,0,0,0), 0.105)


        # if len(self.lane_invasion_hist) != 0:
        #     done = True
        #     reward += -100
        #     self.lane_invasion_hist = []

        # if kmh < 1:
        #     reward += -1

        # if kmh > 25: #setting max speed 25, set max speed elsewhere to directly limit velocity from throttle
        #     done = True
        #     reward += -100


        # reward += 0.1 * kmh

        reward += square_dist_diff

        # # Reward for speed
        # if not self.playing:
        #     reward += 0.1 * kmh * (self.frame_step + 1)
        # else:
        #     reward += 0.1 * kmh        

        # # Reward for distance to road lines
        # if not self.playing:
        #     reward -= math.exp(-dis_to_left)
        #     reward -= math.exp(-dis_to_right)
        
        if self.frame_step >= self.steps_per_episode:
            terminated = False
            truncated = True

        # if not self._on_highway(): #if vehicle is out of highway for more than 4sec, end episode
        #    self.out_of_loop += 1
        #    if self.out_of_loop >= 4:
        #        done = True
        # else:
        #    self.out_of_loop = 0

        # self.total_reward += reward

        if terminated or truncated:
            # info['episode'] = {}
            # info['episode']['l'] = self.frame_step
            # info['episode']['r'] = reward
            logging.debug("Env lasts {} steps, restarting ... ".format(self.frame_step))
            # self._destroy_agents()
        
        return image, reward, terminated, truncated, info
    
#    def close(self):
#        if self.carla_process.is_alive():
#            self.carla_process.terminate()
    #     logging.info("Closes the CARLA server with process PID {}".format(self.server.pid))
    #     os.killpg(self.server.pid, signal.SIGKILL)
    #     atexit.unregister(lambda: os.killpg(self.server.pid, signal.SIGKILL))
    def close(self):
        self._destroy_agents()
        self.world.apply_settings(self.original_sim_settings)



  
#    def render(self, mode='human'):

#
#        if self.preview_camera_enabled:
#
#            self._display, self._clock, self._font = graphics.setup(
            #     width=400,
            #     height=400,
            #     render=(mode=="human"),
            # )

            # preview_img = self.preview_image_Queue.get()
            # preview_img = np.array(preview_img.raw_data)
            # preview_img = preview_img.reshape((400, 400, -1))
            # preview_img = preview_img[:, :, :3]
            # graphics.make_dashboard(
            #     display=self._display,
            #     font=self._font,
            #     clock=self._clock,
            #     observations={"preview_camera":preview_img},
            # )

            # if mode == "human":
            #     # Update window display.
            #     pygame.display.flip()
            # else:
            #     raise NotImplementedError()
                

    # def spawn_traffic(self):
    #     traffic_script_path = "/home/aku8wk/Carla/CARLA_0.9.15/PythonAPI/examples/generate_traffic.py"
    #     num_vehicles = 30
    #     command = ['python3', traffic_script_path, '-n', str(num_vehicles)]      
    #     logging.debug(f"Spawning traffic using command: {' '.join(command)}")
        
    #     try:
    #         result = subprocess.run(command, check=True, capture_output=True, text=True)
    #         logging.debug("Traffic generation output: {}".format(result.stdout))
    #         traffic_ids = [int(line) for line in result.stdout.splitlines() if line.strip().isdigit()]
    #         for actor_id in traffic_ids:
    #             actor = world.get_actor(actor_id)
    #             if actor is not None:
    #                 self.actor_list.append(actor)
    #                 logging.debug(f"Added actor ID {actor_id} to actor list")
                
    #     except subprocess.CalledProcessError as e:
    #         logging.error("Failed to generate traffic: {}".format(e))
    #         raise e


    def spawn_traffic(self):
        # Command to run the generate_traffic.py script
        py_api_path = os.path.join(carla.__path__[0].split('PythonAPI')[0], 'PythonAPI')
        traffic_script_path = os.path.join(py_api_path, 'examples/generate_traffic.py')
        # traffic_script_path = "/home/aku8wk/Carla/CARLA_0.9.15/PythonAPI/examples/generate_traffic.py"
        num_vehicles = 80
        command = ['python3', traffic_script_path, '-n', str(num_vehicles)]
        
        logging.debug(f"Spawning traffic using command: {' '.join(command)}")
        
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            logging.debug("Traffic generation output: {}".format(result.stdout))
        except subprocess.CalledProcessError as e:
            logging.error("Failed to generate traffic: {}".format(e))
            raise e

    
    def _destroy_agents(self):

        time.sleep(0.1)
        for actor in self.actor_list:
            logging.debug("Destroying actor {}".format(actor))

            # If it has a callback attached, remove it first
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()

            # If it's still alive - destroy it
            if actor.is_alive:
                actor.destroy()

        self.actor_list = []

    def _soft_lane_invasion_penalty(self):
        #return -1.0
        return -1.0

    def _collision_data(self, event, ignore_actor_ids=None):
        # raise RuntimeError

        # What we collided with and what was the impulse
        self_actor_type_id = event.actor.type_id
        self_actor_id = event.actor.id
        collision_actor_type_id = event.other_actor.type_id
        collision_actor_id = event.other_actor.id
        if ignore_actor_ids is not None and collision_actor_id in ignore_actor_ids:
            logging.debug(f"Ignoring collision between {self_actor_type_id}:{self_actor_id} and {collision_actor_type_id}:{collision_actor_id}")
            return
        collision_impulse = math.sqrt(event.normal_impulse.x ** 2 + event.normal_impulse.y ** 2 + event.normal_impulse.z ** 2)

        # # Filter collisions
        # for actor_id, impulse in COLLISION_FILTER:
        #     if actor_id in collision_actor_id and (impulse == -1 or collision_impulse <= impulse):
        #         return

        # Add collision
        self.collision_hist.append(collision_impulse)
        logging.debug(f"Collision between {self_actor_type_id}:{self_actor_id} and {collision_actor_type_id}:{collision_actor_id} with impulse {collision_impulse}")
    
    def _lane_invasion_data(self, event):
        # to filter lane invasions
        #invaded_lane = event.crossed_lane
        #lane_type = event.lane_type
        # Filter lane invasions based on lane type, e.g., check for lane type change
        # For example, let's consider only lane changes
        #if lane_type == carla.LaneType.Driving:
        #    if event.old_lane_type != event.lane_type:
                # Add the lane invasion event to history
        #        self.lane_invasion_hist.append(event)
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        #self.hud.notification("Crossed line %s" % " and ".join(text))
        self.lane_invasion_hist.append(text)


    # def _on_highway(self):
    #     goal_abs_lane_id = 4
    #     vehicle_waypoint_closest_to_road = \
    #         self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
    #     road_id = vehicle_waypoint_closest_to_road.road_id
    #     lane_id_sign = int(np.sign(vehicle_waypoint_closest_to_road.lane_id))
    #     assert lane_id_sign in [-1, 1]
    #     goal_lane_id = goal_abs_lane_id * lane_id_sign
    #     vehicle_s = vehicle_waypoint_closest_to_road.s
    #     goal_waypoint = self.map.get_waypoint_xodr(road_id, goal_lane_id, vehicle_s)
    #     return not (goal_waypoint is None)

    def _get_start_transform(self):
        if self.start_transform_type == 'random':
            return random.choice(self.map.get_spawn_points())

        if self.start_transform_type == 'highway':
            if self.map.name == "Town03":
                for trial in range(10):
                    start_transform = random.choice(self.map.get_spawn_points())
                    start_waypoint = self.map.get_waypoint(start_transform.location)
                    if start_waypoint.road_id in list(range(15, 90)): 
                        break
                return start_transform
            else:
                raise NotImplementedError
        
        if self.start_transform_type== 'highway':
            if self.map.name == "Town03":
                for trial in range(15):
                    junction_location = carla.Location(x=100, y=200, z=0)
                    my_junction = self.map.get_junction(junction_location)
                    waypoints_junc = my_junction.get_waypoints()
                    lane_waypoints = waypoints_junc[0]
                    start_waypoint, end_waypoint = lane_waypoints
                    start_transform = carla.Transform(start_waypoint.location, start_waypoint.rotation)
                    destination = end_waypoint.location
                return start_transform
            else:
                raise NotImplementedError

        if self.start_transform_type== 'custom':
            #transform = carla.Transform(carla.Location(x=193.5, y=125, z=1.85), carla.Rotation(yaw=270, pitch=0, roll=0)) #town2 90deg left1
            #transform = carla.Transform(carla.Location(x=183, y=105.5, z=1.85), carla.Rotation(yaw=180, pitch=0, roll=0)) #town2 straight1
            #transform = carla.Transform(carla.Location(x=50, y=105.5, z=1.85), carla.Rotation(yaw=180, pitch=0, roll=0)) #town2 90deg left2
            #transform = carla.Transform(carla.Location(x=18, y=105.5, z=1.85), carla.Rotation(yaw=180, pitch=0, roll=0)) #town2 90deg left2
            #transform = carla.Transform(carla.Location(x=-7, y=160, z=1.85), carla.Rotation(yaw=90, pitch=0, roll=0)) #town2 90deg left3
            #transform = carla.Transform(carla.Location(x=-7, y=235, z=1.85), carla.Rotation(yaw=90, pitch=0, roll=0)) #town2 straight
            # transform = carla.Transform(carla.Location(x=-7, y=285, z=1.85), carla.Rotation(yaw=90, pitch=0, roll=0)) #town2 straight
            #transform = carla.Transform(carla.Location(x=12, y=306.5, z=1.85), carla.Rotation(yaw=0, pitch=0, roll=0)) #town2 90deg left5
            #transform = carla.Transform(carla.Location(x=160, y=306.5, z=1.85), carla.Rotation(yaw=0, pitch=0, roll=0)) #town2 90deg left6
            #transform = carla.Transform(carla.Location(x=193.5, y=270.5, z=1.85), carla.Rotation(yaw=-90, pitch=0, roll=0)) #town2 straight
            #transform = carla.Transform(carla.Location(x=193.5, y=212, z=1.85), carla.Rotation(yaw=-90, pitch=0, roll=0)) #town2 straight
            #vehicle_loc = carla.Location(x=100, y=200, z=0)
            #start_transform = self.map.get_waypoint(vehicle_loc, project_to_road=True, lane_type=carla.LaneType.Driving)

            # x="91.460960" y="54.534821" z="0.0" pitch="0.0" roll="0.0" yaw="-145.505829"
            z_elevate = 0.5
            transform = carla.Transform(carla.Location(x=91.460960, y=54.534821, z=z_elevate), carla.Rotation(yaw=-145.505829, pitch=0, roll=0)) #town3 start of roundabout
            return transform


        if self.map.name ==  "Town02":
            loc = carla.Location(x=180, y=-2, z=1)
            start_transform = self.map.get_waypoint(loc)
            return start_transform

            # Another possible implementation, not as good
            # if self.map.name == "Town04":
            #     road_id = 47
            #     road_length = 117.
            #     init_transforms = []
            #     for _ in range(num_vehicles):
            #         lane_id = random.choice([-1, -2, -3, -4])
            #         vehicle_s = np.random.uniform(road_length)  # length of road 47
            #         init_transforms.append(self.map.get_waypoint_xodr(road_id, lane_id, vehicle_s).transform)

    def pure_pursuit(self):
        L = 2.875
        Kdd = 4.0
        alpha_prev = 0

        transform = carla.Transform(carla.Location(x=193.5, y=125, z=1.85), carla.Rotation(yaw=270, pitch=0, roll=0)) #town2 90deg left1
        #transform = carla.Transform(carla.Location(x=183, y=105.5, z=1.85), carla.Rotation(yaw=180, pitch=0, roll=0)) #town2 straight1
        #transform = carla.Transform(carla.Location(x=50, y=105.5, z=1.85), carla.Rotation(yaw=180, pitch=0, roll=0)) #town2 90deg left2
        #transform = carla.Transform(carla.Location(x=18, y=105.5, z=1.85), carla.Rotation(yaw=180, pitch=0, roll=0)) #town2 90deg left2
        #transform = carla.Transform(carla.Location(x=-7, y=160, z=1.85), carla.Rotation(yaw=90, pitch=0, roll=0)) #town2 90deg left3
        #transform = carla.Transform(carla.Location(x=-7, y=235, z=1.85), carla.Rotation(yaw=90, pitch=0, roll=0)) #town2 straight
        #transform = carla.Transform(carla.Location(x=-7, y=285, z=1.85), carla.Rotation(yaw=90, pitch=0, roll=0)) #town2 straight
        #transform = carla.Transform(carla.Location(x=12, y=306.5, z=1.85), carla.Rotation(yaw=0, pitch=0, roll=0)) #town2 90deg left5
        #transform = carla.Transform(carla.Location(x=160, y=306.5, z=1.85), carla.Rotation(yaw=0, pitch=0, roll=0)) #town2 90deg left6
        #transform = carla.Transform(carla.Location(x=193.5, y=270.5, z=1.85), carla.Rotation(yaw=-90, pitch=0, roll=0)) #town2 straight
        #transform = carla.Transform(carla.Location(x=193.5, y=212, z=1.85), carla.Rotation(yaw=-90, pitch=0, roll=0)) #town2 straight
  
        vehicle_loc = self.vehicle.get_location()
        wp = self.map.get_waypoint(vehicle_loc, project_to_road=True, lane_type=carla.LaneType.Driving)

        waypoints = []
        num_wp = 300
        for _ in range(num_wp):
            wp = wp.next(2.0)[0]
            waypoints.append(wp)

        waypoint_list = [(wp.transform.location.x, wp.transform.location.y) for wp in waypoints]

        def calc_steering_angle(alpha, ld):
            delta_prev = 0
            delta = math.atan2(2 * L * np.sin(alpha), ld)
            delta = np.clip(delta, -1.0, 1.0)
            if math.isnan(delta):
                delta = delta_prev
            else:
                delta_prev = delta
            return delta

        def get_target_wp_index(veh_location, waypoint_list):
            dx = [abs(veh_location.x - wp[0]) for wp in waypoint_list]
            dy = [abs(veh_location.y - wp[1]) for wp in waypoint_list]
            dist = np.hypot(dx, dy)
            idx = np.argmin(dist) + 4
            if idx >= len(waypoint_list):
                idx = len(waypoint_list) - 1
            return idx, waypoint_list[idx][0], waypoint_list[idx][1]

        def get_lookahead_dist(vf):
            #return (20/3.6)*2 #taking 20kmph as vehicle velocity
            return Kdd * vf

        t = 0
        while t < num_wp:
            veh_transform = self.vehicle.get_transform()
            veh_location = self.vehicle.get_location()
            veh_vel = self.vehicle.get_velocity()
            vf = np.sqrt(veh_vel.x**2 + veh_vel.y**2)
            vf = np.clip(vf, 0.1, 2.5)

            min_index, tx, ty = get_target_wp_index(veh_location, waypoint_list)
            ld = get_lookahead_dist(vf)

            yaw = np.radians(veh_transform.rotation.yaw)
            alpha = math.atan2(ty - veh_location.y, tx - veh_location.x) - yaw
            #alpha = np.clip(alpha, -np.pi, np.pi)
            if math.isnan(alpha):
                alpha = alpha_prev
            else:
                alpha_prev = alpha
        
            steer_angle = calc_steering_angle(alpha, ld)
    
            return steer_angle
