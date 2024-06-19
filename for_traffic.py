import os

import carla
import subprocess
import logging
import random

class CarlaEnv:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town02')
        self.blueprint_library = self.world.get_blueprint_library()
        self.traffic_manager = self.client.get_trafficmanager()
        self.ego_vehicle = None
        
        self.setup_environment()

    def setup_environment(self):
        self.spawn_ego_vehicle()
        self.spawn_traffic()

    def reset(self):
        self.destroy_traffic()
        self.spawn_traffic()
        # Reset ego vehicle position and state as needed
        self.reset_ego_vehicle()
        # Reset other environment variables if necessary
        return self.get_state()  # Return initial state

    def spawn_ego_vehicle(self):
        blueprint = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.ego_vehicle = self.world.spawn_actor(blueprint, spawn_point)

    def spawn_traffic(self):
        # Command to run the generate_traffic.py script
        py_api_path = os.path.join(carla.__path__[0].split('PythonAPI')[0], 'PythonAPI')
        traffic_script_path = os.path.join(py_api_path, 'examples', 'generate_traffic.py')
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

    def destroy_traffic(self):
        # Assuming there is a way to destroy all non-ego vehicles
        actors = self.world.get_actors().filter('vehicle.*')
        for actor in actors:
            if actor.id != self.ego_vehicle.id:
                actor.destroy()

    def reset_ego_vehicle(self):
        # Code to reset the ego vehicle's state and position
        pass

    def get_state(self):
        # Code to return the current state of the environment
        pass

    def step(self, action):
        # Apply action, step the environment, and return new state, reward, done, info
        pass

    def close(self):
        if self.ego_vehicle:
            self.ego_vehicle.destroy()
        self.destroy_traffic()
