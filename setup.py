import atexit
import collections
import os
import random
import signal
import subprocess
import sys
import time
from typing import Any
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import carla
import numpy as np
import transforms3d.euler
from absl import logging
import glob

logging.set_verbosity(logging.DEBUG)

def setup(
    town: str,
    fps: int = 20,
    server_timestop: float = 30.0,
    client_timeout: float = 20.0,
    num_max_restarts: int = 10,
):
    """Returns the `CARLA` `server`, `client` and `world`.

    Args:
        town: The `CARLA` town identifier.
        fps: The frequency (in Hz) of the simulation.
        server_timestop: The time interval between spawing the server
        and resuming program.
        client_timeout: The time interval before stopping
        the search for the carla server.
        num_max_restarts: Number of attempts to connect to the server.

    Returns:
        client: The `CARLA` client.
        world: The `CARLA` world.
        frame: The synchronous simulation time step ID.
        server: The `CARLA` server.
    """
    assert town in ("Town01", "Town02", "Town03", "Town04", "Town05")

    # The attempts counter.
    attempts = 0

    while attempts < num_max_restarts:
        logging.debug("{} out of {} attempts to setup the CARLA simulator".format(
            attempts + 1, num_max_restarts))

        # Random assignment of port.
        #port = np.random.randint(2000, 3000)
        port = 2000

        ## Start CARLA server.
        #env = os.environ.copy()
        #env["SDL_VIDEODRIVER"] = "offscreen"
        #env["SDL_HINT_CUDA_DEVICE"] = "0"
        #logging.debug("Inits a CARLA server at port={}".format(port))

        ## Ensure CARLA_ROOT is set, provide a default if not
        #carla_root = os.environ.get("CARLA_ROOT")
        #if carla_root is None:
        #    carla_root = "/home/aku8wk/Carla/CARLA_0.9.15"  # Replace with your actual path if different
        #    print(f"CARLA_ROOT environment variable not set. Using default path: {carla_root}")

        ## Construct the command string with the correct CARLA_ROOT path
        #command = f'DISPLAY= ' + str(os.path.join(carla_root, "CarlaUE4.sh")) + f' -opengl '+ f' -carla-rpc-port={port}' + f" -quality-level=Epic "

        ## Launch the CARLA server
        #server = subprocess.Popen(command, stdout=None, stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env, shell=True)

        ##server = subprocess.Popen(f'DISPLAY= ' + str(os.path.join(os.environ.get("CARLA_ROOT"), "CarlaUE4.sh")) + f' -opengl '+ f' -carla-rpc-port={port}' + f" -quality-level=Epic ", stdout=None, stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env, shell=True)
        ##server = subprocess.Popen(f'DISPLAY= ' + str(os.path.join(os.environ.get("/home/aku8wk/Carla/CARLA_0.9.15"), "CarlaUE4.sh")) + f' -opengl '+ f' -carla-rpc-port={port}' + f" -quality-level=Epic ", stdout=None, stderr=subprocess.STDOUT, preexec_fn=os.setsid, env=env, shell=True)
                     
        #texit.register(os.killpg, server.pid, signal.SIGKILL)
        #time.sleep(server_timestop)

        # config_script_path = "/home/aku8wk/Carla/CARLA_0.9.15/PythonAPI/util/config.py"
        # try:
        #     logging.debug("Running config script to disable rendering: {}".format(config_script_path))
        #     subprocess.run([config_script_path, "--no-rendering"], check=True)
        # except subprocess.CalledProcessError as e:
        #     logging.error("Failed to run config script: {}".format(e))
        #     sys.exit(1)

        # Connect client.
        logging.debug("Connects a CARLA client at port={}".format(port))
        try:
            client = carla.Client("localhost", port)  # pylint: disable=no-member
            client.set_timeout(client_timeout)
            client.load_world(map_name='Town02') #set town, or just set town to assert one of the 5
            world = client.get_world()
            world.set_weather(carla.WeatherParameters.ClearNoon)  # pylint: disable=no-member
            # if hasattr(world, 'set_no_rendering_mode'):                
            #     world.set_no_rendering_mode(True)  # No rendering mode
                
            #world.set_no_rendering_mode(True) #no rendering mode
            frame = world.apply_settings(
                carla.WorldSettings(  # pylint: disable=no-member
                    synchronous_mode=True,
                    fixed_delta_seconds=1.0 / fps,
                ))
            logging.debug("Server version: {}".format(client.get_server_version()))
            logging.debug("Client version: {}".format(client.get_client_version()))
            #return client, world, frame, server
            return client, world, frame
        except RuntimeError as msg: #carla connection attempt failed
            logging.debug(msg)
            attempts += 1
            logging.debug("Stopping CARLA server at port={}".format(port))
            os.killpg(server.pid, signal.SIGKILL)
            atexit.unregister(lambda: os.killpg(server.pid, signal.SIGKILL))

    logging.debug(
        "Failed to connect to CARLA after {} attempts".format(num_max_restarts))
    sys.exit()
