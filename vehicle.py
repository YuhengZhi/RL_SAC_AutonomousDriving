import carla
import time
import numpy as np
import math
import glob
import os
import sys

# Connect to the Carla simulator
client = carla.Client('localhost', 2000)
client.load_world('Town03')
world = client.get_world()
world.set_weather(carla.WeatherParameters.ClearNoon)
place = world.get_map()
#print(place.name)

# # Get the blueprint library
# blueprint_library = world.get_blueprint_library()

# # List all available vehicle blueprints
# #vehicle_blueprints = blueprint_library.filter('vehicle.*')

# # print("Available Vehicle Blueprints:")
# # for blueprint in vehicle_blueprints:
# #     print(blueprint.id)

# truck_blueprint = blueprint_library.find('vehicle.carlamotors.firetruck')

# # Check if the blueprint was found
# if truck_blueprint is not None:
#     print("Truck blueprint found.")
    
#     # Define the spawn location
#     spawn_location = carla.Transform(carla.Location(x=400, y=100, z=0), carla.Rotation(yaw=0))

#     # Spawn the vehicle using the chosen blueprint
#     vehicle = world.try_spawn_actor(truck_blueprint, spawn_location)

#     if vehicle is not None:
#         print(f"truck vehicle spawned successfully! (ID: {vehicle.id})")
        
#         # Get the location of the vehicle
#         vehicle_location = vehicle.get_location()        
        
#         # Add a spectator camera to view the spawned vehicle, height of z=70m, pitch =-90 gives top view
#         spectator = world.get_spectator()
#         spectator.set_transform(carla.Transform(vehicle_location + carla.Location(z=70), carla.Rotation(pitch=-90)))

##pure pursuit control

L = 2.875 # Wheelbase of the vehicle (distance between the front and rear axles)
Kdd = 4.0 # Look-ahead distance gain factor
alpha_prev = 0 # Previous steering angle error or heading error
delta_prev = 0 # Previous steering angle

#spawn vehicle

bp_lib = world.get_blueprint_library()
# vehicle_bp = bp_lib.filter('vehicle.dodge.charger_2020')[0]
vehicle_bp = bp_lib.filter('vehicle.daf.dafxf')[0]

# transform = carla.Transform() #vehicle's initial position and orientation

# transform.location.x = 220
# transform.location.y = -1.6
# transform.location.z = 1.85
# transform.rotation.yaw = 180
# transform.rotation.pitch = 0
# transform.rotation.roll = 0
#transform = carla.Transform(carla.Location(x=193.5, y=150, z=1.85), carla.Rotation(yaw=270, pitch=0, roll=0))
transform = carla.Transform(carla.Location(x=-7, y=285, z=1.85), carla.Rotation(yaw=90, pitch=0, roll=0))
#transform = carla.Transform(carla.Location(x=50, y=105.5, z=1.85), carla.Rotation(yaw=180, pitch=0, roll=0))
vehicle = world.spawn_actor(vehicle_bp, transform) #spawn vehicle

#spectator

spectator = world.get_spectator() # Get the spectator camera from the CARLA world
#sp_transform = carla.Transform(transform.location + carla.Location(z=90, x=-65,y=20),carla.Rotation(pitch=-90))
sp_transform = carla.Transform(transform.location + carla.Location(z=70, x=0,y=0),carla.Rotation(pitch=-90))
# camera_offset = carla.Location(x=-6.0, z=2.5)  # 6 units behind the car and 2.5 units above
# vehicle_transform = vehicle.get_transform()  # Get the vehicle's current transform
# camera_location = vehicle_transform.location + camera_offset
# camera_rotation = vehicle_transform.rotation
# sp_transform = carla.Transform(camera_location, camera_rotation)
spectator.set_transform(sp_transform)

#vehicle control

control = carla.VehicleControl() #creates an object from carla API to specify control commands
control.throttle = 0.4
vehicle.apply_control(control)

#vehicle current location

vehicle_loc = vehicle.get_location() #current vehicle location x,y,z coordinates
wp = place.get_waypoint(vehicle_loc, project_to_road=True, lane_type=carla.LaneType.Driving) #first point or the nearest driving waypoint to the vehicle's current location, ensuring it is on a drivable lane

waypoint_list = [] #stores coordinates x and y
waypoint_obj_list = [] #store the full waypoint objects with location, orientation, and lane type

# print values for debugging and info
def display(disp=False):
    if disp:
        print("--"*20)
        print("\nMin Index= ", min_index)
        print("Forward Vel= %.3f m/s"%vf)
        print("Lookahead Dist= %.2f m"%ld)
        print("Alpha= %.5f rad"%alpha)
        print("Delta= %.5f rad"%steer_angle)
        print("Error= %.3f m"%e)

# Calculate steering angle Delta
def calc_steering_angle(alpha, ld): #takes heading error and look ahead distance
    delta_prev = 0 # Initialize the previous steering angle
    delta = math.atan2(2*L*np.sin(alpha), ld) # Calculate the new steering angle
    delta = np.fmax(np.fmin(delta, 1.0), -1.0) # Clip the steering angle within [-1, 1]
    if math.isnan(delta): # Check if the calculated angle is NaN
        delta = delta_prev # If NaN, revert to the previous steering angle
    else:
        delta_prev = delta # Update the previous steering angle
    
    return delta # Return the calculated steering angle

# Get index of the target waypoint that the vehicle should aim for based on its current location and a list of waypoints
def get_target_wp_index(veh_location, waypoint_list): #takes current location and waypoint list
    dxl, dyl = [], [] # Initialize lists to store the differences in x and y coordinates
    for i in range(len(waypoint_list)):
        dx = abs(veh_location.x - waypoint_list[i][0]) #absolute difference in x coordinates
        dxl.append(dx) # Append the difference to the list
        dy = abs(veh_location.y - waypoint_list[i][1])
        dyl.append(dy)

    dist = np.hypot(dxl, dyl) #Euclidean distance from the vehicle location to each waypoint
    idx = np.argmin(dist) + 4 #index of the waypoint with the minimum distance, with an offset of 4

    # take closest waypoint, else last wp
    if idx < len(waypoint_list):
        tx = waypoint_list[idx][0] #x coordinate of the target waypoint
        ty = waypoint_list[idx][1]
    else:
        tx = waypoint_list[-1][0] #If the index exceeds the list length, set the target waypoint to the last one
        ty = waypoint_list[-1][1]

    return idx, tx, ty, dist #index of closest waypoint+4, coordinates of idx waypoint, distance vector from each waypoint to vehcile location


def get_lookahead_dist(vf, idx, waypoint_list, dist):
    ld = Kdd*vf #lookahead distance, with forward velocity vf
    # while ld > dist[idx] and (idx+1) < len(waypoint_list):
    #     idx += 1
    return ld


# Debug Helper, to visualize the waypoints, starting location for drawing is loc1
def draw(loc1, loc2=None, type=None):
    if type == "string": #draws X at loc1 that lasts for 2000ms
        world.debug.draw_string(loc1, ".",
                            life_time=2000, persistent_lines=True)
    elif type == "line": #draws a green line between loc1 and loc2 that lasts for 0.5 sec
        world.debug.draw_line(loc1, loc2, thickness=0.8,
         color=carla.Color(r=0, g=255, b=0),
                        life_time=0.5, persistent_lines=True)
    elif type == "string2": #draws a green X at loc1 that lasts for 0.3 sec
        world.debug.draw_string(loc1, "X", color=carla.Color(r=0, g=255, b=0),
                            life_time=0.3, persistent_lines=True)

# Generate waypoints, waypoint list using the vehicle location-wp
noOfWp = 50 #total number of waypoints to generate
t = 0
while t < noOfWp:
    wp_next = wp.next(2.0) #Generates the next waypoint(s) from the current waypoint wp, with a spacing of 2.0 meters between each waypoint
    if len(wp_next) > 1: #if multiple waypoints returned
        wp = wp_next[1] #select the second waypoint
    else: #If only one waypoint is returned
        wp = wp_next[0] #select that waypoint

    waypoint_obj_list.append(wp) #Appends the selected waypoint object to waypoint_obj_list
    waypoint_list.insert(t, (wp.transform.location.x, wp.transform.location.y)) # Inserts a tuple (x, y) representing the waypoint's location into waypoint_list at index t
    draw(wp.transform.location, type="string") #for visualization of waypoints
    t += 1

# path tracking
    
t = 0
while t < noOfWp:
    veh_transform = vehicle.get_transform() #current position and orientation of the vehicle
    veh_location = vehicle.get_location() #current location of the vehicle
    veh_vel = vehicle.get_velocity() #current velocity of the vehicle
    vf = np.sqrt(veh_vel.x**2 + veh_vel.y**2) #forward velocity of the vehicle
    vf = np.fmax(np.fmin(vf, 2.5), 0.1) #Clips the forward velocity within the range [0.1, 2.5]

    min_index, tx, ty, dist = get_target_wp_index(veh_location, waypoint_list)
    ld = get_lookahead_dist(vf, min_index, waypoint_list, dist)


    yaw = np.radians(veh_transform.rotation.yaw)
    alpha = math.atan2(ty-veh_location.y, tx-veh_location.x) - yaw
    # alpha = np.arccos((ex*np.cos(yaw)+ey*np.sin(yaw))/ld)

    if math.isnan(alpha):
        alpha = alpha_prev
    else:
        alpha_prev = alpha

    e = np.sin(alpha)*ld
    
    steer_angle = calc_steering_angle(alpha, ld)
    control.steer = steer_angle
    vehicle.apply_control(control)

    # draw(veh_location, waypoint_obj_list[min_index].transform.location, type="line")
    draw(waypoint_obj_list[min_index].transform.location, type="string2")
    display(disp=True)

    time.sleep(0.5)
    t += 1

print("Task Done!")
vehicle.destroy()
