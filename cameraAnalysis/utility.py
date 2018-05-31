import numpy as np

def sphe_to_cart(sphe_coord):
    """Change spherical coordinates to spherical coordinate
    
    Args:
        sphe_coord (np.array([x, y, z])) : unit vector in cartesian coordinate
    
    Return:
        np.array([phi, theta]) : unit vetor in spherical coordinate
    """
    if len(sphe_coord.shape) == 1:
        phi, theta = sphe_coord
    else:
        phi, theta = sphe_coord[:, 0], sphe_coord[:, 1]
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]).T

def cart_to_sphe(cart_coord):
    """Change cartesian coordinates to spherical coordinate
    
    Args:
        cart_coord (np.array([x, y, z])) : unit vector in cartesian coordinate
    
    Return:
        np.array([phi, theta]) : unit vetor in spherical coordinate
    """
    if len(cart_coord.shape) == 1:
        x, y, z = cart_coord
    else:
        x, y, z = cart_coord[:, 0], cart_coord[:, 1], cart_coord[:, 2]
    return np.array([np.arctan(y/x), np.arccos(z)]).T

def calculate_inter_angle_to_camera(camera_normal_vec, photon_dir_vec):
    """It wil calculate the angle between camera normal vector and photon track directional vector
    The result will be used for the purpose of applying opening angle
    
    Args:
        camera_normal_vec (np.array([x, y, z])) : unit vector of camera plane(normal vector)
        photon_dir_vec (np.array([[x,y,z]])) : directional vector of individual photon
    
    Returns:
        inter angle of each photon and camera
    """
    if len(camera_normal_vec.shape) != 1:
        print("Camera normal vector should be just (x, y, z)")
        return 
    return np.arccos(-(np.sum(camera_normal_vec * photon_dir_vec, axis = 1)) / 
                    (np.linalg.norm(camera_normal_vec) * np.linalg.norm(photon_dir_vec, axis=1)))


def get_sensor_filter(photon_pos_sphe, photon_momentum_sphe, camera_pos_sphe, 
                      radius=0.16510, camera_distance=0.02, camera_lens_rad = 0.005, opening_angle = np.pi * 3 / 2):
    """From the boundary, check photon can arrive in lens or not. There are main two parts
    1. Check lens geometry and photon track
    2. Compare opening angle
    
    Args:
        photon_pos_sphe(np.array([phi, theta])) : unit vector of photon arrival position in spherical coordinate
        photon_momentum_sphe(np.array([phi, theta])) : unit vector of photon momentum in spherical coordinate
        camera_pos_sphe(np.array([phi, theta])) : unit vector of camera position in spherical coordinate
        radius (float) : radius of boundary which photon pos are located
        camera_distance (float) : how much far from boundary
        camera_lens_rad (float) : lens radius of camera
        opening_angle (float) : opening angle of camera

    Returns:
        np.array([bool]) : condition that photon will arrive in lens or not
    """
    camera_pos_rad = radius - camera_distance
    
    # For the linear equation of incident photon
    # l : (x - x_0) / alpha = (y - y_0) / beta = (z - z_0) / gamma = t
    arrival_pos_cart = radius * sphe_to_cart(photon_pos_sphe) # (x_0, y_0, z_0)
    arrival_dir_vec = -sphe_to_cart(photon_momentum_sphe) # (alpha, beta, gamma)
    
    # Camera position for the plane equation
    # S : a(x - x_c) + b(y - y_c) + c(z - z_c) = 0
    camera_normal_vec = sphe_to_cart(camera_pos_sphe) # (a, b, c)
    camera_pos_cart = camera_pos_rad * camera_normal_vec # (x_c, y_c, z_c)
    
    # Find t calculated in linear in plane equation
    # solve simultaneous equations between l and S
    t = -(np.sum(camera_normal_vec * (arrival_pos_cart - camera_pos_cart), axis=1)) / np.sum(camera_normal_vec * arrival_dir_vec, axis=1)
    
    # Calculate intersection on the plane(S) using t
    # x_1 = alpha * t + x_0, y_1 = beta * t + y_0, z_1 = gamma * t + z_0
    intersect_cart = (t * arrival_dir_vec.T).T + arrival_pos_cart # (x_1, y_1, z_1)
    
    # Calculate distance how much far from camera centor point
    # np.sqrt((x_1 - x_c) ** 2 + (y_1 - y_c) ** 2 + (z_1 - z_c) ** 2)
    distance_from_center = np.linalg.norm((intersect_cart - camera_pos_cart), axis=1)
    
    # Compare with the lens radius 
    is_arrive_sensor = distance_from_center < camera_lens_rad
    
    # Calculate angle between line and plane normal vector
    inter_angle = calculate_inter_angle_to_camera(camera_normal_vec, arrival_dir_vec)
    
    # Combine all condtion to arrive in sensor
    sensor_filter = is_arrive_sensor * (inter_angle < opening_angle / 2)
    
    return sensor_filter