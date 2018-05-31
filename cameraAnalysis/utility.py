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
                      radius=0.16510, camera_distance=0.02, camera_lens_rad = 0.005, opening_angle = np.pi * 2 / 3):
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


# Calculate refractive index of ice which depends on the wavelength
def refractive_index_ice(wv):
    """Return refractive index of ice at certain wavelength
    [https://wiki.icecube.wisc.edu/index.php/Refractive_index_of_ice]

    Args:
        wv (float) : wavelength of photon in [nm]
    
    Returns:
        (float) : refractive index
    """
    wv_um = wv / 1000
    return 1.55749 - 1.57988 * wv_um + 3.99993 * wv_um ** 2 - 4.68271 * wv_um ** 3 + 2.093354 * wv_um ** 4


def calc_refracted_dir(s1, N, n1, n2):
    """Calculate refracted direction using 3 dimensional Snell's law
    [http://www.starkeffects.com/snells-law-vector.shtml]
    
    Args:
        s1 (np.array([x, y, z])) : directional vector of incident vector
        N (np.array([x, y, z])) : normal vector on the surface
        n1 (float) : refractive index of incident place
        n2 (float) : refractive index of refracted place
    
    Returns:
        s2 (np.array([x, y, z])) : directional vector of refracted light
    """
    n = n1 / n2
    cross = np.cross(N,s1)
    # print(cross)
    # print(n**2 * np.sum(cross * cross, axis=1))
    s2 = n * np.cross(N, -cross) - (N.T * np.sqrt(1 - n**2 * np.sum(cross * cross, axis=1))).T
    return s2


def pass_pressure_vessel(photon_pos_sphe, photon_momentum_sphe, n0, r0 = 0.16510, d0 = 0.0127, n1 = 1.47, n2 = 1.):
    """Using Snell's law, pass two steps (ice -> pressure vessel, pressure vessel -> inner DOM air)
    [http://www.starkeffects.com/snells-law-vector.shtml]
    
    Args:
        photon_pos_sphe(np.array([phi, theta])) : unit vector of photon arrival position in spherical coordinate
        photon_momentum_sphe(np.array([phi, theta])) : unit vector of photon momentum in spherical coordinate
        n0 (float) : refractive index of ice
        r0 (float) : DOM radius(pressure vessel)
        d0 (float) : thickness of pressue vessel
        n1 (float) : refractive index of pressure vessel
        n2 (float) : refractiev index of air(inside of vessel)
    
    """
    r1 = r0 - d0
    
    # step 1 (ice -> vessel)
    s1 = sphe_to_cart(photon_momentum_sphe) * -1 # inicident angle
    N = sphe_to_cart(photon_pos_sphe)
    s2 = calc_refracted_dir(s1, N, n0, n1)
    
    # Move from outer boundary to inner boundary inside of vessel 
    # It is just solving simultaneous equations of sphere(inner vessel) and refracted photon track
    p0 = r0 * sphe_to_cart(photon_pos_sphe)
    A = np.sum(s2 ** 2, axis=1) # alpha ^ 2 + beta ^2 + gamma ^2 (It is ~ 1 : unit vector)
    B = 2 * np.sum(s2 * p0, axis = 1) # 2 * (alpha * x_0 + beta * y_0 + gamma * z_0)
    C = np.sum(p0 ** 2, axis=1) - r1 ** 2 # x_0 ^ 2 + y_0 ^ 2 + z_0 ^ 2 - r_1 ^ 2
    t0 = (-B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
    t1 = (-B - np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)
    
    # p1 : position vector on the inner vessel, p0 : position vector on the outer vessel
    p1_0 = (s2.T * t0).T   + p0
    p1_1 = (s2.T * t1).T   + p0
    
    # Find the closest point to the p0
    l0 = np.linalg.norm(p1_0 - p0, axis=1) 
    l1 = np.linalg.norm(p1_1 - p0, axis=1)
    if np.sum(l0 < l1) == 0: # l0 is longer than l1
        p1 = p1_1
    else:
        p1 = p1_0
        
    # Step 2 (vessel -> DOM interior)
    s1 = s2
    N = p1 / r1
    s2 = calc_refracted_dir(s1, N, n1, n2)

    # Because there are some photons cannot pass the vessel, s2 includes np.nan. We need to filter that value
    nan_filter = ~np.isnan(s2)
    
    # With nan filter, calculate momentum and position of photon after passing the vessel
    new_momentum_sphe = cart_to_sphe(-s2[nan_filter].reshape(-1, 3))
    new_pos_sphe = cart_to_sphe(N[nan_filter].reshape(-1, 3))
    
    return new_momentum_sphe, new_pos_sphe