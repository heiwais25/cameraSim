import numpy as np
from photon_interaction import PhotonInteraction
from utility import DOM, Camera, Ice, refractive_index_ice

# Initial setting
wv = 405  # [nm]
diff = [30, 0, 0]
dom = DOM(radius=0.16510, thickness=0.0127, vessel_index=1.47, inner_index=1, pos_diff=diff)
camera = Camera(pos_sphe=[0, np.pi/2], pos_distance=0.02, lens_rad=0.005, FOV=90)
ice = Ice(sca_length=42.53, abs_length=147.811, index=refractive_index_ice(wv))

# For now, PPC ran with the option of photon order 12 and oversizing of area order 2
# So, the photon order of result is about 10^14
# To do 10 times measure ment, iteration was done ten time
photon = PhotonInteraction(dom, camera, ice, iteration=1, num_photon_order=12)
photon.set_src_dir("/mnt/hdd/ppc_log/remote/")
photon.set_output_dir("../output/")

total_iteration = 10
for i in range(1, total_iteration):
    photon.iteration = i
    photon.save_all_photon_after_vessel(num_files=200)