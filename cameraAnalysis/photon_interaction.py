import numpy as np
import os
import h5py
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from utility import Camera, DOM, Ice, pass_pressure_vessel, get_sensor_filter, calculate_inter_angle_to_camera
from utility import cart_to_sphe, sphe_to_cart, get_total_hist2d, refractive_index_ice
import copy
import itertools


class PhotonInteraction:
    """It will calculate expected photon track and check with camera lens

    The main process is 3 steps to get photon information on the lens

    1. Calculate the refraction effect at (ice -> vessel, vessel -> dom)
        Method : 
            pass_pressure_vessel(), save_all_photon_after_vessel() 
        - This process is most time consuming part
        - To cover time, it is good to save result after this process

    2. Calculate each photon track inside the dom and relative distance between camera and track
        Method :
            get_sensor_filter()
        - After the refraction, we get the momentum and position info of photon inside of DOM
        - Using line of photon track and plane equation of camera, check whether photon pass the camera

    3. Using the FOV of camera, cut photons which are incident from outside of FOV
        Method :
            calculate_inter_angle_to_camera()
        - Calculate the interior angle between camera normal vector and photon momentum.
        - If the inerior angle is larger than FOV / 2, exclude photon

    After this process, we can get the photon information on the lens(camera)
    By binning the photon momentum, we can get sample image

    Current maximum number of photon emitted by PPC is 10^12. With oversizing the camera radius 10 times, 
    We can get 10^14. Also, if we assume each photon corresponds to the 100 real photon, we can set 10^16.
    """
    def __init__(self, dom, camera, ice, num_photon_order, src_dir="", output_dir="", hist_output_dir="", iteration=0):
        """Init the class

        Args:
            dom (DOM) : DOM parameters like radius, thickness and refractive index
            camera (Camera) : camera parameters like position, rad, FOV
            ice (Ice) : ice parameters like scattering, absorption length and refractive index
            num_photon_order (float) : order of total number of emitted photon
            src_dir (str) : source directory where raw file
            output_dir (str) : output directory where plot will be stored
            hist_output_dir (str) : output directory where final hist is stored(hdf5)
        """
        self.dom = copy.deepcopy(dom)
        self.camera = copy.deepcopy(camera)
        self.ice = copy.copy(ice)
        self.num_photon_order = num_photon_order
        self.iteration = iteration
        self.src_dir = src_dir
        self.output_dir = output_dir
        self.hist_output_dir = hist_output_dir


    def set_src_dir(self, src_dir):
        """Set the source directory which include results from PPC

        Args:
            src_dir (str) : source directory where raw file
        """
        self.src_dir = dir

    def set_output_dir(self, output_dir):
        """Set the output directory which plots will be stored

        Args:
            output_dir (str) : output directory where plot will be stored
        """
        self.output_dir = dir


    def set_hist_output_dir(self, dir):
        """output directory where final hist is stored(hdf5)

        Args:
            hist_output_dir (str) : output directory where final hist is stored(hdf5)
        """
        self.hist_output_dir = dir

    def get_single_data(self):
        """Get first data corresponding to the set option in initialization
        Because the file size is too large, the file separated into several files
        Maximum size of each file is 256MB(8388608 photon log). 
        This is for the checking photon behavior
        """
        # Get photon information from first file at current setting
        file_name = "ppc_log_x_%.2f_y_%.2f_z_%.2f_order_%d_iter_%d_0.dat" \
                            % (self.dom.pos_diff[0], self.dom.pos_diff[1], self.dom.pos_diff[2], self.num_photon_order, self.iteration)
        self.photon_log = np.fromfile(self.src_dir + file_name, dtype='<f4').reshape(-1, 8)
        self.orig_momentum= self.photon_log[:,5:3:-1] 
        self.orig_position = self.photon_log[:,7:5:-1]


    def pass_pressure_vessel(self):
        """Pass pressure vessel with cosideration of refraction in two steps
        1. Ice -> Pressure vessel
        2. Pressure_vessel -> Inner side of DOM
        With this function, we can get momentum and position after refraction
        Some photons cannot pass into the DOM because of the refractive index
        """
        self.refracted_momentum, self.refracted_position = pass_pressure_vessel(self.orig_position, self.orig_momentum, 
                                                    r0=self.dom.radius,
                                                    d0=self.dom.thickness,
                                                    n0=self.ice.index, 
                                                    n1=self.dom.vessel_index, 
                                                    n2=self.dom.inner_index)



    def get_total_hist2d(self, num_files, oversize_order=0, bins=40, maxbin=False, print_num=False, use_stored_data=False, store_result=False):
        """Get total data stored in separated files. It is hard to combine all data in memory, so we 
        will store in the histogram format and add continuously

        Args:
            maxbin (bool) : if it is true, bin will be the maximum area within the FOV / 2

        """
        if maxbin:
            phi_range, theta_range = self.calc_maxbin_within_FOV()
            phi = np.linspace(phi_range[0], phi_range[-1], bins+1)
            theta = np.linspace(theta_range[0], theta_range[-1], bins+1)
            bins=[phi, theta]

        file_name = "ppc_log_x_%.2f_y_%.2f_z_%.2f_order_%d_iter_%d_0.dat" \
                            % (self.dom.pos_diff[0], self.dom.pos_diff[1], self.dom.pos_diff[2], self.num_photon_order, self.iteration)
        self.total_hist, self.total_hist_x_edges, self.total_hist_y_edges = get_total_hist2d(file_name, num_files, 
                            bins=bins,
                            r0=self.dom.radius, 
                            d0=self.dom.thickness,
                            camera_pos_sphe=self.camera.pos_sphe,
                            camera_distance=self.camera.pos_distance,
                            camera_lens_rad=self.camera.lens_rad,
                            wv=405,
                            FOV=self.camera.FOV,
                            src_raw_dir=self.src_dir,
                            print_num=print_num,
                            use_stored_data=use_stored_data)

        if store_result:
            file_name = "photon_lens_x_%.2f_y_%.2f_z_%.2f_order_%d_over_%d_iter_%d_camera_%.2f_%.2f_bins_%d.hdf5" \
                                % (self.dom.pos_diff[0], self.dom.pos_diff[1], self.dom.pos_diff[2], 
                                self.num_photon_order, oversize_order, self.iteration, self.camera.pos_sphe[0], self.camera.pos_sphe[1], bins)

            f = h5py.File(self.hist_output_dir + file_name, 'w')
            photon_hist = f.create_group(u"photon")
            photon_hist.create_dataset(u"histogram", data=self.total_hist)
            photon_hist.create_dataset(u"x_edges", data=self.total_hist_x_edges)
            photon_hist.create_dataset(u"y_edges", data=self.total_hist_y_edges)
            f.close()

        return self.total_hist, self.total_hist_x_edges, self.total_hist_y_edges


    def calc_maxbin_within_FOV(self):
        """Calculate maximum range of phi and theta within the FOV

        """
    
        half_opening_angle = np.deg2rad(self.camera.FOV / 2)

        # Find the phi range
        phi = np.linspace(self.camera.pos_sphe[0] - np.pi, self.camera.pos_sphe[0] + np.pi, 200)
        theta = [self.camera.pos_sphe[1]]
        test_cases = np.array(list(itertools.product(phi, theta)))
        test_angle = calculate_inter_angle_to_camera(sphe_to_cart(self.camera.pos_sphe), -sphe_to_cart(test_cases))
        larger = True
        phi_range = [0] * 2
        for i in range(len(phi)):
            if larger and test_angle[i] < half_opening_angle:
                larger = False
                phi_range[0] = phi[i-1]
            elif not larger and test_angle[i] > half_opening_angle:
                larger = True
                phi_range[1] = phi[i]

        # Find the theta range
        phi = [self.camera.pos_sphe[0]]
        theta = np.linspace(self.camera.pos_sphe[1] - np.pi / 2, self.camera.pos_sphe[1] + np.pi / 2, 200)
        test_cases = np.array(list(itertools.product(phi, theta)))
        test_angle = calculate_inter_angle_to_camera(sphe_to_cart(self.camera.pos_sphe), -sphe_to_cart(test_cases))
        larger = True
        theta_range = [0] * 2
        for i in range(len(theta)):
            if larger and test_angle[i] < half_opening_angle:
                larger = False
                theta_range[0] = theta[i-1]
            elif not larger and test_angle[i] > half_opening_angle:
                larger = True
                theta_range[1] = theta[i]
        return phi_range, theta_range


    


    def save_all_photon_after_vessel(self, out_dir = "", num_files=1, wv=405):
        """Because it is hard to do every process at every file(high computational cost), we can save photon
        information just after the refraction on the vessel. So, in the large data process we can deal with this data
        with short time. The format would be 
        
        momentum(phi, theta), position(phi, theta)

        Maximum size of each file is 256MB
        Each photon information is 4 float data(momentum[phi, theta], position[phi, theta]) : 4 x 4 = 16bytes
        So, max contents can be stored in each file is (256 x 1024 x 1024) / 16 = 16777216

        The result will be stored in the directory (out_dir)
        If "out_dir" is not set, it will store output data in the directory of "src_dir"/refraction/

        It is easy to get file again with keys. Keys can check by np.load().files

        Args:
            out_dir (str) : output directory where photon data after refraction will be stored
            wv (float) : wavlength of photon to calculate the refraction index in ice
        """
        all_momentum = []
        all_position = []
        next_momentum = []
        next_position = []
        ice_index = refractive_index_ice(wv)
        src_name = "ppc_log_x_%.2f_y_%.2f_z_%.2f_order_%d_iter_%d_0.dat" \
                            % (self.dom.pos_diff[0], self.dom.pos_diff[1], self.dom.pos_diff[2], self.num_photon_order, self.iteration)
        out_name = "ppc_log_refraction_x_%.2f_y_%.2f_z_%.2f_order_%d_iter_%d_0.dat" \
                            % (self.dom.pos_diff[0], self.dom.pos_diff[1], self.dom.pos_diff[2], self.num_photon_order, self.iteration)
        max_content_count = 16777216 # For 256MB 
        out_file_count = 0
        current_count = 0
        total_read_length = 0
        if len(out_dir) == 0:
            out_dir = self.src_dir + "refraction/"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

        for i in range(num_files):
            this_src_name = src_name.replace('0.dat', '%d.dat' % i)
            if not os.path.isfile(self.src_dir + this_src_name):
                break

            raw_data = np.fromfile(self.src_dir + this_src_name, dtype='<f4')
            photon_log = raw_data.reshape(-1, 8)
            momentum = photon_log[:,5:3:-1]
            position = photon_log[:,7:5:-1]
            momentum, position = pass_pressure_vessel(position, momentum, ice_index)
            total_read_length += len(momentum)
            if current_count == 0:
                all_momentum = momentum
                all_position = position
                current_count += len(all_momentum)
            elif current_count + len(momentum) < max_content_count:
                all_momentum = np.concatenate((all_momentum, momentum))
                all_position = np.concatenate((all_position, position))
                current_count += len(momentum)
            else: # Cut to max content count
                all_momentum = np.concatenate((all_momentum, momentum[:max_content_count - current_count]))
                all_position = np.concatenate((all_position, position[:max_content_count - current_count]))
                next_momentum = momentum[max_content_count - current_count:]
                next_position = position[max_content_count - current_count:]
                current_count += max_content_count - current_count

            if current_count == max_content_count:
                current_count = 0
                this_out_name = out_dir + out_name.replace('0.dat', '%d' % (out_file_count))
                out_file_count += 1
                np.savez(this_out_name, momentum = all_momentum, position = all_position)            
                if len(next_momentum) > 0:
                    all_momentum = copy.deepcopy(next_momentum)
                    all_position = copy.deepcopy(next_position)
                    current_count = len(all_momentum)
                else:
                    all_momentum = []
                    all_position = []
                print('Save file : %s.npz' % this_out_name)
        
        if current_count != 0:
            this_out_name = out_dir + out_name.replace('0.dat', '%d' % (out_file_count))
            out_file_count += 1
            np.savez(this_out_name, momentum = all_momentum, position = all_position)            
            print('Save file : %s.npz' % this_out_name)
                
        print("Total Length : ", total_read_length)
        


    def filter_photon_by_lens(self):
        """Filter refracted photon by lens. There are two steps to get the filtered data
        1. Check the photon track and lens geometry, select photons which can pass the lens
        2. Among selected photon, select photons incident in the lower angle than half of FOV 
        """
        lens_filter = get_sensor_filter(self.refracted_position, self.refracted_momentum, self.camera.pos_sphe,
                                            radius = self.dom.radius,
                                            camera_distance = self.camera.pos_distance,
                                            camera_lens_rad = self.camera.lens_rad,
                                            FOV=self.camera.FOV
                                            )
        self.momentum_on_lens = self.refracted_momentum[lens_filter]
        self.position_on_lens = self.refracted_position[lens_filter]

    def draw_single_data(self, mode, 
                            figsize=(12,6), 
                            fontsize=15, 
                            bins=40,
                            refraction=False,
                            savefig=False, 
                            showfig=False):
        """Draw orig_momentum or orig_position 

        Args:
            mode (str) : choose momentum or position to plot
            savefig (bool) : whether to save figure 
            showfig (bool) : whether to show figure
        """    
        if mode != "momentum" and mode != "position":
            raise Exception("Invalid mode")


        if refraction:
            data = self.refracted_momentum if mode == "momentum" else self.refracted_position
        else:
            data = self.orig_momentum if mode == "momentum" else self.orig_position

        
        fig, ax = plt.subplots(figsize=figsize)
        plt.rcParams['font.size']=fontsize
        h = plt.hist2d(data[:,0], data[:,1], bins=bins, norm=LogNorm());
        plt.colorbar(h[3])
        plt.xlabel("Arrival %s [rad]" % (r'$\varphi$'))
        plt.ylabel("Arrival %s [rad]" % (r'$\theta$'))
        title = 'Arrival %s of the $\gamma $(on the DOM surface)\n[$l_{scat}$ = %.2fm, $l_{abs}$ = %.2fm, $10^{%d}$ $\gamma$ from %dm away LED]' \
          % (mode, self.ice.sca_length, self.ice.abs_length, self.num_photon_order, self.dom.pos_diff[0])
        if mode == "momentum":
            file_name = 'orig_arr_mom_scat_%.2f_abs_%.2f_diff_%dm_order_%d.png' \
                        % (self.ice.sca_length, self.ice.abs_length, self.dom.pos_diff[0], self.num_photon_order)
        else:
            file_name = 'orig_arr_pos_scat_%.2f_abs_%.2f_diff_%dm_order_%d.png' \
                        % (self.ice.sca_length, self.ice.abs_length, self.dom.pos_diff[0], self.num_photon_order)

        if refraction:
            title = title.replace("on the DOM surface", "after refraction")
            file_name = file_name.replace("orig", "refr")

        plt.title(title, fontsize=fontsize-1)

        if savefig:
            plt.savefig(self.output_dir + file_name, dpi=400)
        if not showfig:
            plt.close()


    def draw_single_data_on_lens(self, mode,
                                    figsize=(12,6),
                                    fontsize=15,
                                    bins=40,
                                    xlim=[],
                                    ylim=[],
                                    totalbin=False,
                                    refraction=False,
                                    showFOV=True,
                                    savefig=False,
                                    showfig=False):
        """Draw single data on the lens after refracton

        Args:
            totalbin(bool) : same bin size with original data to compare the lens effect
        """
        if mode != "momentum" and mode != "position":
            raise Exception("Invalid mode")
        data = self.momentum_on_lens if mode == "momentum" else self.position_on_lens

        fig, ax = plt.subplots(figsize=figsize)
        plt.rcParams['font.size']=fontsize
        if totalbin:
            H0, y_edges, x_edges = np.histogram2d(self.orig_momentum[:,1], self.orig_momentum[:,0], bins=bins)
            h = plt.hist2d(data[:,0], data[:,1], bins=[x_edges, y_edges], norm=LogNorm());
        else:
            H0, y_edges, x_edges = np.histogram2d(self.momentum_on_lens[:,1], self.momentum_on_lens[:,0], bins=bins)
            h = plt.hist2d(data[:,0], data[:,1], bins=bins, norm=LogNorm());
            if len(xlim) == 0:
                width = x_edges[-1] - x_edges[0]
                xlim = [max(-np.pi, x_edges[0] - width/2), min(np.pi, x_edges[-1] + width/2)]
            if len(ylim) == 0:
                height = y_edges[-1] - y_edges[0]
                ylim = [max(0, y_edges[0] - height/2), min(np.pi, y_edges[-1] + height/2)]
            plt.xlim(xlim)
            plt.ylim(ylim)
        plt.colorbar(h[3])

        if showFOV:
            theta = np.linspace(0, np.pi, 100)
            phi = np.linspace(-np.pi, np.pi, 100)
            test_cases = np.array(list(itertools.product(phi, theta)))
            test_angle = calculate_inter_angle_to_camera(sphe_to_cart(self.camera.pos_sphe), -sphe_to_cart(test_cases))

            half_opening_angle = np.deg2rad(self.camera.FOV) / 2
            CS = plt.contour(test_angle.reshape(100, 100).T, levels = [half_opening_angle], extent=[-np.pi, np.pi, 0, np.pi])
            CS.collections[0].set_label("FOV/2")
            plt.legend()
                        
            plt.clabel(CS, inline=1, fontsize=13)
            

        plt.xlabel("Arrival %s [rad]" % (r'$\varphi$'))
        plt.ylabel("Arrival %s [rad]" % (r'$\theta$'))
        title = 'Arrival %s of the $\gamma$ (on lens)\n[$l_{scat}$ = %.2fm, $l_{abs}$ = %.2fm, $10^{%d}$ $\gamma$ from %dm away LED]' \
            % (mode, self.ice.sca_length, self.ice.abs_length, self.num_photon_order, self.dom.pos_diff[0])

        if mode == "momentum":
            file_name = 'lens_arr_mom_scat_%.2f_abs_%.2f_diff_%dm_order_%d.png' \
                        % (self.ice.sca_length, self.ice.abs_length, self.dom.pos_diff[0], self.num_photon_order)
        else:
            file_name = 'lens_arr_pos_scat_%.2f_abs_%.2f_diff_%dm_order_%d.png' \
                        % (self.ice.sca_length, self.ice.abs_length, self.dom.pos_diff[0], self.num_photon_order)

        if totalbin:
            title = title.replace('on lens', 'on lens, total bin')
            file_name = file_name.replace('lens', 'lens_total_bin')

        plt.title(title, fontsize=fontsize-1)

        if savefig:
            plt.savefig(self.output_dir + file_name, dpi=400)
        if not showfig:
            plt.close()


    def draw_all_data_on_lens(self, mode, 
                                    figsize=(8,6),
                                    fontsize=15,
                                    bins=40,
                                    xlim=[],
                                    ylim=[],
                                    oversize_order=0,
                                    refraction=False,
                                    showFOV=True,
                                    gray=False,
                                    zoom=False,
                                    savefig=False,
                                    showfig=False):
        """Draw all data read from separated file

        """
        if mode != "momentum" and mode != "position":
            raise Exception("Invalid mode")

        fig, ax = plt.subplots(figsize=figsize)

        if showFOV:
            theta = np.linspace(0, np.pi, 100)
            phi = np.linspace(-np.pi, np.pi, 100)
            test_cases = np.array(list(itertools.product(phi, theta)))
            test_angle = calculate_inter_angle_to_camera(sphe_to_cart(self.camera.pos_sphe), -sphe_to_cart(test_cases))

            half_opening_angle = np.deg2rad(self.camera.FOV) / 2
            CS = plt.contour(test_angle.reshape(100, 100).T, levels = [half_opening_angle], extent=[-np.pi, np.pi, 0, np.pi])
            CS.collections[0].set_label("FOV/2")
            plt.legend()
                        
            plt.clabel(CS, inline=1, fontsize=13)
            

        plt.rcParams['font.size']=fontsize
        if gray:
            cax = plt.imshow(self.total_hist, extent=[self.total_hist_x_edges[0], self.total_hist_x_edges[-1], 
                                    self.total_hist_y_edges[0], self.total_hist_y_edges[-1]], origin='low', cmap='gray')
        else:
            cax = plt.imshow(self.total_hist, extent=[self.total_hist_x_edges[0], self.total_hist_x_edges[-1], 
                                        self.total_hist_y_edges[0], self.total_hist_y_edges[-1]], norm=LogNorm(), origin='low')

        if not zoom:
            if len(xlim) == 0:
                    width = self.total_hist_x_edges[-1] - self.total_hist_x_edges[0]
                    xlim = [max(-np.pi, self.total_hist_x_edges[0] - width/2), min(np.pi, self.total_hist_x_edges[-1] + width/2)]
            if len(ylim) == 0:
                height = self.total_hist_y_edges[-1] - self.total_hist_y_edges[0]
                ylim = [max(0, self.total_hist_y_edges[0] - height/2), min(np.pi, self.total_hist_y_edges[-1] + height/2)]
            plt.xlim(xlim)
            plt.ylim(ylim)

        cbar = plt.colorbar(cax)



        plt.xlabel("Arrival %s [rad]" % (r'$\varphi$'))
        plt.ylabel("Arrival %s [rad]" % (r'$\theta$'))

        if oversize_order == 0:
            title = 'Arrival %s of the $\gamma $(on lens[%.2f, %.2f])\n[$l_{scat}$ = %.2fm, $l_{abs}$ = %.2fm, $10^{%d}$ $\gamma$ from %dm away LED]' \
                % (mode, self.camera.pos_sphe[0], self.camera.pos_sphe[1], self.ice.sca_length, self.ice.abs_length, self.num_photon_order, self.dom.pos_diff[0])
            file_name = 'lens_all_arr_mom_camera_%.2f_%.2f_scat_%.2f_abs_%.2f_diff_%dm_order_%d.png' \
                            % (self.camera.pos_sphe[0], self.camera.pos_sphe[1], self.ice.sca_length, self.ice.abs_length, self.dom.pos_diff[0], self.num_photon_order)
        else:
            title = 'Arrival %s of the $\gamma $(on lens[%.2f, %.2f])\n[$l_{scat}$ = %.2fm, $l_{abs}$ = %.2fm, $10^{%d+%d}$ $\gamma$ from %dm away LED]' \
                                % (mode, self.camera.pos_sphe[0], self.camera.pos_sphe[1], self.ice.sca_length, 
                                self.ice.abs_length, self.num_photon_order, oversize_order, self.dom.pos_diff[0])
            file_name = 'lens_all_arr_mom_camera_%.2f_%.2f_scat_%.2f_abs_%.2f_diff_%dm_order_%d_plus_%d.png' \
                                % (self.camera.pos_sphe[0], self.camera.pos_sphe[1], self.ice.sca_length, 
                                self.ice.abs_length, self.dom.pos_diff[0], self.num_photon_order, oversize_order)
        plt.title(title, fontsize=fontsize-1)

        if zoom and gray:
            title = title.replace('on lens', 'on lens, gray')
            file_name = file_name.replace('lens', 'lens_zoom_gray')
        elif zoom:
            file_name = file_name.replace('lens', 'lens_zoom')
        elif gray:
            file_name = file_name.replace('lens', 'lens_gray')            

        if savefig:
            plt.savefig(self.output_dir + file_name, dpi=400)
        if not showfig:
            plt.close()


    
