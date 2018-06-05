import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from utility import Camera, DOM, Ice, pass_pressure_vessel, get_sensor_filter, calculate_inter_angle_to_camera
from utility import cart_to_sphe, sphe_to_cart, get_total_hist2d
import copy
import itertools


class PhotonInteraction:
    """It will calculate expected photon track and check with camera lens

    """
    def __init__(self, dom, camera, ice, num_photon_order):
        """Init

        Args:
            dom (DOM) : DOM parameters like radius, thickness and refractive index
            camera (Camera) : camera parameters like position, rad, FOV
            ice (Ice) : ice parameters like scattering, absorption length and refractive index
            num_photon_order (float) : order of total number of emitted photon
        """
        self.dom = copy.deepcopy(dom)
        self.camera = copy.deepcopy(camera)
        self.ice = copy.copy(ice)
        self.num_photon_order = num_photon_order

    def set_src_dir(self, dir):
        """Set the source directory which include results from PPC

        Args:
            dir (string) : src directory
        """
        self.src_dir = dir

    def set_output_dir(self, dir):
        """Set the output directory which plots will be stored

        Args:
            dir (string) : output directory
        """
        self.output_dir = dir

    def get_single_data(self):
        """Get first data corresponding to the pos_diff option
        Because the file size is too large, the file separated into several files
        Maximum size of each file is 256MB

        """
        # Get photon information from first file at current setting
        file_name = "ppc_log_x_%.2f_y_%.2f_z_%.2f_order_%d_0.dat" \
                            % (self.dom.pos_diff[0], self.dom.pos_diff[1], self.dom.pos_diff[2], self.num_photon_order)
        self.photon_log = np.fromfile(self.src_dir + file_name, dtype='<f4').reshape(-1, 8)
        self.orig_momentum= self.photon_log[:,5:3:-1] 
        self.orig_position = self.photon_log[:,7:5:-1]

    def get_total_hist2d(self, num_files, bins=40, maxbin=False):
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

        file_name = "ppc_log_x_%.2f_y_%.2f_z_%.2f_order_%d_0.dat" \
                            % (self.dom.pos_diff[0], self.dom.pos_diff[1], self.dom.pos_diff[2], self.num_photon_order)
        self.total_hist, self.total_hist_x_edges, self.total_hist_y_edges = get_total_hist2d(file_name, num_files, 
                            bins=bins,
                            r0=self.dom.radius, 
                            d0=self.dom.thickness,
                            camera_pos_sphe=self.camera.pos_sphe,
                            camera_distance=self.camera.pos_distance,
                            camera_lens_rad=self.camera.lens_rad,
                            wv=405,
                            FOV=self.camera.FOV,
                            src_dir=self.src_dir)
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


    def pass_pressure_vessel(self):
        """Pass pressure vessel with cosideration of refraction in two steps
        1. Ice -> Pressure vessel
        2. Pressure_vessel -> Inner side of DOM
        With this function, we can get momentum and position after refraction
        Some photons cannot pass into the DOM because of the refractive index
        """
        momentum, position = pass_pressure_vessel(self.orig_position, self.orig_momentum, 
                                                    r0=self.dom.radius,
                                                    d0=self.dom.thickness,
                                                    n0=self.ice.index, 
                                                    n1=self.dom.vessel_index, 
                                                    n2=self.dom.inner_index)
        self.refracted_momentum = momentum
        self.refracted_position = position

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
        title = 'Arrival %s of the photon(on the DOM surface)\n[$l_{scat}$ = %.2fm, $l_{abs}$ = %.2fm, $10^{%d}$ photons from %dm away LED]' \
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
        title = 'Arrival %s of the photon(on lens)\n[$l_{scat}$ = %.2fm, $l_{abs}$ = %.2fm, $10^{%d}$ photons from %dm away LED]' \
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
                                    figsize=(12,6),
                                    fontsize=15,
                                    bins=40,
                                    xlim=[],
                                    ylim=[],
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
        title = 'Arrival %s of the photon(on lens[%.2f, %.2f])\n[$l_{scat}$ = %.2fm, $l_{abs}$ = %.2fm, $10^{%d}$ photons from %dm away LED]' \
            % (mode, self.camera.pos_sphe[0], self.camera.pos_sphe[1], self.ice.sca_length, self.ice.abs_length, self.num_photon_order, self.dom.pos_diff[0])
        file_name = 'lens_all_arr_mom_camera_%.2f_%.2f_scat_%.2f_abs_%.2f_diff_%dm_order_%d.png' \
                        % (self.camera.pos_sphe[0], self.camera.pos_sphe[1], self.ice.sca_length, self.ice.abs_length, self.dom.pos_diff[0], self.num_photon_order)
        plt.title(title, fontsize=fontsize-1)

        if zoom and gray:
            title = title.replace('on lens', 'on lens, gray')
            file_name = file_name.replace('lens', 'lens_gray')

        if savefig:
            plt.savefig(self.output_dir + file_name, dpi=400)
        if not showfig:
            plt.close()


    
