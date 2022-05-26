import os
import time
import pathlib
import numpy as np
from scipy.io import *
from langevinfts import *
from saddle_net import *

class DeepLangevinFTS:
    def __init__(self, input_params):
        
        # Simulation Box
        nx = input_params['nx']
        lx = input_params['lx']
        
        # Polymer Chain
        n_contour   = input_params['chain']['n_contour']
        f           = input_params['chain']['f']
        chi_n       = input_params['chain']['chi_n']
        epsilon     = input_params['chain']['epsilon']
        chain_model = input_params['chain']['model']

        # Anderson Mixing
        am_n_var       = np.prod(nx).item()
        am_max_hist    = input_params['am']['max_hist']
        am_start_error = input_params['am']['start_error']
        am_mix_min     = input_params['am']['mix_min']
        am_mix_init    = input_params['am']['mix_init']
        
        # etc.
        self.verbose_level = input_params['verbose_level']

        # -------------- initialize ------------
        # choose platform among [cuda, cpu-mkl, cpu-fftw]
        factory = PlatformSelector.create_factory("cuda")

        # create polymer simulation instances
        self.sb     = factory.create_simulation_box(nx, lx)
        self.pc     = factory.create_polymer_chain(f, n_contour, chi_n, chain_model, epsilon)
        self.pseudo = factory.create_pseudo(self.sb, self.pc)
        self.am     = factory.create_anderson_mixing(am_n_var,
                      am_max_hist, am_start_error, am_mix_min, am_mix_init)
        
        # -------------- print simulation parameters ------------
        print("---------- Simulation Parameters ----------")
        print("Box Dimension: %d"  % (self.sb.get_dim()) )
        print("Precision: 8")
        print("chi_n: %f, f: %f, N: %d" % (self.pc.get_chi_n(), self.pc.get_f(), self.pc.get_n_contour()) )
        print("%s chain model" % (self.pc.get_model_name()) )
        print("Nx: %d, %d, %d" % (self.sb.get_nx(0), self.sb.get_nx(1), self.sb.get_nx(2)) )
        print("Lx: %f, %f, %f" % (self.sb.get_lx(0), self.sb.get_lx(1), self.sb.get_lx(2)) )
        print("dx: %f, %f, %f" % (self.sb.get_dx(0), self.sb.get_dx(1), self.sb.get_dx(2)) )
        print("Volume: %f" % (self.sb.get_volume()) )
        #print("Invariant Polymerization Index: %d" % (langevin_nbar) )
        print("Random Number Generator: ", np.random.RandomState().get_state()[0])
        
        # free end initial condition. q1 is q and q2 is qdagger.
        # q1 starts from A end and q2 starts from B end.
        self.q1_init = np.ones(self.sb.get_n_grid(), dtype=np.float64)
        self.q2_init = np.ones(self.sb.get_n_grid(), dtype=np.float64)
        
    def save_training_data(self, path, nbar, w_minus, g_plus, w_plus_diff):
        np.savez(path,
            nx=self.sb.get_nx(), lx=self.sb.get_lx(),
            N=self.pc.get_n_contour(), f=self.pc.get_f(), chi_n=self.pc.get_chi_n(),
            polymer_model=self.pc.get_model_name(), nbar=nbar,
            w_minus=w_minus.astype(np.float16),
            g_plus=g_plus.astype(np.float16),
            w_plus_diff=w_plus_diff.astype(np.float16))

    def save_simulation_data(self, path, w_plus, w_minus, phi_a, phi_b, dt, nbar):
        mdic = {"dim":self.sb.get_dim(), "nx":self.sb.get_nx(), "lx":self.sb.get_lx(),
            "N":self.pc.get_n_contour(), "f":self.pc.get_f(), "chi_n":self.pc.get_chi_n(),
            "chain_model":self.pc.get_model_name(),
            "dt": dt, "nbar":nbar,
            "random_generator":np.random.RandomState().get_state()[0],
            "random_seed":np.random.RandomState().get_state()[1],
            "w_plus":w_plus, "w_minus":w_minus, "phi_a":phi_a, "phi_b":phi_b}
        savemat(path, mdic)

    def make_training_data(self,
        w_plus, w_minus,
        saddle_max_iter, saddle_tolerance, saddle_tolerance_ref,
        dt, nbar, max_step, 
        path_dir, recording_period, recording_n_data,
        net=None):

        # training data directory
        if(path_dir):
            pathlib.Path(path_dir).mkdir(parents=True, exist_ok=True)

        # standard deviation of normal noise for single segment
        langevin_sigma = np.sqrt(2*dt*self.sb.get_n_grid()/ 
            (self.sb.get_volume()*np.sqrt(nbar)))

        # find saddle point 
        phi_a, phi_b, _, _, _, _, _ = DeepLangevinFTS.find_saddle_point(
            sb=self.sb, pc=self.pc, pseudo=self.pseudo, am=self.am,
            q1_init=self.q1_init, q2_init=self.q2_init, 
            w_plus=w_plus, w_minus=w_minus,
            max_iter=saddle_max_iter,
            tolerance=saddle_tolerance,
            verbose_level=self.verbose_level)

        #------------------ run ----------------------
        print("iteration, mass error, total_partition, energy_total, error_level")
        print("---------- Collect Training Data ----------")
        for langevin_step in range(1, max_step+1):
            print("Langevin step: ", langevin_step)
            
            # update w_minus
            normal_noise = np.random.normal(0.0, langevin_sigma, self.sb.get_n_grid())
            g_minus = phi_a-phi_b + 2*w_minus/self.pc.get_chi_n()
            w_minus += -g_minus*dt + normal_noise

            # find saddle point
            phi_a, phi_b, _, _, _, _, _ = DeepLangevinFTS.find_saddle_point(
                sb=self.sb, pc=self.pc, pseudo=self.pseudo, am=self.am, 
                q1_init=self.q1_init, q2_init=self.q2_init,
                w_plus=w_plus, w_minus=w_minus,
                max_iter=saddle_max_iter,
                tolerance=saddle_tolerance,
                verbose_level=self.verbose_level)
            w_plus_tol = w_plus.copy()
            g_plus_tol = phi_a + phi_b - 1.0

            # find more accurate saddle point
            phi_a, phi_b, _, _, _, _, _ = DeepLangevinFTS.find_saddle_point(
                sb=self.sb, pc=self.pc, pseudo=self.pseudo, am=self.am, 
                q1_init=self.q1_init, q2_init=self.q2_init,
                w_plus=w_plus, w_minus=w_minus,
                max_iter=saddle_max_iter,
                tolerance=saddle_tolerance_ref,
                verbose_level=self.verbose_level)
            w_plus_ref = w_plus.copy()
            phi_a_ref = phi_a.copy()
            phi_b_ref = phi_b.copy()
            
            # training data is sampled from random noise distribution
            # with various standard deviations
            if (langevin_step % recording_period == 0):
                log_std_w_plus = np.log(np.std(w_plus))
                log_std_w_plus_diff = np.log(np.std(w_plus_tol - w_plus_ref))
                diff_exps = np.linspace(log_std_w_plus, log_std_w_plus_diff, num=recording_n_data+2)[1:-1]
                #print(diff_exps)
                for std_idx, exp in enumerate(diff_exps):
                    std_w_plus_diff = np.exp(exp)
                    #print(std_w_plus_diff)
                    w_plus_noise = w_plus_ref + np.random.normal(0, std_w_plus_diff, self.sb.get_n_grid())
                    phi_a, phi_b, Q = self.pseudo.find_phi(
                            self.q1_init, self.q2_init,
                            w_plus_noise + w_minus,
                            w_plus_noise - w_minus)
                    g_plus = phi_a + phi_b - 1.0
                    
                    path = os.path.join(path_dir, "training_data_%d_%06d_%03d.npz" % (np.round(self.pc.get_chi_n()*100), langevin_step, std_idx))
                    self.save_training_data(path, nbar, w_minus, g_plus, w_plus_ref-w_plus_noise)
            
        # save final configuration to use it as input in actual simulation
        self.save_simulation_data(
            path="LastTrainingStep.mat", 
            w_plus=w_plus_ref, w_minus=w_minus,
            phi_a=phi_a_ref, phi_b=phi_b_ref, dt=dt, nbar=nbar)
            
    def run(self,
        w_plus, w_minus,
        saddle_max_iter, saddle_tolerance,
        dt, nbar, max_step, 
        path_dir=None, recording_period=10000, sf_computing_period=10, sf_recording_period=50000,
        net=None):

        # simulation data directory
        if(path_dir):
            pathlib.Path(path_dir).mkdir(parents=True, exist_ok=True)

        # standard deviation of normal noise
        langevin_sigma = np.sqrt(2*dt*self.sb.get_n_grid()/ 
            (self.sb.get_volume()*np.sqrt(nbar)))

        # find saddle point 
        phi_a, phi_b, _, _, _, _, _ = DeepLangevinFTS.find_saddle_point(
            sb=self.sb, pc=self.pc, pseudo=self.pseudo, am=self.am,
            q1_init=self.q1_init, q2_init=self.q2_init, 
            w_plus=w_plus, w_minus=w_minus,
            max_iter=saddle_max_iter,
            tolerance=saddle_tolerance,
            verbose_level=self.verbose_level)

        # structure function
        sf_average = np.zeros_like(np.fft.rfftn(np.reshape(w_minus, self.sb.get_nx())),np.float64)

        # init timers
        total_saddle_iter = 0
        total_time_neural_net = 0.0
        total_time_pseudo = 0.0
        total_net_failed = 0
        time_start = time.time()

        #------------------ run ----------------------
        print("iteration, mass error, total_partition, energy_total, error_level")
        print("---------- Run  ----------")
        for langevin_step in range(1, max_step+1):
            print("Langevin step: ", langevin_step)
            
            # update w_minus
            for w_step in ["predictor", "corrector"]:
                if w_step == "predictor":
                    w_minus_copy = w_minus.copy()
                    normal_noise = np.random.normal(0.0, langevin_sigma, self.sb.get_n_grid())
                    lambda1 = phi_a-phi_b + 2*w_minus/self.pc.get_chi_n()
                    w_minus += -lambda1*dt + normal_noise
                elif w_step == "corrector": 
                    lambda2 = phi_a-phi_b + 2*w_minus/self.pc.get_chi_n()
                    w_minus = w_minus_copy - 0.5*(lambda1+lambda2)*dt + normal_noise
                    
                (phi_a, phi_b, time_pseudo, time_neural_net, saddle_iter, error_level, is_net_failed) \
                    = DeepLangevinFTS.find_saddle_point(
                        sb=self.sb, pc=self.pc, pseudo=self.pseudo, am=self.am, 
                        q1_init=self.q1_init, q2_init=self.q2_init,
                        w_plus=w_plus, w_minus=w_minus,
                        max_iter=saddle_max_iter,
                        tolerance=saddle_tolerance,
                        verbose_level=self.verbose_level, net=net)
                total_time_pseudo += time_pseudo
                total_time_neural_net += time_neural_net
                total_saddle_iter += saddle_iter
                if (is_net_failed): total_net_failed += 1
                if (np.isnan(error_level) or error_level >= saddle_tolerance):
                    print("Could not satisfy tolerance")
                    break

            if (path_dir):
                # calcaluate structure function
                if langevin_step % sf_computing_period == 0:
                    sf_average += np.absolute(np.fft.rfftn(np.reshape(w_minus, self.sb.get_nx()))/self.sb.get_n_grid())**2

                # save structure function
                if langevin_step % sf_recording_period == 0:
                    sf_average *= sf_computing_period/sf_recording_period* \
                          self.sb.get_volume()*np.sqrt(nbar)/self.pc.get_chi_n()**2
                    sf_average -= 1.0/(2*self.pc.get_chi_n())
                    mdic = {"dim":self.sb.get_dim(), "nx":self.sb.get_nx(), "lx":self.sb.get_lx(),
                    "N":self.pc.get_n_contour(), "f":self.pc.get_f(), "chi_n":self.pc.get_chi_n(),
                    "chain_model":self.pc.get_model_name(),
                    "dt":dt, "nbar":nbar,
                    "structure_function":sf_average}
                    savemat(os.path.join(path_dir, "structure_function_%06d.mat" % (langevin_step)), mdic)
                    sf_average[:,:,:] = 0.0

                # save simulation data
                if (langevin_step) % recording_period == 0:
                    self.save_simulation_data(
                        path=os.path.join(path_dir, "fields_%06d.mat" % (langevin_step)),
                        w_plus=w_plus, w_minus=w_minus,
                        phi_a=phi_a, phi_b=phi_b, dt=dt, nbar=nbar)

        # estimate execution time
        time_duration = time.time() - time_start
        return total_saddle_iter, total_saddle_iter/max_step, time_duration/max_step, total_time_pseudo/time_duration, total_time_neural_net/time_duration, total_net_failed

    @staticmethod
    def find_saddle_point(sb, pc, pseudo, am, 
                q1_init, q2_init, 
                w_plus, w_minus,
                max_iter, tolerance,
                verbose_level=1, net=None):
                    
        # assign large initial value for the energy and error
        energy_total = 1e20
        error_level = 1e20

        # reset Anderson mixing module
        am.reset_count()

        # init timers
        time_neural_net = 0.0
        time_pseudo = 0.0
        is_net_failed = False
        
        # saddle point iteration begins here
        for saddle_iter in range(1,max_iter+1):
           
            # for the given fields find the polymer statistics
            time_p_start = time.time()
            phi_a, phi_b, Q = pseudo.find_phi(q1_init, q2_init,
                    w_plus+w_minus, w_plus-w_minus)
            time_pseudo += time.time() - time_p_start
            phi_plus = phi_a + phi_b
            
            # calculate output fields
            g_plus = phi_plus-1.0

            # error_level measures the "relative distance" between the input and output fields
            old_error_level = error_level
            error_level = np.sqrt(sb.inner_product(g_plus,g_plus)/sb.get_volume())
            if is_net_failed == False and error_level >= old_error_level:
               is_net_failed = True

            # print iteration # and error levels
            if(verbose_level == 2 or
             verbose_level == 1 and
             (error_level < tolerance or saddle_iter == max_iter)):
                 
                # calculate the total energy
                energy_old = energy_total
                energy_total  = -np.log(Q/sb.get_volume())
                energy_total += sb.inner_product(w_minus,w_minus)/pc.get_chi_n()/sb.get_volume()
                energy_total -= sb.integral(w_plus)/sb.get_volume()

                # check the mass conservation
                mass_error = sb.integral(phi_plus)/sb.get_volume() - 1.0
                print("%8d %12.3E %15.7E %15.9f %15.7E" %
                    (saddle_iter, mass_error, Q, energy_total, error_level))
                    
            # conditions to end the iteration
            if error_level < tolerance:
                break
           
            if net and not is_net_failed:
                # calculte new fields using neural network
                time_d_start = time.time()
                w_plus_diff = net.predict_w_plus(w_minus, g_plus, sb.get_nx()[-sb.get_dim():])
                w_plus += w_plus_diff
                time_neural_net += time.time() - time_d_start
            else:
                # calculte new fields using simple and Anderson mixing
                w_plus_out = w_plus + g_plus 
                am.caculate_new_fields(w_plus, w_plus_out, g_plus, old_error_level, error_level)
                
        sb.zero_mean(w_plus)
        return phi_a, phi_b, time_pseudo, time_neural_net, saddle_iter, error_level, is_net_failed
