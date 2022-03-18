import os
import time
import numpy as np
from scipy.io import *
from langevinfts import *
from deep_fts import *

def save_data(sb, pc, path, nbar, w_minus, g_plus, w_plus_diff):
    np.savez( path,
        nx=sb.get_nx(), lx=sb.get_lx(), N=pc.get_n_contour(), f=pc.get_f(), chi_n=pc.get_chi_n(),
        polymer_model=pc.get_model_name(), n_bar=nbar,
        w_minus=w_minus.astype(np.float16),
        g_plus=g_plus.astype(np.float16),
        w_plus_diff=w_plus_diff.astype(np.float16))

def make_training_data(
    sb, pc, pseudo, am, 
    q1_init, q2_init, 
    phi_a, phi_b, w_plus, w_minus,
    saddle_max_iter, saddle_tolerance, saddle_tolerance_ref,
    max_iter, dt, nbar,
    path_dir,
    recording_period=5, recording_n_data=3,
    verbose_level=1, net=None):

    # standard deviation of normal noise for single segment
    langevin_sigma = np.sqrt(2*dt*sb.get_n_grid()/ 
        (sb.get_volume()*np.sqrt(nbar)))

    # find saddle point 
    find_saddle_point(
        sb=sb, pc=pc, pseudo=pseudo, am=am, 
        q1_init=q1_init, q2_init=q2_init,
        phi_a=phi_a, phi_b=phi_b,
        w_plus=w_plus, w_minus=w_minus,
        max_iter=saddle_max_iter,
        tolerance=saddle_tolerance,
        verbose_level=verbose_level)

    #------------------ run ----------------------
    print("iteration, mass error, total_partition, energy_total, error_level")
    print("---------- Data Collection ----------")
    for langevin_step in range(1, max_iter+1):
        print("Langevin step: ", langevin_step)
        # update w_minus
        normal_noise = np.random.normal(0.0, langevin_sigma, sb.get_n_grid())
        g_minus = phi_a-phi_b + 2*w_minus/pc.get_chi_n()
        w_minus += -g_minus*dt + normal_noise
        sb.zero_mean(w_minus)

        # find saddle point
        find_saddle_point(
            sb=sb, pc=pc, pseudo=pseudo, am=am, 
            q1_init=q1_init, q2_init=q2_init,
            phi_a=phi_a, phi_b=phi_b,
            w_plus=w_plus, w_minus=w_minus,
            max_iter=saddle_max_iter,
            tolerance=saddle_tolerance,
            verbose_level=verbose_level)
        w_plus_tol = w_plus.copy()
        g_plus_tol = phi_a + phi_b - 1.0

        # find more accurate saddle point
        find_saddle_point(
            sb=sb, pc=pc, pseudo=pseudo, am=am, 
            q1_init=q1_init, q2_init=q2_init,
            phi_a=phi_a, phi_b=phi_b,
            w_plus=w_plus, w_minus=w_minus,
            max_iter=saddle_max_iter,
            tolerance=saddle_tolerance_ref,
            verbose_level=verbose_level)
        w_plus_ref = w_plus.copy()

        # record data
        if (langevin_step % recording_period == 0):
            log_std_w_plus = np.log(np.std(w_plus))
            log_std_w_plus_diff = np.log(np.std(w_plus_tol - w_plus_ref))
            diff_exps = np.linspace(log_std_w_plus, log_std_w_plus_diff, num=recording_n_data+2)[1:-1]
            #print(diff_exps)
            for idx, exp in enumerate(diff_exps):
                std_w_plus_diff = np.exp(exp)
                #print(std_w_plus_diff)
                w_plus_noise = w_plus_ref + np.random.normal(0, std_w_plus_diff, sb.get_n_grid())
                QQ = pseudo.find_phi(
                        phi_a, phi_b,
                        q1_init, q2_init,
                        w_plus_noise + w_minus,
                        w_plus_noise - w_minus)
                g_plus = phi_a + phi_b - 1.0
                
                path = os.path.join(path_dir, "fields_%d_%06d_%03d.npz" % (np.round(pc.get_chi_n()*100), langevin_step, idx))
                save_data(sb, pc, path, nbar, w_minus, g_plus, w_plus_ref-w_plus_noise)


def run_langevin_dynamics(
    sb, pc, pseudo, am, 
    q1_init, q2_init, 
    phi_a, phi_b, w_plus, w_minus,
    saddle_max_iter, saddle_tolerance,
    max_iter, dt, nbar,
    path_dir=None,
    recording_period = 10000, 
    sf_computing_period = 10,
    sf_recording_period= 50000,
    verbose_level=1, net=None):

    # standard deviation of normal noise for single segment
    langevin_sigma = np.sqrt(2*dt*sb.get_n_grid()/ 
        (sb.get_volume()*np.sqrt(nbar)))

    # find saddle point 
    find_saddle_point(
        sb=sb, pc=pc, pseudo=pseudo, am=am,
        q1_init=q1_init, q2_init=q2_init,
        phi_a=phi_a, phi_b=phi_b,
        w_plus=w_plus, w_minus=w_minus,
        max_iter=saddle_max_iter,
        tolerance=saddle_tolerance,
        verbose_level=verbose_level)

    # structure factor
    sf_average = np.zeros_like(np.fft.rfftn(np.reshape(w_minus, sb.get_nx()[:sb.get_dim()])),np.float64)

    # init timers
    total_saddle_iter = 0
    total_time_neural_net = 0.0
    total_time_pseudo = 0.0
    total_net_failed = 0
    time_start = time.time()

    #------------------ run ----------------------
    print("iteration, mass error, total_partition, energy_total, error_level")
    print("---------- Run  ----------")
    for langevin_step in range(1, max_iter+1):
        print("Langevin step: ", langevin_step)
        
        # update w_minus
        for w_step in ["predictor", "corrector"]:
            if w_step == "predictor":
                w_minus_copy = w_minus.copy()
                normal_noise = np.random.normal(0.0, langevin_sigma, sb.get_n_grid())
                lambda1 = phi_a-phi_b + 2*w_minus/pc.get_chi_n()
                w_minus += -lambda1*dt + normal_noise
            elif w_step == "corrector": 
                lambda2 = phi_a-phi_b + 2*w_minus/pc.get_chi_n()
                w_minus = w_minus_copy - 0.5*(lambda1+lambda2)*dt + normal_noise
                
            sb.zero_mean(w_minus)
            (time_pseudo, time_neural_net, saddle_iter, error_level, is_net_failed) \
                = find_saddle_point(
                    sb=sb, pc=pc, pseudo=pseudo, am=am, 
                    q1_init=q1_init, q2_init=q2_init,
                    phi_a=phi_a, phi_b=phi_b,
                    w_plus=w_plus, w_minus=w_minus,
                    max_iter=saddle_max_iter,
                    tolerance=saddle_tolerance,
                    verbose_level=verbose_level, net=net)
            total_time_pseudo += time_pseudo
            total_time_neural_net += time_neural_net
            total_saddle_iter += saddle_iter
            if (is_net_failed): total_net_failed += 1
            if (np.isnan(error_level) or error_level >= saddle_tolerance):
                print("Could not satisfy tolerance")
                break;

        if (path_dir):
            # calcaluate structure factor
            if ( langevin_step % sf_computing_period == 0):
                sf_average += np.absolute(np.fft.rfftn(np.reshape(w_minus, sb.get_nx()[:sb.get_dim()]))/sb.get_n_grid())**2

            # save structure factor
            if ( langevin_step % sf_recording_period == 0):
                sf_average *= sf_computing_period/sf_recording_period* \
                      sb.get_volume()*np.sqrt(nbar)/pc.get_chi_n()**2
                sf_average -= 1.0/(2*pc.get_chi_n())
                mdic = {"dim":sb.get_dim(), "nx":sb.get_nx(), "lx":sb.get_lx(),
                "N":pc.get_n_contour(), "f":pc.get_f(), "chi_n":pc.get_chi_n(),
                "chain_model":pc.get_model_name(),
                "langevin_dt":dt, "langevin_nbar":nbar,
                "structure_factor":sf_average}
                savemat(os.path.join(path_dir, "structure_factor_%06d.mat" % (langevin_step)), mdic)
                sf_average[:,:,:] = 0.0

            # save simulation data
            if( (langevin_step) % recording_period == 0 ):
                mdic = {"dim":sb.get_dim(), "nx":sb.get_nx(), "lx":sb.get_lx(),
                "N":pc.get_n_contour(), "f":pc.get_f(), "chi_n":pc.get_chi_n(),
                "chain_model":pc.get_model_name(),
                "langevin_dt": dt, "langevin_nbar":nbar,
                "w_plus":w_plus, "w_minus":w_minus, "phi_a":phi_a, "phi_b":phi_b}
                savemat(os.path.join(path_dir, "fields_%06d.mat" % (langevin_step)), mdic)

    # estimate execution time
    time_duration = time.time() - time_start; 
    return total_saddle_iter, total_saddle_iter/max_iter, time_duration/max_iter, total_time_pseudo/time_duration, total_time_neural_net/time_duration, total_net_failed

def find_saddle_point(sb, pc, pseudo, am, 
            q1_init, q2_init, 
            phi_a, phi_b, w_plus, w_minus,
            max_iter=100, tolerance=1e-4,
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
        QQ = pseudo.find_phi(phi_a, phi_b, 
                q1_init,q2_init,
                w_plus + w_minus,
                w_plus - w_minus)
        time_pseudo += time.time() - time_p_start
        phi_plus = phi_a + phi_b
        
        # calculate output fields
        g_plus = phi_plus-1.0

        # error_level measures the "relative distance" between the input and output fields
        old_error_level = error_level
        error_level = np.sqrt(sb.inner_product(g_plus,g_plus)/sb.get_volume())
        if(is_net_failed == False and error_level >= old_error_level):
           is_net_failed = True

        # print iteration # and error levels
        if(verbose_level == 2 or
         verbose_level == 1 and
         (error_level < tolerance or saddle_iter == max_iter)):
             
            # calculate the total energy
            energy_old = energy_total
            energy_total  = -np.log(QQ/sb.get_volume())
            energy_total += sb.inner_product(w_minus,w_minus)/pc.get_chi_n()/sb.get_volume()
            energy_total -= sb.integral(w_plus)/sb.get_volume()

            # check the mass conservation
            mass_error = sb.integral(phi_plus)/sb.get_volume() - 1.0
            print("%8d %12.3E %15.7E %13.9f %13.9f" %
                (saddle_iter, mass_error, QQ, energy_total, error_level))
                
        # conditions to end the iteration
        if(error_level < tolerance):
            break;
       
        if (net and not is_net_failed):
            # calculte new fields using neural network
            time_d_start = time.time()
            w_plus_diff = net.predict_w_plus(w_minus, g_plus, sb.get_nx()[:sb.get_dim()])
            w_plus += w_plus_diff
            sb.zero_mean(w_plus)
            time_neural_net += time.time() - time_d_start
        else:
            # calculte new fields using simple and Anderson mixing
            w_plus_out = w_plus + g_plus 
            sb.zero_mean(w_plus_out)
            am.caculate_new_fields(w_plus, w_plus_out, g_plus, old_error_level, error_level);
    return time_pseudo, time_neural_net, saddle_iter, error_level, is_net_failed
