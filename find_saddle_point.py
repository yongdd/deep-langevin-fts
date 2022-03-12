import time
import numpy as np
from langevinfts import *
from deep_fts import *

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
