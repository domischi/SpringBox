from .illustration import plot_data_w_fluid, plot_mixing
from .activation import activation_fn_dispatcher
import json
import numpy as np
import sys
from scipy.spatial import Delaunay

def do_measurements(ex, _config, _run, sim_info, pXs, pVs, acc, ms, fXs, fVs, plotting_this_iteration, save_all_data_this_iteration):
    if acc is not None:
        _run.log_scalar("Ratio activated", sum(acc)/len(acc), sim_info['time_step_index'])
        _run.log_scalar("Mass ratio activated", sum(ms*acc)/sum(ms), sim_info['time_step_index'])
        _run.log_scalar("Mass activated", sum(ms*acc), sim_info['time_step_index'])

    if pVs is not None:
        ## Avg velocity
        pV_avg = np.mean(pVs, axis=0)
        _run.log_scalar("pv_avg_x", pV_avg[0], sim_info['time_step_index'])
        _run.log_scalar("pv_avg_y", pV_avg[1], sim_info['time_step_index'])

    if fVs is not None:
        ## Max Velocity
        ind_fVs_max = np.argmax(np.linalg.norm(fVs,axis=1))
        _run.log_scalar("vmax_x", fVs[ind_fVs_max,0], sim_info['time_step_index'])
        _run.log_scalar("vmax_y", fVs[ind_fVs_max,1], sim_info['time_step_index'])

        _run.log_scalar("x_vmax", fXs[ind_fVs_max,0], sim_info['time_step_index'])
        _run.log_scalar("y_vmax", fXs[ind_fVs_max,1], sim_info['time_step_index'])
        _run.log_scalar("x_vmax_c", fXs[ind_fVs_max,0]-(sim_info['x_max']+sim_info['x_min'])/2, sim_info['time_step_index'])
        _run.log_scalar("y_vmax_c", fXs[ind_fVs_max,1]-(sim_info['x_max']+sim_info['x_min'])/2, sim_info['time_step_index'])
        
        ## Avg velocity
        fV_avg = np.mean(fVs, axis=0)
        _run.log_scalar("fv_avg_x", fV_avg[0], sim_info['time_step_index'])
        _run.log_scalar("fv_avg_y", fV_avg[1], sim_info['time_step_index'])

        ## Avg velocity in activated area
        w = activation_fn_dispatcher(_config, sim_info['t'])(fXs)
        fV_acc_avg = np.average(fVs ,weights=w, axis=0)
        _run.log_scalar("fv_acc_avg_x", fV_acc_avg[0], sim_info['time_step_index'])
        _run.log_scalar("fv_acc_avg_y", fV_acc_avg[1], sim_info['time_step_index'])
    if save_all_data_this_iteration:
        d= {
                'pXs' : pXs.tolist() ,
                'pVs' : pVs.tolist() ,
                'acc' : acc.tolist() ,
                'ms'  : ms.tolist() ,
                'fXs' : fXs.tolist() ,
                'fVs' : fVs.tolist() ,
                'sim_info' : sim_info
           }
        dump_file_loc = f"{sim_info['data_dir']}/data_dump-{sim_info['time_step_index']}.json"
        with open(dump_file_loc, 'w') as f:
            json.dump(d,f, indent=4)
            ex.add_artifact(dump_file_loc)

    if plotting_this_iteration:
        if _config.get('mixing_experiment',False):
            plot_mixing(pXs,
                  sim_info,
                  image_folder=sim_info['data_dir'],
                  title=f"t={sim_info['t']:.3f}",
                  L=_config['L'],
                  fix_frame=True,
                  SAVEFIG=_config['SAVEFIG'],
                  ex=ex)
        else:
            plot_data_w_fluid(pXs, pVs, fXs, fVs,
                      sim_info,
                      image_folder=sim_info['data_dir'],
                      title=f"t={sim_info['t']:.3f}",
                      L=_config['L'],
                      fix_frame=True,
                      SAVEFIG=_config['SAVEFIG'],
                      ex=ex,
                      plot_particles=True,
                      plot_fluids=True,
                      side_by_side=True,
                      fluid_plot_type = 'quiver')

def do_one_timestep_correlation_measurement(ex, _config, _run, sim_info, pXs, pXs_old):
    assert(pXs.shape==pXs_old.shape)
    p1 = pXs.flatten()
    p2 = pXs_old.flatten()
    corr = np.dot(p1,p2)/(np.linalg.norm(p1)*np.linalg.norm(p2))
    _run.log_scalar("One timestep correlator", corr, sim_info['time_step_index'])
    return corr

def get_mixing_score(pXs, _config, sim_info):
    # https://github.com/danielegrattarola/spektral/blob/master/spektral/datasets/delaunay.py
    tri = Delaunay(pXs)
    edges_explicit = np.concatenate((tri.vertices[:, :2],
                                     tri.vertices[:, 1:],
                                     tri.vertices[:, ::2]), axis=0)

    adj = np.zeros((len(pXs), len(pXs)))
    adj[edges_explicit[:, 0], edges_explicit[:, 1]] = 1.
    adj_matrix = np.clip(adj + adj.T, 0, 1) 
    v = np.ones(len(pXs))
    v[:len(pXs)//2] = -1.
    ret = - np.dot(v,np.dot(adj_matrix,v))  ## How much mixing in total
    ret /= len(edges_explicit) ## how much per bond
    ret += 1 ## only give a positive score (to encourage human players)
    return ret
