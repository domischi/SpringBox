from .illustration import plot_data
import json
import numpy as np
import sys

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
        _run.log_scalar("pv_avg_x-vx", pV_avg[0]-_config['window_velocity'][0], sim_info['time_step_index'])
        _run.log_scalar("pv_avg_y-vy", pV_avg[1]-_config['window_velocity'][1], sim_info['time_step_index'])

    if fVs is not None:
        ## Max Velocity
        ind_fVs_max = np.argmax(np.linalg.norm(fVs,axis=1))
        _run.log_scalar("vmax_x", fVs[ind_fVs_max,0], sim_info['time_step_index'])
        _run.log_scalar("vmax_y", fVs[ind_fVs_max,1], sim_info['time_step_index'])
        _run.log_scalar("vmax_x-vx", fVs[ind_fVs_max,0]-_config['window_velocity'][0], sim_info['time_step_index'])
        _run.log_scalar("vmax_y-vy", fVs[ind_fVs_max,1]-_config['window_velocity'][1], sim_info['time_step_index'])

        _run.log_scalar("x_vmax", fXs[ind_fVs_max,0], sim_info['time_step_index'])
        _run.log_scalar("y_vmax", fXs[ind_fVs_max,1], sim_info['time_step_index'])
        _run.log_scalar("x_vmax_c", fXs[ind_fVs_max,0]-(sim_info['x_max']+sim_info['x_min'])/2, sim_info['time_step_index'])
        _run.log_scalar("y_vmax_c", fXs[ind_fVs_max,1]-(sim_info['x_max']+sim_info['x_min'])/2, sim_info['time_step_index'])
        
        ## Avg velocity
        fV_avg = np.mean(fVs, axis=0)
        _run.log_scalar("fv_avg_x", fV_avg[0], sim_info['time_step_index'])
        _run.log_scalar("fv_avg_y", fV_avg[1], sim_info['time_step_index'])
        _run.log_scalar("fv_avg_x-vx", fV_avg[0]-_config['window_velocity'][0], sim_info['time_step_index'])
        _run.log_scalar("fv_avg_y-vy", fV_avg[1]-_config['window_velocity'][1], sim_info['time_step_index'])
        
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
            json.dump(d,f)
            ex.add_artifact(dump_file_loc)

    if plotting_this_iteration:
        plot_data(pXs, pVs, fXs, fVs,
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
