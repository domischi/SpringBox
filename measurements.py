from illustration import plot_data
import json
import numpy as np
import sys

def do_measurements(ex, _config, _run, sim_info, pXs, pVs, acc, ms, fXs, fVs, plotting_this_iteration, save_all_data_this_iteration):
    _run.log_scalar("Ratio activated", sum(acc)/len(acc), sim_info['time_step_index'])
    _run.log_scalar("Mass ratio activated", sum(ms*acc)/sum(ms), sim_info['time_step_index'])
    if fVs is not None:
        ind_fVs_max = np.argmax(np.linalg.norm(fVs,axis=1))
        _run.log_scalar("vmax_x", fVs[ind_fVs_max,0], sim_info['time_step_index'])
        _run.log_scalar("vmax_y", fVs[ind_fVs_max,1], sim_info['time_step_index'])
        _run.log_scalar("x_vmax", fXs[ind_fVs_max,0], sim_info['time_step_index'])
        _run.log_scalar("y_vmax", fXs[ind_fVs_max,1], sim_info['time_step_index'])
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
