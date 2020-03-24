from illustration import plot_data

def do_measurements(ex, _config, _run, sim_info, pXs, pVs, acc, ms, fXs, fVs, plotting_this_iteration):
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
