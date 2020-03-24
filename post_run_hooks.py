from illustration import generate_video_from_png
def post_run_hooks(ex, _config, _run, working_folder):
    if _config['MAKE_VIDEO']:
        video_path = generate_video_from_png(working_folder)
        ex.add_artifact(video_path, name=f"video.avi")
