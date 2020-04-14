import sys 
import os
import shutil
from pathlib import Path
import json
import time
import sacred
from example import ex

if len(sys.argv)<2:
    raise RuntimeError('Did not provide a data path... Aborting')
BASEPATH = sys.argv[-1]+'/'

def get_one_queued(basedir, n_attempts = 3):
    if n_attempts == 0:
        print('Could not find any new queued runs.... Exiting.')
        return None
    queue_base_dir = basedir
    queued_run_folders = [queue_base_dir + f for f in  os.listdir(queue_base_dir) ]
    if len(queued_run_folders)==0:
        return None
    for our_folder in queued_run_folders:
        ## only continue with executition if queued:
        try:
            with open(our_folder+'/run.json', 'r') as f:
                status = json.load(f)['status']
            if status != 'QUEUED':
                continue
        except FileNotFoundError:
            continue
        ## Try and enter the first folder:
        try:
            Path(our_folder+'/lock').touch(exist_ok=False)
        except FileExistsError:
            # Entered a folder that another process is already working on... retry...
            continue

        ## Now we are sure that this is our folder, and we can work on this.
        config = sacred.config.load_config_file(our_folder+'/config.json')
        qid = our_folder.split('/')[-1]
        shutil.rmtree(our_folder)
        return qid, config
    return None

def execute_one_queued(basedir):
    ret = get_one_queued(basedir)
    if ret is None:
        return False
    else:
        qid, config = ret
        print(f"Loaded Queue ID {qid}")
        ex.run(config_updates=config)
        print(f"Successfully executed Queue ID {qid}")
        return True

## Execute one job:
#execute_one_queued(BASEPATH)

## Execute as many as you want:
while(execute_one_queued(BASEPATH)):
    pass
