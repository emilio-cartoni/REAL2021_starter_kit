import numpy as np
import real_robots
from my_controller import SubmittedPolicy
from interval import interval
import yaml

with open('options.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    EVALUATION_ACTION_TYPE = cfg['EVALUATION_ACTION_TYPE']
    EVALUATION_N_OBJECTS = cfg['EVALUATION_N_OBJECTS']
    envs = {'easy': 'R1', 'hard' : 'R2'}
    ENVIRONMENT = envs[cfg['ENVIRONMENT_TYPE']]

DATASET_PATH = "./data/goals-REAL2020-s2021-25-15-10-%s.npy.npz" % EVALUATION_N_OBJECTS

result, detailed_scores = real_robots.evaluate(
                SubmittedPolicy,
                environment=ENVIRONMENT,
                action_type=EVALUATION_ACTION_TYPE,
                n_objects=EVALUATION_N_OBJECTS,
                intrinsic_timesteps=0e6, # Full intrinsic phase: 15e6
                extrinsic_timesteps=10e3, # Extrinsic phase 10e3 for each trial
                extrinsic_trials=50, # Full extrinsic phase: 50 trials
                visualize=False, # Render the environment in a window
                goals_dataset_path=DATASET_PATH
               , video = (True, True, True) # Record a video (intrinsic, extrinsic, debug info)
            )

print(result)
print(detailed_scores)

result_data = {'result': result,
               'detailed_scores': detailed_scores}

run_id = np.random.randint(0, 1000000)

np.save("./R{}-{:.3f}".format(run_id, result['score_total']), result_data)
