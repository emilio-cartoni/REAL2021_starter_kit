import yaml
import os
import time

cfg = None
sim = None
abst = None
plan = None
exp = None


def save(score):
    strings = time.strftime("%Y,%m,%d,%H,%M,%S")
    t = strings.split(',')
    numbers = [int(x) for x in t]
    filename = "config_{}_{}_{}_{}_{}.yaml".format(numbers[2], numbers[1],
                                                   numbers[0], numbers[3],
                                                   numbers[4])
    fullpath = sim['data_dir'] + os.sep + filename
    with open(fullpath, 'w') as outfile:
        yaml.dump(str(cfg) + "\n" + str(score), outfile)


def load(configFile):
    with open(configFile, 'r') as ymlfile:
        global cfg, sim, abst, plan, exp
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        sim = cfg['simulation']
        abst = cfg['abstraction']
        plan = cfg['planner']
        exp = cfg['exploration']

load("baseline/config.yaml")
