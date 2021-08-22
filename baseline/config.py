import yaml

cfg = None
sim = None
abst = None
plan = None
exp = None


def loadBaselineCfg(configFile):
    with open(configFile, 'r') as ymlfile:
        global cfg, sim, abst, plan, exp
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        sim = cfg['simulation']
        abst = cfg['abstraction']
        plan = cfg['planner']
        exp = cfg['exploration']


def loadOptions(optionsFile):
    with open(optionsFile, 'r') as ymlfile:
        options = yaml.load(ymlfile, Loader=yaml.FullLoader)
        if options['EVALUATION_ACTION_TYPE'] == 'macro_action':
            exp['action_size'] = 1000
            exp['action_parts_max'] = 1
            exp['home_duration'] = 0
        elif options['EVALUATION_ACTION_TYPE'] == 'joints':
            exp['action_size'] = 1000
            exp['action_parts_max'] = 10
            exp['home_duration'] = 150
        else:
            raise Exception('Baseline autoconfig only supports macro_action'
                            + ' and joints control')
    print("Baseline autoconfig:", cfg)


loadBaselineCfg("baseline/config.yml")
if exp['autoconfig']:
    loadOptions("options.yml")
