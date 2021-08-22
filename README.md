![REAL2021 banner](https://raw.githubusercontent.com/wiki/emilio-cartoni/REAL2021_starter_kit/images/banner2021.gif)
Starting kit for the [REAL2021 competition on Robot open-Ended Autonomous Learning](https://eval.ai/web/challenges/challenge-page/1134/overview).

Competition will start on August 23rd and will last up to mid December 2021.  

## Index
* [Install the software](#install-the-software)
* [How to launch the agent](#how-to-launch-the-baseline-agent)
* [Make your own submission](#make-your-own-submission)
* [How do I...?](#how-do-i)
* [Support](#support)

## Install the software
Pre-requisites: [Python](https://www.python.org/), [Git](https://git-scm.com/), [Conda (Anaconda or Miniconda)](https://www.anaconda.com/products/individual) and [Docker](https://www.docker.com/)   

1. Clone this repository:  
```bash
git clone https://github.com/emilio-cartoni/REAL2021_starter_kit.git  
cd REAL2021_starter_kit
```

2. Create a conda environment named "real_robots" from the provided `environment.yml`  
```
conda env create -f environment.yml
```
This could last a while as many libraries that are needed by the baseline will be installed, including Tensorflow.  
(It is possible to reduce the libraries in the environment if you do not want to run the baseline).  

## How to launch the baseline agent
1. Activate the real_robots conda environment  
```
conda activate real_robots
```
2. Launch the local evaluation:  
```
python local_evaluation.py
```

The default values will only launch a short evaluation, edit `local_evaluation.py` to launch full evaluations and customize the options.

## Make your own submission
To make submissions to the challenge, it is necessary to create and submit a docker container, which will contain all the dependencies to run your submission and the simulation.  
1. Launch `build.sh` to create a docker image for your submission.   
(The first build may take a long time since it will have to recreate the environment inside the container - subsequent build will be faster due to docker caching mechanism)
3. Go to [EvalAI Submit page](https://eval.ai/web/challenges/challenge-page/1134/submission) and follow the instructions there to submit the image as a submission for the REAL2021 competition.  


## How do I...?
For further informations, check our [Wiki](https://github.com/emilio-cartoni/REAL2021_starter_kit/wiki).  
In particular, if you want to know more about:
- the environment, see [here](https://github.com/emilio-cartoni/REAL2021_starter_kit/wiki/Environment)
- the options available for the environment and the simulation, see [here](https://github.com/emilio-cartoni/REAL2021_starter_kit/wiki/Environment-options)
- the docker image build, see [here](https://github.com/emilio-cartoni/REAL2021_starter_kit/wiki/Submissions)  
- to know more about the baseline, see [here](https://github.com/emilio-cartoni/REAL2021_starter_kit/wiki/Baseline)  

... or see the [FAQ](https://github.com/emilio-cartoni/REAL2021_starter_kit/wiki/FAQ) for common questions or errors.

## Support
For any problems or additional questions on the competition, feel free to post on [EvalAI forum](https://evalai-forum.cloudcv.org/) or [contact the organizers](mailto:emilio.cartoni@yahoo.it?subject=[REAL2021]%20Question).


**Best of Luck** :tada: :tada:
