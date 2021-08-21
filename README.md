# REAL2021_starter_kit
Starting kit for the REAL2021 competition on Robot open-Ended Autonomous Learning.

Competition will start on August 23rd.  **Starting kit will be populated shortly before.**

## Index
* [Install the software](#install-the-software)
* [How to launch the agent](#how-to-launch-the-agent)
* [Make your own submission](#make-your-own-submission)

## Install the software
Pre-requisites: [Python](https://www.python.org/), [git](https://git-scm.com/),[Anaconda](https://www.anaconda.com/products/individual) 

1. Clone this repository:
`git clone https://github.com/emilio-cartoni/REAL2021_starter_kit.git`
`cd REAL2021_starter_kit`
2. Create a conda environment named "real_robots" from the provided `environment.yml`
`conda env create -f environment.yml`

## How to launch the baseline agent
1. Activate the real_robots conda environment
`conda activate real_robots`
2. Launch the local evaluation:
`python local_evaluation.py`

The default values will only launch a short evaluation, edit `local_evaluation.py` to launch full evaluations and customize the options.

## Make your own submission
1. Launch `build.sh` to create a docker image for your submission.
2. Go to [EvalAI Submit page](https://eval.ai/web/challenges/challenge-page/1134/submission) and follow the instructions there to submit the image as a submission for the REAL2021 competition.

To know more ...
