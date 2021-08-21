import numpy as np
from real_robots.policy import BasePolicy
from baseline.planner import Planner
from baseline.abstractor import currentAbstraction
import baseline.config as config
import baseline.explorer as exp


class State:
    """
    This class represents a generic state that is used to implement the
    Baseline class behavior as a State Machine.

    Attributes
    ----------
        caller : obj
            a reference to the State caller (the Baseline object)
        actionsData : list
            parts of the action data are kept here until the action is
            complete and then saved into the Baseline allActions list.

    """
    caller = None
    actionData = []


class DoAction(State):
    """
    This class represents the state in which the agent performs an
    action. The Baseline stays in this state for a fixed number of
    timesteps, until the action is complete.

    Parameters
    ----------
        action : list
            the action that is being performed

    Attributes
    ----------
        n_timesteps : int
            action total duration (default = 1000)
        actionTimer : int
            steps taken of the action taken so far
        action : ndarray
            the action that is being performed

   """
    def __init__(self, action):
        self.n_timesteps = config.exp['action_size']
        self.actionTimer = -1
        self.action = action

    def step(self, observation, reward, done):
        """Return the action that is being performed.

        On the last timestep, the render flag is set to True so that
        an image of the results of the action can be stored by the
        subsequent EndAction state.

        Parameters
        ----------
            observation : dict
                observations received by the environment
            reward : float
                reward from the environment
            done : bool
                done flag from the environment

        Returns
        -------
            State
                the next State for the State Machine
            action
                action to be passed to the environment
            bool
                render flag

        """
        self.actionTimer += 1
        if self.actionTimer < self.n_timesteps:
            render = self.actionTimer == (self.n_timesteps - 1)
            if config.exp['action_parts_max'] > 1 and self.action is not None:
                current_action = self.action[self.actionTimer - 1]
            else:
                current_action = self.action
            return self, current_action, render
        else:
            nextState = EndAction()
            return nextState.step(observation, reward, done)


class EndAction(State):
    """
    This class represents the state immediately after the DoAction has
    ended. The action data (pre, action, post) is saved by calling the
    Baseline storeAction() method.

    """

    def step(self, observation, reward, done):
        """
        Save the performed action and pass to the ActionStart state.

        Parameters
        ----------
            observation : dict
                observations received by the environment
            reward : float
                reward from the environment
            done : bool
                done flag from the environment

        Returns
        -------
            State
                the next State for the State Machine
            action
                action to be passed to the environment
            bool
                render flag

        """
        post_pos = None
        post_image = None
        post_mask = None
        if config.sim['save_positions']:
            post_pos = observation['object_positions']
        if config.sim['save_images']:
            post_image = observation['retina']
        if config.sim['save_masks']:
            post_mask = observation['mask']

        post = (post_image, post_pos, post_mask)

        self.actionData += [post]
        self.caller.storeAction(self.actionData)

        State.actionData = []

        nextState = ActionStart()
        return nextState.step(observation, reward, done)


class ProposeNewAction(State):
    """
    In this state, the Explorer is called to generate a new action to be
    performed.

    """

    def step(self, observation, reward, done):
        """
        Save the starting condition of the action and then return a
        DoAction state to perform the action.

        Parameters
        ----------
            observation : dict
                observations received by the environment
            reward : float
                reward from the environment
            done : bool
                done flag from the environment

        Returns
        -------
            State
                the next State for the State Machine
            action
                action to be passed to the environment
            bool
                render flag

        """

        pre_pos = None
        pre_image = None
        pre_mask = None
        if config.sim['save_positions']:
            pre_pos = observation['object_positions']
        if config.sim['save_images']:
            pre_image = observation['retina']
        if config.sim['save_masks']:
            pre_mask = observation['mask']

        pre = (pre_image, pre_pos, pre_mask)

        action, debug = self.caller.explorer.selectNextAction()

        self.actionData += [pre, action]
        nextState = DoAction(action)
        return nextState.step(observation, reward, done)


class ActionStart(State):
    """
    In this state, a new action is started. If a goal is present, the
    action to be performed is decided by moving to PlanAction, otherwise
    the next state is ProposeNewAction, where the Explorer is called.

    """
    def step(self, observation, reward, done):
        """
        Check if the goal is contained in the observations, if there was
        then transit to the PlanAction state, otherwise go to the
        ProposeNewAction state

        Parameters
        ----------
            observation : dict
                observations received by the environment
            reward : float
                reward from the environment
            done : bool
                done flag from the environment

        Returns
        -------
            State
                the next State for the State Machine
            action
                action to be passed to the environment
            bool
                render flag

        """
        goal = observation['goal']
        if np.any(goal):
            nextState = PlanAction()
            return nextState.step(observation, reward, done)
        else:
            nextState = ProposeNewAction()
            return nextState.step(observation, reward, done)


class PlanAction(State):
    """
    This class represents the state in which the agent plans the
    sequence of actions to do

    """
    def step(self, observation, reward, done):
        """
        Use the Planner class to obtain an action sequence to be
        performed to achieve the required goal. If the sequence is not
        empty then go into the DoAction state with the first action
        of the sequence, otherwise wait for a new goal, or try a random
        action depending on the config file.

        Parameters
        ----------
            observation : dict
                observations received by the environment
            reward : float
                reward from the environment
            done : bool
                done flag from the environment

        Returns
        -------
            State
                the next State for the State Machine
            action
                action to be passed to the environment
            bool
                render flag

        """

        pre_pos = None
        pre_image = None
        pre_mask = None
        if config.sim['save_positions']:
            pre_pos = observation['object_positions']
        if config.sim['save_images']:
            pre_image = observation['retina']
        if config.sim['save_masks']:
            pre_mask = observation['mask']

        pre = (pre_image, pre_pos, pre_mask)

        goal = (observation['goal'], None, None)

        pre_abs = currentAbstraction(pre)
        goal_abs = currentAbstraction(goal)
        plan = self.caller.plan(goal_abs, pre_abs)

        if len(plan) > 0:
            action = plan[0][1]
            self.actionData += [pre, action]
            nextState = DoAction(action)
            return nextState.step(observation, reward, done)
        else:
            if config.plan['try_random']:
                nextState = ProposeNewAction()
                return nextState.step(observation, reward, done)
            else:
                nextState = WaitForNewGoal(observation['goal'])
                return nextState.step(observation, reward, done)


class WaitForNewGoal():
    """
    In this State the agent waits for a new goal.

    Parameters
    ----------
        goal_abs : ndarray
            The current goal.

    Attributes
    ----------
        current_goal : ndarray

    """
    def __init__(self, goal_abs):
        self.current_goal = goal_abs

    def step(self, observation, reward, done):
        """
        It is checked whether the goal contained in the observations is
        different from the current_goal attribute.
        If it is, then a new goal has been received and the next state
        will be an ActionStart.

        Parameters
        ----------
            observation : dict
                observations received by the environment
            reward : float
                reward from the environment
            done : bool
                done flag from the environment

        Returns
        -------
            State
                the next State for the State Machine
            action
                action to be passed to the environment
            bool
                render flag

        """
        goal = observation['goal']

        sameGoal = np.all(goal == self.current_goal)

        if sameGoal:
            return self, None, False

        nextState = ActionStart()
        return nextState.step(observation, reward, done)

class Baseline(BasePolicy):
    """
    Main class of the baseline

    Parameters
    ----------

    Attributes
    ----------
        allActions : list
            list of all actions performed during the intrinsic phase
        state : State
            current state of the State Machine
        planner : Planner
            planner object used in the Extrinsic phase
        goal :
            current goal
        plan_sequence : list
            stored planned sequence to be executed
        explorer : Explorer
            Explorer object used to propose new actions during the
            Intrinsic phase.

    """
    def __init__(self, action_space, observation_space):
        self.allActions = []
        self.state = ActionStart()
        State.caller = self
        self.planner = None
        self.goal = None
        self.plan_sequence = []
        self.action_type = list(action_space.spaces.keys())[0]
        action_parameter_space = action_space[self.action_type]
        self.explorer = exp.RandomExploreAgent(action_parameter_space)

    def storeAction(self, actionData):
        """
        Save the data of the action just performed to the 'allActions'
        list.

        Parameters
        ----------
            actionData : tuple
                actionData contains the pre-conditon, the action and the
                post-condition. The pre and post conditions contain the
                observation received from the environment at the start
                and at the end of the action.

        """
        self.allActions += [actionData]

    def save(self, fileName):
        """
        Save the list of all actions performed during the intrinsic
        phase in a file

        Parameters
        ----------
            fileName : str
                name of the file

        """
        if config.sim['compressed_data']:
            np.savez_compressed(fileName, self.allActions)
        else:
            np.save(fileName, self.allActions)

    def step(self, observation, reward, done):
        """
        Return the action to be performed in the environment based on
        the current observation and the inner state of the policy.

        Parameters
        ----------
            observation : dict
                observations received by the environment
            reward : float
                reward from the environment
            done : bool
                done flag from the environment

        Returns
        -------
            State
                the next State for the State Machine
            action
                action to be passed to the environment
            bool
                render flag

        """
        self.state, action, render = self.state.step(observation, reward, done)
        return {self.action_type: action, 'render': render}

    def plan(self, goal_abs, pre_abs):
        """
        Use the Planner class to plan the list of actions to be
        performed to reach the goal

        Parameters
        ----------
            goal_abs (numpy.ndarray): desidered state of the world
            pre_abs (numpy.ndarray): current state of the world

                Returns
        -------
            plan_sequence (list): list of actions planned by the planner
        """
        goalChanged = not(np.all(self.goal == goal_abs))
        self.goal = goal_abs
        no_sequence = len(self.plan_sequence) == 0

        if config.plan['replan'] or goalChanged or no_sequence:
            plannerType = config.plan['type']
            self.plan_sequence = self.planner.plan(goal_abs, pre_abs,
                                                   alg=plannerType)
        else:
            self.plan_sequence = self.plan_sequence[1:]

        return self.plan_sequence

    def end_intrinsic_phase(self, observation, reward, done):
        """
        Perform a final step of the State machine to receive and store
        the last observation from the environment, then save all the
        actions performed.

        """
        self.step(observation, reward, done)
        fileID = np.random.randint(0, 1000000)
        self.save("./transitions_{}".format(fileID))

    def start_extrinsic_phase(self):
        """
        Instantiate the Planner class with the actions collected in the
        intrinsic phase, or with the actions saved in the file written
        in the config file

        """

        allActions = self.allActions

        if config.sim['use_experience_data']:
            if config.sim['compressed_data']:
                allActions = np.load(config.sim['experience_data'],
                                     allow_pickle=True)['arr_0']
            else:
                allActions = np.load(config.sim['experience_data'],
                                     allow_pickle=True)


        # filter actions where the arm did not go back home
        firstRow = allActions[0][0][0][0,:]

        def validAction(action):
            ok_pre = np.all(action[0][0][0, :] == firstRow)
            ok_post = np.all(action[2][0][0, :] == firstRow)
            return ok_pre and ok_post

        allActions = [action for action in allActions if validAction(action)]


        allAbstractedActions = [[currentAbstraction(a[0]), a[1],
                                 currentAbstraction(a[2])]
                                for a in allActions]

        self.planner = Planner(allAbstractedActions)
        del allActions
        del allAbstractedActions

    def start_extrinsic_trial(self):
        """ Start a new trial with an ActionStart state."""
        self.state = ActionStart()

    def end_extrinsic_trial(self, observation, reward, done):
        """ Collect the last observation of the trial."""
        self.step(observation, reward, done)
