import baseline.node as nodeClass
import baseline.priorityQueue as pqClass
import numpy as np
import baseline.abstractor as abstr
import baseline.config as config
import matplotlib.pyplot as plt


class Planner():
    """
    This class instantiates several data structures for planning.
    It use the abstractor class to make abstraction on the actions given
    in input.
    The planning algorithm is an A*-like algorithm and the abstractor
    use a Variational Auto-Encoder to do abstraction.

    Parameters
    ----------
        actions : list
            a list of actions with format
            (precondition, action, postcondition)

    Attributes
    ----------
        actions : list
            where the actions are saved
        last_goal : ndarray
            where the current goal is saved to intercept the moment when
            it changes
        abstractor : DynamicAbstractor
            this instance allows you to have the abstract actions to
            pass to the planner
        actions_dicts : dict
            where for each pair (i-th action, k-th level of abstraction)
            are saved all the actions with abstract precondition at the
            k-th level of abstraction compatible with the postcondition
            of the i-th abstract action at the k-th level of abstraction
        pre_post_different_for_abstractions : dict
            where a list is created for each level of abstraction that
            represents the actions that have precondition equal to the
            postcondition due to the abstraction

    """
    def __init__(self, actions=None):
        self.abstractor = abstr.DynamicAbstractor(actions)
        self.actions = self.abstractor.actions

        # data structures used to optimize
        self.last_goal = np.array([])
        self.actions_dicts = {}
        self.pre_post_different_for_abstractions = {}

        fig, axes = plt.subplots(1, 2)
        self.fig = fig
        self.axes = axes

    def plan(self, goal, start, actions=None, alg='mega'):
        """
        The planning is executed in the following way: it use the
        abstractor to transform the actions in abstracted actions of
        level 1 (where level 1 is the least abstraction);
        it tries to find a sequence of actions which get the current
        state to goal state; if it find it, then return the sequence,
        else augment the abstraction level.

        Parameters
        ----------
            goal : ndarray
                vector representing the desired state
            start : ndarray
                vector representing the current state
            actions : float
                a list of actions with format
                (precondition, action, postcondition)

        Attributes
        ----------
            last_goal : ndarray
                where the current goal is saved to allows the goal
                change detection
            plan_size : int
                maximum plan size

        Returns
        -------
            sequence : list
                actions list with format [(precondition, action,
                postcondition), ..., (precondition, action,
                postcondition)] where the first action represents the
                current state and the last action the desired state

        """
        if alg == 'mega':
            if not np.all(self.last_goal == goal):
                self.last_goal = goal
                self.plan_size = int(np.floor(config.sim['extrinsic_steps'] / config.exp['action_size']))
            else:
                self.plan_size -= 1

            if config.abst['type'] == 'filtered_mask':
                abstr_goal = self.abstractor.get_encoder().predict(np.reshape(goal, [-1, len(goal) * len(goal[0])]))[0][0]
                abstr_start = self.abstractor.get_encoder().predict(np.reshape(start, [-1, len(start) * len(start[0])]))[0][0]
                self.axes[0].imshow(start)
                self.axes[1].imshow(goal)
                self.axes[0].set_title("Current situation")
                self.axes[1].set_title("Goal")
                plt.savefig("planning_situation")

            elif config.abst['type'] == 'image':
                fore_goal = self.abstractor.background_subtractor(goal)
                fore_start = self.abstractor.background_subtractor(start)

                abstr_goal = self.abstractor.get_encoder().predict(np.reshape(fore_goal, [-1, len(goal) * len(goal[0])]))[0][0]
                abstr_start = self.abstractor.get_encoder().predict(np.reshape(fore_start, [-1, len(start) * len(start[0])]))[0][0]

                self.axes[0].imshow(start)
                self.axes[1].imshow(goal)
                self.axes[0].set_title("Current situation")
                self.axes[1].set_title("Goal")
                plt.savefig("planning_situation")

            else:
                abstr_goal = goal
                abstr_start = start

            plan = []
            for abstraction_level in range(config.abst['total_abstraction']):
                # print("Abstraction level: {}".format(abstraction_level))

                if not abstraction_level in self.pre_post_different_for_abstractions:
                    self.pre_post_different_for_abstractions[abstraction_level] = np.where(np.sum(
                                                                                  np.array(list(abs(np.take(self.actions, 0, axis=1) - np.take(self.actions, 2, axis=1))))
                                                                                  > self.abstractor.get_abstraction(abstraction_level), axis=1))[0]

                plan = self.forward_planning_with_prior_abstraction(abstr_goal, abstr_start, abstraction_level, self.plan_size)

                if plan:
                    return plan

            if not plan:
                return []

            return plan
        if alg == 'noplan':
            return []
        raise NotImplementedError

    def forward_planning_with_prior_abstraction(self, goal_image, current, lev_abstr, depth):
        """
        Forward planning algorithm divided into three steps:
        (1) It find all the actions have a precondition equal to the
            current state of the world and it put them in the frontier
            set,
        (2) it add for each action in the frontier set the actions with
            precondition compatible with the postcondition of the action
            in question,
        (3) when it find a condition like the goal condition in the
            frontier, it returns the sequence of correspondent actions

        Parameters
        ----------
            goal_image -: ndarray
                vector representing the desired abstract state at
                level lev_abstr
            current : ndarray
                vector representing the current abstract state at
                level 0 (to be precise)
            actions : float
                a list of actions with format (abstract precondition,
                action, abstract postcondition)
            lev_abstr : int
                abstraction level
            depth : int
                maximum sequence length

        Returns
        -------
            sequence - ndarray
                a list of actions with format [(precondition, action,
                postcondition),...,(precondition, action,
                postcondition)] where the first action represents the
                current state and the last action the desired state

        """

        abstraction_dists = self.abstractor.get_abstraction(lev_abstr)

        # checks whether the current condition is the same as the
        # desired condition in the current abstraction
        if np.all(abs(goal_image - current) <= abstraction_dists):
            return [[None, None, None]]

        # Initialize two priority queues. The first for nodes with a sequence less than the allowed depth, the second for nodes blocked due to having exceeded the depth limit
        # This allows you to restore the search tree in the event that a solution with depth less than that allowed is not found
        q = pqClass.PriorityQueue()

        frontier = set()
        post_goal_equals_for_abstractions = np.where(np.all(abs(np.array(list(np.take(self.actions, 2, axis=1))) - goal_image) <= abstraction_dists, axis=1))[0]
        s1 = set(self.pre_post_different_for_abstractions[lev_abstr])
        s2 = set(post_goal_equals_for_abstractions)
        # It takes actions with a precondition other than
        # postcondition in current abstraction
        s = s1.intersection(s2)

        if list(s):
            # It takes actions with precondition equal to the
            # current condition in current abstraction

            pre_current_equals_for_abstractions = np.where(np.all(abs(np.array(list(np.take(self.actions, 0, axis=1))) - current) <= abstraction_dists, axis=1))[0]
            s2 = set(pre_current_equals_for_abstractions)
            # It takes actions with a precondition other than
            # postcondition in current abstraction
            s = s2.intersection(s1)

            for i in s:
                node = nodeClass.Node(i, self.abstractor.get_dist(self.actions[i][2], goal_image), self.abstractor.get_dist(self.actions[i][0], self.actions[i][2]), None)
                q.enqueue(node, node.get_value_plus_cost())
                frontier.add(i)

        # print("Add {} initial states".format(len(frontier)))

        visited = set()
        while not q.is_empty():
            if len(visited) % 100 == 0 or len(visited) == 0:
                # print("Visited actions: {} Ready actions in queue: {}".format(len(visited), len(frontier)))
                pass

            # It takes the node with
            # less distance traveled + distance from the goal
            node, value = q.dequeue()

            if node.get_attribute() in visited:
                continue

            # Check if the current node is still at an allowable depth
            if node.get_depth() == depth + 1:
                frontier.remove(node.get_attribute())
                while node.get_father() is not None:
                    node = node.get_father()
                    if node.get_attribute() in visited:
                        visited.remove(node.get_attribute())
                continue

            # Check if the postcondition of the current node
            # corresponds to the goal
            if np.all(abs(self.actions[node.get_attribute()][2] - goal_image) <= abstraction_dists) and node.get_value() < self.abstractor.get_dist(current, goal_image):
                frontier.remove(node.get_attribute())
                visited.add(node.get_attribute())
                break

            frontier.remove(node.get_attribute())
            visited.add(node.get_attribute())

            post = self.actions[node.get_attribute()][2]

            # It takes the possible actions from the current node
            if (node.get_attribute(), lev_abstr) in self.actions_dicts:
                actions_list = self.actions_dicts[(node.get_attribute(), lev_abstr)]
            else:
                self.actions_dicts[(node.get_attribute(), lev_abstr)] = []
                pre_post_equals_for_abstractions = np.where(np.all(abs(np.array(list(np.take(self.actions, 0, axis=1))) - post) <= abstraction_dists, axis=1))[0]
                s1 = set(self.pre_post_different_for_abstractions[lev_abstr])
                s2 = set(pre_post_equals_for_abstractions)
                s = s2.intersection(s1)
                self.actions_dicts[(node.get_attribute(), lev_abstr)] = s

                actions_list = self.actions_dicts[(node.get_attribute(), lev_abstr)]

            actions_set = actions_list.difference(visited)

            # Create list of (i,action), where i represent i-th action in
            # self.actions and action the triple
            # (precondition, trajectory points, postcondition)
            f = lambda i: (i, self.actions[i])
            l2 = map(f, actions_set)

            for a, action in l2:
                pre1 = action[0]
                post1 = action[2]

                if not a in visited and not a in frontier:
                    node1 = nodeClass.Node(a, self.abstractor.get_dist(post1, goal_image), self.abstractor.get_dist(pre1, post1), node)
                    q.enqueue(node1, node1.get_value_plus_cost())
                    frontier.add(a)

                # In case the node was in the border, therefore still not
                # expanded and has a higher value of the child node than the
                # current one, then it is replaced in the queue with it
                elif a in frontier:
                    node1 = nodeClass.Node(a, self.abstractor.get_dist(post1, goal_image), self.abstractor.get_dist(pre1, post1), node)
                    q.replace_if_better(node1, node1.get_value_plus_cost())

        sequence = []
        if not visited or node is None or q.is_empty():
            return sequence

        if node is not None and not q.is_empty():
            while node.get_father() is not None:
                sequence += [[self.actions[node.get_attribute()][0], self.actions[node.get_attribute()][1], self.actions[node.get_attribute()][2]]] + sequence
                node = node.get_father()
            sequence = [[self.actions[node.get_attribute()][0], self.actions[node.get_attribute()][1], self.actions[node.get_attribute()][2]]] + sequence

        return sequence
