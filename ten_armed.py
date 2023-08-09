"""
script for implementing a multi-armed bandit problem. based on chapter two of
sutton and barto
"""
import os
from pathlib import Path
import numpy as np
from scipy.integrate import simps
import random
from matplotlib import pyplot as plt
from matplotlib import rc, cm
import webbrowser
from distutils.spawn import find_executable

if find_executable('latex'):
    rc('text', usetex=True)


# --------------------------------------------------------------------------- #
def write_path(dir_loc, filename=''):
    """
    given the location of a directory (either absolute or relative), this
    function will return an os-agnostic path to that directory. if a file name
    has been provided, it will be appended
    """
    # write the object-agnostic path to the specified directory
    path_written = Path(os.path.abspath(dir_loc))
    # make sure this location exists
    assert path_written.exists(), '\n\td\'oh! the location pointed to by ' + \
                                  str(path_written) + \
                                  ' doesn\'t actually exist!\n'
    # if the location does exist, make sure it's actually a directory
    assert path_written.is_dir(), '\n\td\'oh! the location pointed to by ' + \
                                  str(path_written) + \
                                  ' isn\'t actually a directory!\n'
    # if a file name has been provide it, append it to the path
    if filename:
        path_written = path_written.joinpath(filename)
    # return the ful path created
    return path_written


# --------------------------------------------------------------------------- #
def create_directory(desired_location):
    """
    given the desired location of a new directory (either as a relative or an
    absolute path, this function will create the directory, along with any
    necessary parent directories. if the directory already exists, nothing will
    happen. the full path to the newly created directory is return
    """
    # write the full path to where the directory is to be created
    path_to_directory = Path(os.path.abspath(desired_location))
    # create a directory at this location. if any of the parent directories are
    # missing, create them too
    path_to_directory.mkdir(parents=True, exist_ok=True)
    # return the path
    return path_to_directory


# --------------------------------------------------------------------------- #
def compute_gaussian_pdf(x, mu, sigma):
    """
    for the value(s) x, this function computes discrete values of the pdf
    corresponding to the gaussian distribution with mean mu and standard
    deviation sigma
    """
    # if it's not already, convert x to a numpy array
    x = np.array(x)
    # compute the the pdf
    pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (
                sigma * np.sqrt(2.0 * np.pi))
    # check to make sure this is actually a pdf
    integral_of_pdf = simps(pdf, x)
    machine_eps = np.finfo(float).eps
    assert integral_of_pdf - 1.0 < 1e3 * machine_eps, \
        '\n\tthe integral of computed pdf isn\'t summing to one. might ' + \
        '\n\tneed to extend the range of abscissas, x.' + \
        '\n\t  - computed integral of pdf: ' + str(integral_of_pdf)
    # return the probability values
    return pdf


# --------------------------------------------------------------------------- #
def create_reward_std_list(identical_stds, reward_stds, n_actions):
    """
    given the user inputs for defining the standard deviations for the reward
    distributions (i.e. the identical_stds boolean, the list of reward standard
    deviations, and the number of available actions), this function will return
    a verified list of standard deviations. recall, these are the standard
    deviations used to define the gaussian distributions from which each
    action's reward is drawn
    """
    # check to make sure the input type are correct
    assert isinstance(identical_stds, bool), '\n\tidentical_stds must be ' + \
                                             'a boolean\n'
    assert isinstance(reward_stds, (list, int, float, np.ndarray)), \
        '\n\treward_stds must be either a list, int, or float\n'
    # if the given standard deviation(s) are not in a list or array, then we
    # are dealing with either an int or a float. in this case, put the number
    # in a list of its own
    if not isinstance(reward_stds, (list, np.ndarray)):
        reward_stds_as_list = [reward_stds]
    else:
        # otherwise, record the list or array as is, assuming different
        # standard deviations between reward distributions is allowed
        if not identical_stds:
            reward_stds_as_list = reward_stds
        else:
            # otherwise, just out the first value given and put it in a list
            reward_stds_as_list = [reward_stds[0]]
    # create the list of standard deviations accordingly
    if identical_stds:
        reward_std_list = n_actions * reward_stds_as_list
    else:
        # in this case, the original list will just be returned, assuming it is
        # of the right length
        assert len(reward_stds_as_list) == n_actions
        # and contains the correct types
        for reward_std in reward_stds_as_list:
            assert isinstance(reward_std, (int, float))
        # copy and rename the list
        reward_std_list = list(reward_stds_as_list)
    # return the list of standard deviations
    return reward_std_list


# --------------------------------------------------------------------------- #
def plot_reward_distributions(action_values, reward_stds, auto_open=True,
                              plots_directory='.'):
    """
    given the true action values associated with each action and the desired
    standard deviations to be used with each of the reward distributions, this
    function will create a plot showing all the reward pdfs associated with
    each action
    """
    # print a header message to the screen
    print('\n\t- plotting the distributions from which rewards will be ' +
          'drawn for each available action')
    # plotting preliminaries
    plot_name = 'reward distribution'
    the_fontsize = 14
    plt.figure(plot_name)
    # create a long, fine mesh over which to to plot the distributions (this
    # mesh will be truncated later)
    q_star_min = min(action_values) - max(reward_stds)
    q_star_max = max(action_values) + max(reward_stds)
    q_star_span = q_star_max - q_star_min
    q_star_overall_min = q_star_min - 3 * q_star_span
    q_star_overall_max = q_star_max + 3 * q_star_span
    n_points_per_curve = 2000
    x = np.linspace(q_star_overall_min, q_star_overall_max, n_points_per_curve)
    # make a list of colors, one for each action
    colors = cm.rainbow_r(np.linspace(0, 1, n_actions))
    # get machine zero
    machine_eps = np.finfo(float).eps
    # plotting
    for i in range(n_actions):
        # for each available action, pull out the corresponding mean and
        # standard deviation for the reward distribution
        reward_mean = action_values[i]
        reward_std = reward_stds[i]
        # compute the pdf describing this action's reward distribution
        reward_dist = compute_gaussian_pdf(x, reward_mean, reward_std)
        # pull out the indices where the pdf is non negligible
        indices_to_keep = np.where(reward_dist > 1e8 * machine_eps)
        # pull out the abscissas and ordinates at these indices
        x_reward = x[indices_to_keep]
        prob_reward = reward_dist[indices_to_keep]
        # plot the distribution
        plt.plot(x_reward, prob_reward, color=colors[i],
                 label='$A_t=a_{' + str(i + 1) + '}$')
        # write the expected reward for this action above the curve
        y_lims = plt.ylim()
        text_padding = (y_lims[1] - y_lims[0]) / 75
        q_star_str = str(round(reward_mean, 2))
        plt.text(reward_mean - 20 * text_padding,
                 max(reward_dist) + text_padding,
                 '$q_*(a_{' + str(i + 1) + '})=' + q_star_str + '$',
                 fontsize=the_fontsize - 6)
    # label the x axis and write the title
    plt.xlabel('$R_t$', fontsize=the_fontsize)
    plt.title('$reward\; distributions\colon\; \\textrm{PDF}s\; f\! or\; ' +
              'R_t \\vert A_t$', fontsize=the_fontsize)
    plt.legend(loc='best')
    # create the plots directory, if it doesn't already exist
    path_to_plots = create_directory(plots_directory)
    # write the file name for the plot and the corresponding full path
    file_name = plot_name + '.png'
    path_to_file = path_to_plots.joinpath(file_name)
    path_to_cwd = os.getcwd()
    relative_file_path = str(path_to_file).replace(path_to_cwd, '')
    relative_file_path = relative_file_path.lstrip('\\').lstrip('/')
    # save and close the figure
    print('\n\t\t' + 'saving figure ... ', end='')
    plt.savefig(path_to_file, dpi=300)
    print('done.\n\t\tfigure saved: ' + relative_file_path)
    plt.close(plot_name)
    # open the saved image, if desired
    if auto_open:
        webbrowser.open(path_to_file)


# --------------------------------------------------------------------------- #
def random_method(actions_list):
    """
    given a list of actions, this function randomly selects one
    """
    # make sure the input is either a list or a numpy array
    assert isinstance(actions_list, (list, np.ndarray)), '\n\tactions_list' + \
                                                         'must be either a ' +\
                                                         'list or a numpy ' + \
                                                         'array\n'
    # make sure there is at least one action
    assert len(actions_list) > 1, '\n\tat least one action must be available\n'
    # randomly pick an action
    selected_action = random.choice(actions_list)
    # return the selected action
    return selected_action


# --------------------------------------------------------------------------- #
def greedy_method(actions_list, action_value_estimates):
    """
    given a list of available actions and the associated estimates of action-
    value, this function selects the action in a fully exploitative, greedy way
    """
    # being greedy means always picking the action that has the highest
    # action-value estimate. if there happens to be a tie for the action
    # with the greatest action value, break the tie by randomly choosing
    # between those actions with the highest action-value estimates. start
    # by converting the list of action-value estimates to a numpy array
    action_value_estimates = np.array(action_value_estimates)
    # find the largest action-value estimate
    max_action_value = max(action_value_estimates)
    # pull out the indices where this max value is found
    max_indices = np.where(action_value_estimates == max_action_value)[0]
    # if the max appears at more than one index, break the tie by choosing
    # one of indices at random
    if len(max_indices) > 1:
        max_index = random.choice(max_indices)
    else:
        max_index = max_indices[0]
    # select the corresponding action from the list
    selected_action = actions_list[max_index]
    # return the selected action
    return selected_action


# --------------------------------------------------------------------------- #
def select_action(actions_list, action_value_estimates, selection_strategy='',
                  epsilon=0.1):
    """
    given a list of available actions and the corresponding action-value
    estimates, this subroutine picks an action based on the specified selection
    strategy. available selection strategies include 'random', 'greedy', and
    'eps-greedy' (requires specifying epsilon)
    """
    # make sure an action-selection strategy has been specified and that it is
    # a valid choice
    valid_strategies = ['random', 'greedy', 'eps-greedy']
    assert selection_strategy in valid_strategies, '\n\tplease select a ' + \
                                                   'valid selection strategy\n'
    # initialize the selected_action variable to suppress warning
    selected_action = None
    # deploy the appropriate policy, based on the desired selection strategy
    if selection_strategy == 'random':
        # use the random policy
        selected_action = random_method(actions_list)
    if selection_strategy == 'greedy':
        # use the greedy policy
        selected_action = greedy_method(actions_list, action_value_estimates)
    if selection_strategy == 'eps-greedy':
        # with the epsilon-greedy selection strategy, a greedy policy is used
        # (1-eps)*100% of the time. for the other eps*100% of the time, a
        # random policy is used. start by checking to see if a valid value of
        # epsilon has been provided
        assert isinstance(epsilon, (float, int)), '\n\tepsilon must be ' + \
                                                  'either a float or an int\n'
        assert 0.0 <= epsilon <= 1.0, '\n\tepsilon must in the range [0, 1]\n'
        # decide whether this timestep will be exploitative (greedy) or
        # exploratory (random)
        selection_modes = ['exploit', 'explore']
        mode_probabilities = [1.0 - epsilon, epsilon]
        selection_mode = random.choices(selection_modes, mode_probabilities)[0]
        # deploy the right policy, based on the chosen mode
        if selection_mode == 'exploit':
            selected_action = greedy_method(actions_list,
                                            action_value_estimates)
        else:
            selected_action = random_method(actions_list)
    # return the selected action
    return selected_action


# --------------------------------------------------------------------------- #
def collect_reward(selected_action, actions_list, action_values, reward_stds):
    """
    this function returns the reward associated with a given action. in the
    case of the multi-armed bandit, the rewards are drawn from gaussian
    distributions, whose means are the predetermined action values and whose
    standard deviations have been user-specified
    """
    # pull out the index of the selected action
    action_index = actions_list.index(selected_action)
    # pull out the corresponding reward distribution's mean and std
    reward_mean = action_values[action_index]
    reward_std = reward_stds[action_index]
    # sample the corresponding distribution
    reward = np.random.normal(reward_mean, reward_std)
    # return the reward
    return reward


# --------------------------------------------------------------------------- #
def update_action_counts(selected_action, action_counts, actions_list):
    """
    at the end of the given timestep, this function increments the counter
    associated with the selected action
    - input:
      - selected_action:            the action selected
      - action_counts:              the list keeping track of how many times
                                    each action has been taken
      - actions_list:               the ordered list of available actions
    - output:
      - action_counts:              the list keeping track of how many times
                                    each action has been taken with an updated
                                    value for the selected action
    """
    # find the index of the selected action
    action_index = actions_list.index(selected_action)
    # increment the count at the corresponding index
    action_counts[action_index] += 1
    # return the updated list
    return action_counts


# --------------------------------------------------------------------------- #
def update_action_value_estimates(selected_action, collected_reward,
                                  action_value_estimates, action_counts,
                                  actions_list):
    """
    this function updates the action-value estimate for the selected action at
    each timestep. it does this by sample-averaging the observed reward with
    the existing estimate. in order to be memory-efficient and computationally
    frugal, the incremental implementation of the averaging formula is used
    - input:
      - selected_action:            the action selected
      - collected_reward:           the reward observed
      - action_value_estimates:     the list of action-value estimates at the
                                    start of the timestep
      - action_counts:              the list keeping track of how many times
                                    each action has been taken
      - actions_list:               the ordered list of available actions
    - output:
      - action_value_estimates:     the list of action-value estimates with an
                                    updated value for the selected action
    """
    # find the index of the selected action
    action_index = actions_list.index(selected_action)
    # pull out the current estimate for this action's value
    action_value_old = action_value_estimates[action_index]
    # pull out the count of how many times this action has been taken
    action_count = action_counts[action_index]
    # use the the incremental implementation of the averaging formula, which
    # take the form of a general update rule, to compute the new action-value
    # estimate for the selected action. see section 2.4 of sutton and barto
    step_size = 1.0 / action_count
    error = collected_reward - action_value_old
    action_value_new = action_value_old + step_size * error
    # replace the old action-value estimate with the updated one in the given
    # list of action-value estimates
    action_value_estimates[action_index] = action_value_new
    # return the updated
    return action_value_estimates


# --------------------------------------------------------------------------- #
def plot_rewards_collected(rewards_collected, auto_open=True,
                           plots_directory='.'):
    """
    given a list of the reward collected at each timestep during a run, this
    function plots a time trace of accumulated reward
    """
    # print a header message to the screen
    print('\n\t- plotting the time history of accumulated reward')
    # count the number of timesteps
    n_timesteps = len(rewards_collected)
    # make a list of the timesteps (the abscissas)
    timesteps = np.arange(1, n_timesteps + 1)
    # compute a cumulative sum of the recorded rewards
    reward_accumulated = np.cumsum(rewards_collected)
    # plotting preliminaries
    plot_name = 'accumulated reward'
    the_fontsize = 14
    fig = plt.figure(plot_name)
    # stretch the plotting window
    width, height = fig.get_size_inches()
    fig.set_size_inches(1.75 * width, 1.0 * height, forward=True)
    # plotting
    # plot the immediate reward
    plt.subplot(2, 1, 1)
    for i in range(n_timesteps):
        if rewards_collected[i] >= 0:
            plt.plot(timesteps[i], rewards_collected[i], 'k.', ms=2)
        else:
            plt.plot(timesteps[i], rewards_collected[i], 'r.', ms=2)
    # label the axes
    plt.xticks([], [])
    plt.ylabel('$R_t$', fontsize=the_fontsize)
    # plot the accumulated reward
    plt.subplot(2, 1, 2)
    plt.plot(timesteps, reward_accumulated, 'b-', linewidth=0.2)
    for i in range(n_timesteps):
        if reward_accumulated[i] >= 0:
            plt.plot(timesteps[i], reward_accumulated[i], 'k.', ms=2)
        else:
            plt.plot(timesteps[i], reward_accumulated[i], 'r.', ms=2)
    # label the axes
    plt.xlabel('$t$', fontsize=the_fontsize)
    plt.ylabel('$\sum_1^t R_t$', fontsize=the_fontsize)
    plt.suptitle('$immediate \; and\; accumulated \; reward$',
                 fontsize=the_fontsize)
    # create the plots directory, if it doesn't already exist
    path_to_plots = create_directory(plots_directory)
    # write the file name for the plot and the corresponding full path
    file_name = plot_name + '.png'
    path_to_file = path_to_plots.joinpath(file_name)
    path_to_cwd = os.getcwd()
    relative_file_path = str(path_to_file).replace(path_to_cwd, '')
    relative_file_path = relative_file_path.lstrip('\\').lstrip('/')
    # save and close the figure
    print('\n\t\t' + 'saving figure ... ', end='')
    plt.savefig(path_to_file, dpi=300)
    print('done.\n\t\tfigure saved: ' + relative_file_path)
    plt.close(plot_name)
    # open the saved image, if desired
    if auto_open:
        webbrowser.open(path_to_file)


# --------------------------------------------------------------------------- #
def plot_actions_selected(actions_selected, actions_list, auto_open=True,
                          plots_directory='.'):
    """
    given a list of the action selected at each timestep during a run, this
    function plots a time history of the selection process
    """
    # print a header message to the screen
    print('\n\t- plotting the time history of selected actions')
    # count the number of timesteps
    n_timesteps = len(actions_selected)
    # make a list of the timesteps (the abscissas)
    timesteps = np.arange(1, n_timesteps + 1)
    # create a list of action numbers from the list action names
    action_numbers_selected = [int(action_name.split('#')[-1]) for action_name
                               in actions_selected]
    # make a list of colors, one for each action
    n_actions = len(actions_list)
    colors = cm.rainbow_r(np.linspace(0, 1, n_actions))
    # plotting preliminaries
    plot_name = 'selected actions'
    the_fontsize = 14
    fig = plt.figure(plot_name)
    # stretch the plotting window
    width, height = fig.get_size_inches()
    fig.set_size_inches(1.75 * width, 1.0 * height, forward=True)
    # plotting
    for i in range(n_timesteps):
        action_number = action_numbers_selected[i]
        action_number_index = action_number - 1
        plt.plot(timesteps[i], action_number, marker='.', ms=1,
                 color=colors[action_number_index])
    # set the y-axis ticks
    action_indices = list(range(1, n_actions+1))
    y_tick_labels = ['$a_{'+str(index)+'}$' for index in action_indices]
    plt.yticks(action_indices, y_tick_labels)
    # label the axes
    plt.xlabel('$t$', fontsize=the_fontsize)
    plt.ylabel('$A_t$', fontsize=the_fontsize)
    plt.title('$selected \; actions$', fontsize=the_fontsize)
    # create the plots directory, if it doesn't already exist
    path_to_plots = create_directory(plots_directory)
    # write the file name for the plot and the corresponding full path
    file_name = plot_name + '.png'
    path_to_file = path_to_plots.joinpath(file_name)
    path_to_cwd = os.getcwd()
    relative_file_path = str(path_to_file).replace(path_to_cwd, '')
    relative_file_path = relative_file_path.lstrip('\\').lstrip('/')
    # save and close the figure
    print('\n\t\t' + 'saving figure ... ', end='')
    plt.savefig(path_to_file, dpi=300)
    print('done.\n\t\tfigure saved: ' + relative_file_path)
    plt.close(plot_name)
    # open the saved image, if desired
    if auto_open:
        webbrowser.open(path_to_file)


# --------------------------------------------------------------------------- #
# [user input] specify the number of arms the bandit has
n_actions = 10

# [user input] specify the standard deviations for each of the reward
# distributions. that is, we are assuming the reward for a given action is a
# gaussian random variable, with a mean at the action value. if all the reward
# distributions have constant standard deviations, set the boolean accordingly
# and only a single value needs to be provided
identical_stds = False
# reward_stds = 1.4
reward_stds = np.random.rand(n_actions) + 1

# [user input] provide an initial guess for the action values. this value will
# be used as the initial action-value estimate for all available actions
initial_action_value_estimate = 0.0

# [user input] specify an action-selection strategy
# choose from: 'random', 'greedy', 'eps-greedy'
selection_strategy = 'eps-greedy'

# [user input] if using an epsilon-greedy strategy, then specify epsilon
epsilon = 0.1

# [user input] specify the desired number of timesteps per run
n_timesteps_per_run = 1000

# describe the action space by creating an (ordered) list holding the names of
# all available actions
actions_list = ['action #' + str(i + 1) for i in range(n_actions)]

# randomly the generate the true action values associated with each arm. draw
# each value randomly from a scaled, standard normal distribution. n.b. the
# standard normal distribution one with zero mean and unit standard deviation
action_values = 5 * np.random.normal(0, 1, n_actions)

# create a list of reward-distribution standard deviations
reward_stds = create_reward_std_list(identical_stds, reward_stds, n_actions)

# the reward associated with each action will be drawn from a normal
# distribution centered at the true action value (computed above) and with unit
# standard deviation. plot the reward distribution associated with each action
plot_reward_distributions(action_values, reward_stds, plots_directory='plots')

# initialize the list of action-value estimates using the user-defined value
action_value_estimates = [initial_action_value_estimate
                          for _ in range(n_actions)]
# initialize a list keeping track of the number of times each action was taken
action_counts = [0] * n_actions

# initialize some variables for storing running quantities
actions_selected = []
rewards_collected = []
verbose = True
# simulate a run
for t in range(1, n_timesteps_per_run + 1):
    # select an action. the multi-armed bandit problem is nonassociative, i.e.
    # the state never changes. so, evaluation at each timestep can begin here
    selected_action = select_action(actions_list, action_value_estimates,
                                    selection_strategy=selection_strategy,
                                    epsilon=epsilon)
    # determine the reward associated with the selected action
    collected_reward = collect_reward(selected_action, actions_list,
                                      action_values, reward_stds)
    # update the count for the selected action
    action_counts = update_action_counts(selected_action, action_counts,
                                         actions_list)
    # and update the action-value estimate for the selected action
    action_value_estimates = \
        update_action_value_estimates(selected_action, collected_reward,
                                      action_value_estimates, action_counts,
                                      actions_list)
    # for plotting and record-keeping, make a note of the action and reward
    actions_selected.append(selected_action)
    rewards_collected.append(collected_reward)
    # if desired, print some details about the current timestep to the screen
    if verbose:
        print('\n\t- timestep #' + str(t))
        print('\t  - action selected:   ', selected_action)
        print('\t  - collected reward:  ', round(collected_reward, 2))
        print('\t  - accumulated reward:', round(sum(rewards_collected), 2))
        print('\t  - Q(a) =', str(np.round(action_value_estimates, 2)))

# plot a time history of accumulated reward
plot_rewards_collected(rewards_collected)
# plot a time history of the selected actions
plot_actions_selected(actions_selected, actions_list)
