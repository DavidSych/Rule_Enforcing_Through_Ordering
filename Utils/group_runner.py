import numpy as np
from Environment.queue import Queue
import argparse, datetime, shutil
import pickle, random, os
import torch
from PPO.ppo import PPOLearner
from BRS.brs import BRSAgent
from Utils.tabular import TabularAgent
from Utils.misc import *

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser()
# TF params
parser.add_argument("--seed", default=1, type=int, help="Random seed for reproducibility.")
parser.add_argument("--threads", default=15, type=int, help="Number of CPU threads to use.")
parser.add_argument("--hidden_layer_actor", default=4, type=int, help="Size of the hidden layer of the network.")
parser.add_argument("--hidden_layer_critic", default=32, type=int, help="Size of the hidden layer of the network.")
parser.add_argument("--actor_learning_rate", default=3e-4, type=float, help="Learning rate.")
parser.add_argument("--critic_learning_rate", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--entropy_weight", default=1e-3, type=float, help="Entropy regularization constant.")
parser.add_argument("--l2", default=0, type=float, help="L2 regularization constant.")
parser.add_argument("--clip_norm", default=0.1, type=float, help="Gradient clip norm.")

parser.add_argument("--buffer_len", default=20_000, type=int, help="Number of time steps to train on.")
parser.add_argument("--epsilon", default=0.05, type=float, help="Clipping constant.")
parser.add_argument("--gamma", default=1., type=float, help="Return discounting.")
parser.add_argument("--_lambda", default=0.97, type=float, help="Advantage discounting.")
parser.add_argument("--train_cycles", default=32, type=int, help="Number of PPO passes.")
parser.add_argument("--train_sims", default=512, type=int, help="How many simulations to train from.")
parser.add_argument("--evaluate", default=False, type=bool, help="If NashConv should be computed as well.")

# Queue parameters
parser.add_argument("--F", default=4, type=int, help="Amount to pay to leave queue.")
parser.add_argument("--Q", default=6, type=int, help="End fine to pay.")
parser.add_argument("--T", default=4, type=int, help="Time to survive in queue.")
parser.add_argument("--k", default=2, type=int, help="How many people have to pay in each step.")
parser.add_argument("--algorithms", default=['PPO', 'BRS'], type=list[str], help="List of algorithms to use, len(algorithms) = g, each unique entry is a separate instance. See `algorithm_initializer` for syntax.")
parser.add_argument("--g_prob", default=[0.1, 0.9], type=list[float], help="List of percentages representing probability each agents belongs to a given group.")
parser.add_argument("--tau", default=2, type=int, help="Don't use agents added to queue before tau * T steps.")
parser.add_argument("--x_mean", default=32, type=int, help="Mean number of agents to add each step.")
parser.add_argument("--x_std", default=0, type=float, help="Standard deviation of the number of agents to add each step.")
parser.add_argument("--N_init", default=32, type=int, help="Initial number of agents.")
parser.add_argument("--ignorance_distribution", default='fixed', type=str, help="What distribuin to use to sample probability of ignorance. Supported: fixed, uniform, beta.")
parser.add_argument("--p", default=0.5, type=float, help="Fixed probability of ignorance")
parser.add_argument("--p_min", default=0.0, type=float, help="Parameter of uniform distribution of ignorance.")
parser.add_argument("--alpha", default=2, type=float, help="Parameter of Beta distribution of ignorance.")
parser.add_argument("--beta", default=4, type=float, help="Parameter of Beta distribution of ignorance.")
parser.add_argument("--reward_shaping", default=False, type=bool, help="If rewards shaping should be used.")

args = parser.parse_args([] if "__file__" not in globals() else None)

args.g = len(args.g_prob)
if args.algorithms == ['BRS']:
	args.train_sims = 1
np.random.seed(args.seed)
torch.manual_seed(args.seed)


def algorithm_initializer(alg_name, args):
	if alg_name[:3] == 'PPO':
		return PPOLearner(args)
	elif alg_name[:3] == 'BRS':
		return BRSAgent(args)
	else:
		raise NotImplementedError(f'Unknown algorithm name {alg_name}.')


class GroupRunner:
	def __init__(self, args):
		self.sim_num = 0
		self.queue = Queue(args)
		self.args = args

		assert len(args.algorithms) == len(args.g_prob), "Sizes don't match."
		assert sum(args.g_prob) == 1, "Probabilities don't sum to one."
		self.groups = {}
		for i, alg in enumerate(args.algorithms):
			self.groups[i] = algorithm_initializer(alg, args)

	def prepare_results_dir(self):
		self.results_path = os.getcwd()
		os.chdir(f'../Results/Group')
		dir_name = f'_{"-".join(args.algorithms)}_{"-".join([str(pr) for pr in args.g_prob])}_e{str(args.evaluate)[0]}_F{int(args.F)}_Q{int(args.Q)}_T{int(args.T)}_k{int(args.k)}_x{int(args.x_mean)}_p{np.round(args.p, 4)}_b{args.buffer_len}_t{args.train_cycles}_s{int(args.seed)}'
		os.mkdir(dir_name)
		os.chdir(dir_name)
		pickle.dump(args, open("args.pickle", "wb"))

	def optimization_step(self):
		state = self.queue.initialize()

		while not np.all([learner.train_ready for learner in self.groups.values()]):
			ids = state[:, 3]
			actions = np.empty_like(ids)
			for g, learner in self.groups.items():
				group_ids = np.where(ids == g)
				if group_ids[0].shape[0] == 0:  # There is nobody in this group now
					continue

				group_states = state[group_ids, :3]
				actions[group_ids] = learner.actions(group_states[0], [self.queue.agents[i] for i in group_ids[0]])

			state, removed = self.queue.step(actions)

			for r in removed:
				self.groups[r.group].store(r)

		for learner in self.groups.values():
			learner.train()

	def run_optimization(self):
		self.prepare_results_dir()
		for _ in range(self.args.train_sims):
			print(f'Running simulation {self.sim_num+1}.')
			self.optimization_step()
			self.save()
			self.sim_num += 1

	def save(self):
		queue_saver(self.queue, self.sim_num)
		if args.evaluate or self.sim_num + 1 == args.train_sims:
			for g, learner in self.groups.items():
				learner.save(name=self.args.algorithms[g], iteration=self.sim_num)

	def nash_conv(self, folder, br_frac):
		if folder is not None:
			os.chdir(f'../Results/Group/{folder}')
			args = pickle.load(open("args.pickle", "rb"))
		else:
			args = self.args
		root = os.getcwd()
		if not args.evaluate:
			print('Cannot evaluate NashConv, because only the final policy was saved.')
			return

		# First get expected terminal utility in the original background
		on_policy_utility = np.empty((len(args.algorithms), args.train_sims - 1))
		for sim in range(args.train_sims - 1):
			print(f'Running on policy utility, iteration {sim+1}.')
			self.groups = {}
			for i, alg in enumerate(args.algorithms):
				os.chdir(alg+'/policies')
				group_policy = np.load(f'policy_{sim}.npy')
				self.groups[i] = TabularAgent(group_policy, utility_samples=args.buffer_len)
				os.chdir(root)

			self.optimization_step()
			for g, learner in self.groups.items():
				on_policy_utility[g, sim] = np.mean(learner.utilities)

		# Second, for each algorithm, make `br_frac`-percent of population its next policy iteration and get its utility
		best_response_utility = np.empty((len(args.algorithms), args.train_sims - 1))
		original_g_probs = args.g_prob.copy()
		for sim in range(args.train_sims - 1):
			print(f'Running best response policy utility, iteration {sim+1}.')
			for br_num, br_alg in enumerate(args.algorithms):
				self.groups = {}
				args.g_prob = np.array(original_g_probs) / (1 + br_frac * args.g_prob[br_num])  # Normalize the rest
				args.g_prob = np.append(args.g_prob, br_frac * args.g_prob[br_num])  # Add the br group
				for i, alg in enumerate(args.algorithms):
					os.chdir(alg+'/policies')
					group_policy = np.load(f'policy_{sim}.npy')
					self.groups[i] = TabularAgent(group_policy, utility_samples=args.buffer_len)
					os.chdir(root)

				os.chdir(br_alg+'/policies')
				group_policy = np.load(f'policy_{sim+1}.npy')
				self.groups[len(args.algorithms)] = TabularAgent(group_policy, utility_samples=args.buffer_len)
				os.chdir(root)

				self.queue = Queue(args)
				self.optimization_step()
				best_response_utility[br_num, sim] = np.mean(self.groups[len(args.algorithms)].utilities)

		nashconv = best_response_utility - on_policy_utility
		np.savetxt('nashconv.npy', nashconv)

	def clean_up(self):
		root = os.getcwd()
		os.chdir(f'{self.args.algorithms[0]}/policies')
		for i in range(args.train_sims - 1):
			os.remove(f'policy_{i}.npy')
		os.chdir(root)

		os.chdir('queue_data')
		for i in range(args.train_sims - 1):
			os.remove(f'info_{i}.pickle')
			os.remove(f'leaving_payment_{i}.npy')
			os.remove(f'leaving_time_{i}.npy')
		os.chdir(root)


group_runner = GroupRunner(args)

group_runner.run_optimization()
group_runner.nash_conv(folder=None, br_frac=0.1)





