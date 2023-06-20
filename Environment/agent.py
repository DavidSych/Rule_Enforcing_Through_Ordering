import numpy as np


class Agent:
	def __init__(self, args, group_num):
		self.t, self.payment, self.group = 0, 0, group_num
		self.strategy, self.T = 0, args.T
		if args.ignorance_distribution == 'fixed':
			self.p = args.p
		elif args.ignorance_distribution == 'uniform':
			self.p = np.random.uniform(args.p_min, 1)
		elif args.ignorance_distribution == 'beta':
			self.p = np.random.beta(args.alpha, args.beta)
		else:
			raise NotImplementedError(f'Unknown ignorance distribution {args.ignorance_distribution}.')

		# Storing agent's trajectory, (0) running payment, (1) my t, (2) my position, (3) group
		self.my_states = np.zeros((args.T, 3), dtype=np.int32)
		# If agent was allowed to act or not
		self.acting = np.zeros((args.T, ), dtype=np.int32)
		# Storing agent's rewards and costs
		self.my_rewards = np.zeros(args.T)
		# Desired action if allowed to act
		self.my_actions = np.zeros(args.T)

	@property
	def average_payment(self):
		return self.payment / self.t

	@property
	def in_danger(self):
		if self.t > 1:
			n_curr, n_prev = self.my_states[self.t, 2], self.my_states[self.t-1, 2]
			return (n_prev - n_curr) * (self.T - self.t) > n_curr
		else:
			return False

	def terminate(self):
		self.my_states = self.my_states[:self.t]
		self.my_rewards = self.my_rewards[:self.t]
		self.acting = self.acting[:self.t]


