import torch
from torch import nn
import torch.optim as optim
from torch.nn.functional import softplus
import os, pickle
from copy import deepcopy

CLIP_GRADS_NUM = 10**-2
LEARNING_RATE = 10**-3

class PolicyModel:

	def __init__(self, N_side_in, N_ops, N_params, N_hidden_in, **kwargs):

		self.N_side = N_side_in
		self.N_ops = N_ops
		self.N_params = N_params
		self.N_hidden = N_hidden_in

		self.init_kwargs = deepcopy(kwargs)
		if 'N_side' in self.init_kwargs.keys():
			self.init_kwargs.pop('N_side')
		if 'N_hidden' in self.init_kwargs.keys():
			self.init_kwargs.pop('N_hidden')

		### Create separate policies
		self.reset_all_policies()

		### Create optimizers
		self.reset_all_optimizers(**kwargs)

		self.print_NN_info(**kwargs)



	def reset_switches(self, **kwargs):

		if kwargs.get('reset_all_policies', False):
			self.reset_all_policies()

		if kwargs.get('reset_policy_op', False):
			self.reset_policy_op()

		if kwargs.get('reset_policy_params', False):
			self.reset_policy_params()

		if kwargs.get('reset_policy_canv_1', False):
			self.reset_policy_canv_1()

		if kwargs.get('reset_policy_canv_2', False):
			self.reset_policy_canv_2()


		if kwargs.get('reset_all_optimizers', False):
			self.reset_all_optimizers(**kwargs)

		if kwargs.get('reset_optim_op', False):
			self.reset_optim_op(**kwargs)

		if kwargs.get('reset_optim_params', False):
			self.reset_optim_params(**kwargs)

		if kwargs.get('reset_optim_canv_1', False):
			self.reset_optim_canv_1(**kwargs)

		if kwargs.get('reset_optim_canv_2', False):
			self.reset_optim_canv_2(**kwargs)

	def reset_all_policies(self):
		self.reset_policy_op()
		self.reset_policy_params()
		self.reset_policy_canv_1()
		self.reset_policy_canv_2()

	def reset_policy_op(self):
		self.policy_op = Policy_NN_op(self.N_side, self.N_ops, self.N_hidden, **self.init_kwargs)
		self.reset_optim_op()

	def reset_policy_params(self):
		self.policy_params = Policy_NN_params(self.N_side, self.N_ops, self.N_params, self.N_hidden, **self.init_kwargs)
		self.reset_optim_params()

	def reset_policy_canv_1(self):
		self.policy_canv_1 = Policy_NN_canvas_1(self.N_side, self.N_ops, self.N_hidden, **self.init_kwargs)
		self.reset_optim_canv_1()

	def reset_policy_canv_2(self):
		self.policy_canv_2 = Policy_NN_canvas_2(self.N_side, self.N_ops, self.N_hidden, **self.init_kwargs)
		self.reset_optim_canv_2()


	def reset_all_optimizers(self, **kwargs):
		self.reset_optim_op(**kwargs)
		self.reset_optim_params(**kwargs)
		self.reset_optim_canv_1(**kwargs)
		self.reset_optim_canv_2(**kwargs)

	def reset_optim_op(self, **kwargs):
		self.optimizer_op = optim.Adam(list(self.policy_op.parameters()), lr=kwargs.get('lr', LEARNING_RATE))

	def reset_optim_params(self, **kwargs):
		self.optimizer_params = optim.Adam(list(self.policy_params.parameters()), lr=kwargs.get('lr', LEARNING_RATE))

	def reset_optim_canv_1(self, **kwargs):
		self.optimizer_canv_1 = optim.Adam(list(self.policy_canv_1.parameters()), lr=kwargs.get('lr', LEARNING_RATE))

	def reset_optim_canv_2(self, **kwargs):
		self.optimizer_canv_2 = optim.Adam(list(self.policy_canv_2.parameters()), lr=kwargs.get('lr', LEARNING_RATE))





	def zero_all_grads(self, **kwargs):

		self.optimizer_op.zero_grad()
		self.optimizer_params.zero_grad()
		self.optimizer_canv_1.zero_grad()
		self.optimizer_canv_2.zero_grad()


	def step_all_optims(self, **kwargs):

		self.optimizer_op.step()
		self.optimizer_params.step()
		self.optimizer_canv_1.step()
		self.optimizer_canv_2.step()


	def get_grads_info(self):
		#nn.utils.clip_grad_value_(policy_model.policy_params.parameters(), 1.0)
		grads_info = {}
		for p in ['policy_op', 'policy_params', 'policy_canv_1', 'policy_canv_2']:
			grads_info[p + '_max_g'] = -1.0
			grads_info[p + '_mean_g'] = -1.0

		for p in self.policy_op.parameters():
			if p.grad is not None:
				max_g = torch.max(torch.abs(p.grad.data))
				mean_g = torch.mean(torch.abs(p.grad.data))
				if max_g > grads_info['policy_op_max_g']:
					grads_info['policy_op_max_g'] = max_g
				if mean_g > grads_info['policy_op_mean_g']:
					grads_info['policy_op_mean_g'] = mean_g

		for p in self.policy_params.parameters():
			if p.grad is not None:
				max_g = torch.max(torch.abs(p.grad.data))
				mean_g = torch.mean(torch.abs(p.grad.data))
				if max_g > grads_info['policy_params_max_g']:
					grads_info['policy_params_max_g'] = max_g
				if mean_g > grads_info['policy_params_mean_g']:
					grads_info['policy_params_mean_g'] = mean_g

		for p in self.policy_canv_1.parameters():
			if p.grad is not None:
				max_g = torch.max(torch.abs(p.grad.data))
				mean_g = torch.mean(torch.abs(p.grad.data))
				if max_g > grads_info['policy_canv_1_max_g']:
					grads_info['policy_canv_1_max_g'] = max_g
				if mean_g > grads_info['policy_canv_1_mean_g']:
					grads_info['policy_canv_1_mean_g'] = mean_g

		for p in self.policy_canv_2.parameters():
			if p.grad is not None:
				max_g = torch.max(torch.abs(p.grad.data))
				mean_g = torch.mean(torch.abs(p.grad.data))
				if max_g > grads_info['policy_canv_2_max_g']:
					grads_info['policy_canv_2_max_g'] = max_g
				if mean_g > grads_info['policy_canv_2_mean_g']:
					grads_info['policy_canv_2_mean_g'] = mean_g

		for k,v in grads_info.items():
			if isinstance(v, torch.Tensor):
				grads_info[k] = v.item()

		return grads_info


	def get_weights_info(self):
		#nn.utils.clip_grad_value_(policy_model.policy_params.parameters(), 1.0)
		weights_info = {}
		for p in ['policy_op', 'policy_params', 'policy_canv_1', 'policy_canv_2']:
			weights_info[p + '_max_w'] = -1.0
			weights_info[p + '_mean_w'] = -1.0

		for p in self.policy_op.parameters():
			max_w = torch.max(torch.abs(p.data))
			mean_w = torch.mean(torch.abs(p.data))
			if max_w > weights_info['policy_op_max_w']:
				weights_info['policy_op_max_w'] = max_w
			if mean_w > weights_info['policy_op_mean_w']:
				weights_info['policy_op_mean_w'] = mean_w

		for p in self.policy_params.parameters():
			max_w = torch.max(torch.abs(p.data))
			mean_w = torch.mean(torch.abs(p.data))
			if max_w > weights_info['policy_params_max_w']:
				weights_info['policy_params_max_w'] = max_w
			if mean_w > weights_info['policy_params_mean_w']:
				weights_info['policy_params_mean_w'] = mean_w

		for p in self.policy_canv_1.parameters():
			max_w = torch.max(torch.abs(p.data))
			mean_w = torch.mean(torch.abs(p.data))
			if max_w > weights_info['policy_canv_1_max_w']:
				weights_info['policy_canv_1_max_w'] = max_w
			if mean_w > weights_info['policy_canv_1_mean_w']:
				weights_info['policy_canv_1_mean_w'] = mean_w

		for p in self.policy_canv_2.parameters():
			max_w = torch.max(torch.abs(p.data))
			mean_w = torch.mean(torch.abs(p.data))
			if max_w > weights_info['policy_canv_2_max_w']:
				weights_info['policy_canv_2_max_w'] = max_w
			if mean_w > weights_info['policy_canv_2_mean_w']:
				weights_info['policy_canv_2_mean_w'] = mean_w

		for k,v in weights_info.items():
			weights_info[k] = v.item()

		return weights_info


	def clip_all_grads(self):

		nn.utils.clip_grad_value_(self.policy_op.parameters(), CLIP_GRADS_NUM)
		nn.utils.clip_grad_value_(self.policy_params.parameters(), CLIP_GRADS_NUM)
		nn.utils.clip_grad_value_(self.policy_canv_1.parameters(), CLIP_GRADS_NUM)
		nn.utils.clip_grad_value_(self.policy_canv_2.parameters(), CLIP_GRADS_NUM)


	def print_NN_info(self, **kwargs):

		pol_names = ['policy_op', 'policy_params', 'policy_canv_1', 'policy_canv_2']
		pols = [self.policy_op, self.policy_params, self.policy_canv_1, self.policy_canv_2]

		print('\n')
		tot = 0
		for name, pol in zip(pol_names, pols):
			params_list = list(pol.parameters())
			param_sizes = [len(p.detach().flatten()) for p in params_list]
			num = sum(param_sizes)
			tot += num
			print('N_params in policy {} = {}'.format(name, num))
			if kwargs.get('print_layer_details', False):
				for p,ps in zip(params_list, param_sizes):
					print('\tshape = {} ===> N_params = {}'.format(list(p.shape), ps))

		print('Total params = {}'.format(tot))
		if kwargs.get('print_layer_details', False):
			exit()


class Policy_NN_op(nn.Module):

	def __init__(self, N_side, N_ops, N_hidden, **kwargs):
		super(Policy_NN_op, self).__init__()

		self.N_side = N_side
		self.N_flat_canvas = self.N_side**2
		self.N_ops = N_ops
		self.N_hidden = N_hidden


		self.conv_layer_input = kwargs.get('conv_layer_input', False)

		if self.conv_layer_input:

			self.use_pooling = kwargs.get('use_pooling', False)
			self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
			self.conv_kernel_size = kwargs.get('conv_kernel_size', 3)
			self.N_conv1_out_channels = kwargs.get('N_conv1_out_channels', 16)
			self.N_conv2_out_channels = kwargs.get('N_conv2_out_channels', 32)
			self.N_conv3_out_channels = kwargs.get('N_conv3_out_channels', 1)

			self.conv1 = nn.Conv2d(1, self.N_conv1_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)
			self.conv2 = nn.Conv2d(self.N_conv1_out_channels, self.N_conv2_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)
			self.conv3 = nn.Conv2d(self.N_conv2_out_channels, self.N_conv3_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)

			self.conv1_V = nn.Conv2d(1, self.N_conv1_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)
			self.conv2_V = nn.Conv2d(self.N_conv1_out_channels, self.N_conv2_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)
			self.conv3_V = nn.Conv2d(self.N_conv2_out_channels, self.N_conv3_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)





			if self.use_pooling:
				self.conv_out_side_2 = self.N_side
				self.N_pool_out_side_2 = int(self.conv_out_side_2/2)
				self.conv_out_flat_size = self.N_conv3_out_channels*self.N_pool_out_side_2**2

			else:
				self.conv_out_flat_size = self.N_conv3_out_channels*(self.N_side)**2

			self.layer_1 = nn.Linear(self.conv_out_flat_size, self.N_hidden)
			self.layer_V = nn.Linear(self.conv_out_flat_size, self.N_hidden)
			self.ops_out = nn.Linear(self.N_hidden, self.N_ops)
			self.V_op = nn.Linear(self.N_hidden, 1)

		else:

			self.N_inputs = self.N_flat_canvas


			self.NN_type = kwargs.get('NN_type', 'one_layer')
			if self.NN_type == 'linear':
				self.ops_out = nn.Linear(self.N_inputs, self.N_ops)
				self.V_op = nn.Linear(self.N_inputs, 1)
			else:
				self.layer_1 = nn.Linear(self.N_inputs, self.N_hidden)
				self.layer_V = nn.Linear(self.N_inputs, self.N_hidden)
				if self.NN_type == 'two_layer':
					self.layer_2 = nn.Linear(self.N_hidden, self.N_hidden)

				self.ops_out = nn.Linear(self.N_hidden, self.N_ops)
				self.V_op = nn.Linear(self.N_hidden, 1)



		if kwargs.get('nonlinear', 'relu') == 'tanh':
			self.nonlinear = nn.Tanh()
		else:
			self.nonlinear = nn.ReLU()

		self.sigmoid = nn.Sigmoid()
		self.op_prob_min = 0.01




	def forward(self, canv, **kwargs):

		if self.conv_layer_input:
			canv = canv.view(1, 1, self.N_side, self.N_side)

			conv_1_out = self.nonlinear(self.conv1(canv))
			conv_2_out = self.nonlinear(self.conv2(conv_1_out))
			if self.use_pooling:
				conv_2_out = self.pool(conv_2_out)
			conv_3_out = self.nonlinear(self.conv3(conv_2_out))
			conv_flat = conv_3_out.reshape(-1, self.conv_out_flat_size)


			conv_1_out_V = self.nonlinear(self.conv1_V(canv))
			conv_2_out_V = self.nonlinear(self.conv2_V(conv_1_out_V))
			if self.use_pooling:
				conv_2_out_V = self.pool(conv_2_out_V)
			conv_3_out_V = self.nonlinear(self.conv3_V(conv_2_out_V))
			conv_flat_V = conv_3_out_V.reshape(-1, self.conv_out_flat_size)


			x = self.nonlinear(self.layer_1(conv_flat))
			y = self.nonlinear(self.layer_V(conv_flat_V))


		else:

			canv_flat = canv.reshape(-1, self.N_flat_canvas)

			input = canv_flat


			if self.NN_type == 'linear':
				x = input
				y = input
			else:
				x = self.nonlinear(self.layer_1(input))
				y = self.nonlinear(self.layer_V(input))
				if self.NN_type == 'two_layer':
					x = self.nonlinear(self.layer_2(x))




		ops_out = torch.softmax(self.ops_out(x), dim=1) + self.op_prob_min
		V_op = self.V_op(y)

		return {
			'ops_out' : ops_out,
			'V_op' : V_op
		}



class Policy_NN_params(nn.Module):

	def __init__(self, N_side, N_ops, N_params, N_hidden, **kwargs):
		super(Policy_NN_params, self).__init__()

		self.N_side = N_side
		self.N_flat_canvas = self.N_side**2
		self.N_ops = N_ops
		self.N_params = N_params
		self.N_hidden = N_hidden

		self.conv_layer_input = kwargs.get('conv_layer_input', False)

		if self.conv_layer_input:

			self.use_pooling = kwargs.get('use_pooling', True)
			self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
			self.conv_kernel_size = kwargs.get('conv_kernel_size', 3)
			self.N_conv1_out_channels = kwargs.get('N_conv1_out_channels', 16)
			self.N_conv2_out_channels = kwargs.get('N_conv2_out_channels', 32)
			self.N_conv3_out_channels = kwargs.get('N_conv3_out_channels', 1)

			self.conv1 = nn.Conv2d(1, self.N_conv1_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)
			self.conv2 = nn.Conv2d(self.N_conv1_out_channels, self.N_conv2_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)
			self.conv3 = nn.Conv2d(self.N_conv2_out_channels, self.N_conv3_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)

			self.conv1_V = nn.Conv2d(1, self.N_conv1_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)
			self.conv2_V = nn.Conv2d(self.N_conv1_out_channels, self.N_conv2_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)
			self.conv3_V = nn.Conv2d(self.N_conv2_out_channels, self.N_conv3_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)

			if self.use_pooling:
				self.conv_out_side_2 = self.N_side
				self.N_pool_out_side_2 = int(self.conv_out_side_2/2)
				self.conv_out_flat_size = self.N_conv3_out_channels*self.N_pool_out_side_2**2

			else:
				self.conv_out_flat_size = self.N_conv3_out_channels*(self.N_side)**2

			self.layer_1 = nn.Linear(self.conv_out_flat_size, self.N_hidden)
			self.layer_V = nn.Linear(self.conv_out_flat_size, self.N_hidden)
			self.params_out_mu = nn.Linear(self.N_hidden, self.N_params)
			self.params_out_sigma = nn.Linear(self.N_hidden, self.N_params)
			self.V_params = nn.Linear(self.N_hidden, 1)


		else:


			self.N_inputs = self.N_flat_canvas

			self.NN_type = kwargs.get('NN_type', 'one_layer')
			if self.NN_type == 'linear':
				self.params_out_mu = nn.Linear(self.N_inputs, self.N_params)
				self.params_out_sigma = nn.Linear(self.N_inputs, self.N_params)
				self.V_params = nn.Linear(self.N_inputs, 1)
			else:
				self.layer_1 = nn.Linear(self.N_inputs, self.N_hidden)
				self.layer_V = nn.Linear(self.N_inputs, self.N_hidden)
				if self.NN_type == 'two_layer':
					self.layer_2 = nn.Linear(self.N_hidden, self.N_hidden)

				self.params_out_mu = nn.Linear(self.N_hidden, self.N_params)
				self.params_out_sigma = nn.Linear(self.N_hidden, self.N_params)
				self.V_params = nn.Linear(self.N_hidden, 1)



		if kwargs.get('nonlinear', 'relu') == 'tanh':
			self.nonlinear = nn.Tanh()
		else:
			self.nonlinear = nn.ReLU()

		self.sigmoid = nn.Sigmoid()
		self.sigma_min = kwargs.get('sigma_min', 10**-3)
		self.beta_mu_epsilon = 0.01

		self.params_sigma_min = 0.01
		self.params_sigma_max = 0.5



	def forward(self, canv):

		if self.conv_layer_input:
			canv = canv.view(1, 1, self.N_side, self.N_side)

			conv_1_out = self.nonlinear(self.conv1(canv))
			conv_2_out = self.nonlinear(self.conv2(conv_1_out))
			if self.use_pooling:
				conv_2_out = self.pool(conv_2_out)
			conv_3_out = self.nonlinear(self.conv3(conv_2_out))
			conv_flat = conv_3_out.reshape(-1, self.conv_out_flat_size)


			conv_1_out_V = self.nonlinear(self.conv1_V(canv))
			conv_2_out_V = self.nonlinear(self.conv2_V(conv_1_out_V))
			if self.use_pooling:
				conv_2_out_V = self.pool(conv_2_out_V)
			conv_3_out_V = self.nonlinear(self.conv3_V(conv_2_out_V))
			conv_flat_V = conv_3_out_V.reshape(-1, self.conv_out_flat_size)


			x = self.nonlinear(self.layer_1(conv_flat))
			y = self.nonlinear(self.layer_V(conv_flat_V))

		else:


			canv_flat = canv.reshape(-1, self.N_flat_canvas)

			if self.NN_type == 'linear':
				x = canv_flat
				y = canv_flat
			else:
				x = self.nonlinear(self.layer_1(canv_flat))
				y = self.nonlinear(self.layer_V(canv_flat))
				if self.NN_type == 'two_layer':
					x = self.nonlinear(self.layer_2(x))


		params_mu = self.beta_mu_epsilon + (1 - 2*self.beta_mu_epsilon)*self.sigmoid(self.params_out_mu(x))
		#params_mu = self.sigmoid(self.params_out_mu(x))
		#params_sigma = softplus(self.params_out_sigma(x)) + self.sigma_min

		#params_sigma = self.params_sigma_min + self.sigmoid(self.params_out_sigma(x) - 5.0)*(self.params_sigma_max - self.params_sigma_min)
		params_sigma = self.sigmoid(self.params_out_sigma(x) - 5.0)
		#params_sigma = self.sigmoid(self.params_out_sigma(x))

		V_params = self.V_params(y)

		return {
			'params_mu' : params_mu,
			'params_sigma' : params_sigma,
			'V_params' : V_params
		}



class Policy_NN_canvas_1(nn.Module):

	def __init__(self, N_side, N_ops, N_hidden, **kwargs):
		super(Policy_NN_canvas_1, self).__init__()

		self.N_side = N_side
		self.N_flat_canvas = self.N_side**2
		self.N_ops = N_ops

		self.N_hidden = N_hidden


		self.conv_layer_input = kwargs.get('conv_layer_input', False)
		self.deconv_layer_output = kwargs.get('deconv_layer_output', False)

		if self.conv_layer_input:

			self.use_pooling = kwargs.get('use_pooling', False)

			self.pooling_12 = kwargs.get('pooling_12', False)
			self.pooling_23 = kwargs.get('pooling_23', False)

			self.conv_kernel_size = kwargs.get('conv_kernel_size', 3)
			self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

			self.N_conv1_out_channels = kwargs.get('N_conv1_out_channels', 16)
			self.N_conv2_out_channels = kwargs.get('N_conv2_out_channels', 32)
			self.N_conv3_out_channels = kwargs.get('N_conv3_out_channels', 1)

			self.conv1 = nn.Conv2d(1, self.N_conv1_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)
			self.conv2 = nn.Conv2d(self.N_conv1_out_channels, self.N_conv2_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)
			self.conv3 = nn.Conv2d(self.N_conv2_out_channels, self.N_conv3_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)

			self.conv1_V = nn.Conv2d(1, self.N_conv1_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)
			self.conv2_V = nn.Conv2d(self.N_conv1_out_channels, self.N_conv2_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)
			self.conv3_V = nn.Conv2d(self.N_conv2_out_channels, self.N_conv3_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)

			if self.use_pooling:
				self.N_side_conv_1 = self.N_side

				if self.pooling_12:
					self.N_side_pool_1 = int(self.N_side_conv_1/2)
				else:
					self.N_side_pool_1 = self.N_side_conv_1

				self.N_side_conv_2 = self.N_side_pool_1

				if self.pooling_23:
					self.N_side_pool_2 = int(self.N_side_conv_2/2)
				else:
					self.N_side_pool_2 = self.N_side_conv_2

				self.N_side_conv_3 = self.N_side_pool_2


				self.conv_out_flat_size = self.N_conv3_out_channels*self.N_side_conv_3**2

			else:
				self.conv_out_flat_size = self.N_conv3_out_channels*(self.N_side)**2

			self.linear_1_input_size = self.conv_out_flat_size + self.N_ops

			if self.deconv_layer_output:
				self.N_channels_deconv_1 = 200
				self.N_channels_deconv_2 = 50

				self.N_side_linear_out = 1
				N_linear_out = self.N_channels_deconv_1*self.N_side_linear_out**2
				self.layer_1 = nn.Linear(self.linear_1_input_size, N_linear_out)

				kernel_size_1 = 3
				deconv_1_side_out = self.N_side_linear_out + kernel_size_1 - 1
				self.deconv_1_mu = nn.ConvTranspose2d(self.N_channels_deconv_1, self.N_channels_deconv_2, kernel_size_1)
				self.deconv_1_sigma = nn.ConvTranspose2d(self.N_channels_deconv_1, self.N_channels_deconv_2, kernel_size_1)

				kernel_size_2 = self.N_side - deconv_1_side_out + 1
				print('Deconv kernel sizes: {}, {}'.format(kernel_size_1, kernel_size_2))
				self.deconv_2_mu = nn.ConvTranspose2d(self.N_channels_deconv_2, 1, kernel_size_2)
				self.deconv_2_sigma = nn.ConvTranspose2d(self.N_channels_deconv_2, 1, kernel_size_2)

				self.V_canv_1 = nn.Linear(N_linear_out, 1)


			else:
				self.layer_1 = nn.Linear(self.linear_1_input_size, self.N_hidden)
				self.layer_V = nn.Linear(self.linear_1_input_size, self.N_hidden)
				self.canv_out_1_mu = nn.Linear(self.N_hidden, self.N_flat_canvas)
				self.canv_out_1_sigma = nn.Linear(self.N_hidden, self.N_flat_canvas)
				self.V_canv_1 = nn.Linear(self.N_hidden, 1)



		else:
			self.N_inputs = self.N_flat_canvas + self.N_ops

			self.NN_type = kwargs.get('NN_type', 'one_layer')
			if self.NN_type == 'linear':
				self.canv_out_1_mu = nn.Linear(self.N_inputs, self.N_flat_canvas)
				self.canv_out_1_sigma = nn.Linear(self.N_inputs, self.N_flat_canvas)
				self.V_canv_1 = nn.Linear(self.N_inputs, 1)
			else:
				self.layer_1 = nn.Linear(self.N_inputs, self.N_hidden)
				self.layer_V = nn.Linear(self.N_inputs, self.N_hidden)
				if self.NN_type == 'two_layer':
					self.layer_2 = nn.Linear(self.N_hidden, self.N_hidden)

				self.canv_out_1_mu = nn.Linear(self.N_hidden, self.N_flat_canvas)
				self.canv_out_1_sigma = nn.Linear(self.N_hidden, self.N_flat_canvas)
				self.V_canv_1 = nn.Linear(self.N_hidden, 1)


		if kwargs.get('nonlinear', 'relu') == 'tanh':
			self.nonlinear = nn.Tanh()
		else:
			self.nonlinear = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.beta_mu_epsilon = 0.01



	def forward(self, canv, op_OHE):

		op_OHE = op_OHE.unsqueeze(dim=0)


		if self.conv_layer_input:
			canv = canv.view(1, 1, self.N_side, self.N_side)

			conv_1_out = self.nonlinear(self.conv1(canv))
			if self.pooling_12:
				conv_1_out = self.pool(conv_1_out)

			conv_2_out = self.nonlinear(self.conv2(conv_1_out))
			if self.pooling_23:
				conv_2_out = self.pool(conv_2_out)

			conv_3_out = self.nonlinear(self.conv3(conv_2_out))
			conv_flat = conv_3_out.reshape(-1, self.conv_out_flat_size)


			conv_1_out_V = self.nonlinear(self.conv1_V(canv))
			if self.pooling_12:
				conv_1_out_V = self.pool(conv_1_out_V)

			conv_2_out_V = self.nonlinear(self.conv2_V(conv_1_out_V))
			if self.pooling_23:
				conv_2_out_V = self.pool(conv_2_out_V)

			conv_3_out_V = self.nonlinear(self.conv3_V(conv_2_out_V))
			conv_flat_V = conv_3_out_V.reshape(-1, self.conv_out_flat_size)


			conv_and_op = torch.cat((conv_flat, op_OHE), dim=1)
			conv_and_op_V = torch.cat((conv_flat_V, op_OHE), dim=1)

			x = self.nonlinear(self.layer_1(conv_and_op))
			y = self.nonlinear(self.layer_V(conv_and_op_V))


			if self.deconv_layer_output:

				y_mu = self.nonlinear(self.deconv_1_mu(x.reshape(-1, self.N_channels_deconv_1, self.N_side_linear_out, self.N_side_linear_out)))
				y_sigma = self.nonlinear(self.deconv_1_sigma(x.reshape(-1, self.N_channels_deconv_1, self.N_side_linear_out, self.N_side_linear_out)))

				canv_1_mu = self.beta_mu_epsilon + (1 - 2*self.beta_mu_epsilon)*self.sigmoid(self.deconv_2_mu(y_mu)).squeeze()
				canv_1_sigma = self.sigmoid(self.deconv_2_sigma(y_sigma) - 5.0).squeeze()

				V_canv_1 = self.V_canv_1(x)

			else:

				canv_1_mu = self.beta_mu_epsilon + (1 - 2*self.beta_mu_epsilon)*self.sigmoid(self.canv_out_1_mu(x)).reshape(self.N_side, self.N_side)
				canv_1_sigma = self.sigmoid(self.canv_out_1_sigma(x) - 5.0).reshape(self.N_side, self.N_side)

				V_canv_1 = self.V_canv_1(y)




		else:
			canv_flat = canv.reshape(-1, self.N_flat_canvas)
			input = torch.cat((canv_flat, op_OHE), dim=1)

			if self.NN_type == 'linear':
				x = input
				y = input
			else:
				x = self.nonlinear(self.layer_1(input))
				y = self.nonlinear(self.layer_V(input))
				if self.NN_type == 'two_layer':
					x = self.nonlinear(self.layer_2(x))


			canv_1_mu = self.beta_mu_epsilon + (1 - 2*self.beta_mu_epsilon)*self.sigmoid(self.canv_out_1_mu(x)).reshape(self.N_side, self.N_side)
			canv_1_sigma = self.sigmoid(self.canv_out_1_sigma(x) - 5.0).reshape(self.N_side, self.N_side)

			V_canv_1 = self.V_canv_1(y)

		return {
			'canv_1_mu' : canv_1_mu,
			'V_canv_1' : V_canv_1,
			'canv_1_sigma' : canv_1_sigma,
		}



class Policy_NN_canvas_2(nn.Module):

	def __init__(self, N_side, N_ops, N_hidden, **kwargs):
		super(Policy_NN_canvas_2, self).__init__()

		self.N_side = N_side
		self.N_flat_canvas = self.N_side**2
		self.N_ops = N_ops

		self.N_hidden = N_hidden


		self.conv_layer_input = kwargs.get('conv_layer_input', False)

		if self.conv_layer_input:

			self.use_pooling = kwargs.get('use_pooling', False)
			self.conv_kernel_size = kwargs.get('conv_kernel_size', 3)
			self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

			self.N_conv1_out_channels = kwargs.get('N_conv1_out_channels', 8)
			self.N_conv2_out_channels = kwargs.get('N_conv2_out_channels', 16)
			self.N_conv3_out_channels = kwargs.get('N_conv3_out_channels', 32)

			self.conv1_target = nn.Conv2d(1, self.N_conv1_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)
			self.conv2_target = nn.Conv2d(self.N_conv1_out_channels, self.N_conv2_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)
			self.conv3_target = nn.Conv2d(self.N_conv2_out_channels, self.N_conv3_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)

			self.conv1_arg = nn.Conv2d(1, self.N_conv1_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)
			self.conv2_arg = nn.Conv2d(self.N_conv1_out_channels, self.N_conv2_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)
			self.conv3_arg = nn.Conv2d(self.N_conv2_out_channels, self.N_conv3_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)

			self.conv1_target_V = nn.Conv2d(1, self.N_conv1_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)
			self.conv2_target_V = nn.Conv2d(self.N_conv1_out_channels, self.N_conv2_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)
			self.conv3_target_V = nn.Conv2d(self.N_conv2_out_channels, self.N_conv3_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)

			self.conv1_arg_V = nn.Conv2d(1, self.N_conv1_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)
			self.conv2_arg_V = nn.Conv2d(self.N_conv1_out_channels, self.N_conv2_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)
			self.conv3_arg_V = nn.Conv2d(self.N_conv2_out_channels, self.N_conv3_out_channels, self.conv_kernel_size, padding=(self.conv_kernel_size - 1)//2)

			if self.use_pooling:
				self.conv_out_side_2 = self.N_side
				self.N_pool_out_side_2 = int(self.conv_out_side_2/2)
				self.conv_out_flat_size = self.N_conv3_out_channels*self.N_pool_out_side_2**2

			else:
				self.conv_out_flat_size = self.N_conv3_out_channels*(self.N_side)**2

			self.linear_1_input_size = 2*self.conv_out_flat_size + self.N_ops
			self.layer_1 = nn.Linear(self.linear_1_input_size, self.N_hidden)
			self.layer_V = nn.Linear(self.linear_1_input_size, self.N_hidden)
			self.canv_out_2_mu = nn.Linear(self.N_hidden, self.N_flat_canvas)
			self.canv_out_2_sigma = nn.Linear(self.N_hidden, self.N_flat_canvas)
			self.V_canv_2 = nn.Linear(self.N_hidden, 1)


		else:
			self.N_inputs = 2*self.N_flat_canvas + self.N_ops

			self.NN_type = kwargs.get('NN_type', 'one_layer')
			if self.NN_type == 'linear':
				self.canv_out_2_mu = nn.Linear(self.N_inputs, self.N_flat_canvas)
				self.canv_out_2_sigma = nn.Linear(self.N_inputs, self.N_flat_canvas)
				self.V_canv_2 = nn.Linear(self.N_inputs, 1)
			else:
				self.layer_1 = nn.Linear(self.N_inputs, self.N_hidden)
				self.layer_V = nn.Linear(self.N_inputs, self.N_hidden)
				if self.NN_type == 'two_layer':
					self.layer_2 = nn.Linear(self.N_hidden, self.N_hidden)

				self.canv_out_2_mu = nn.Linear(self.N_hidden, self.N_flat_canvas)
				self.canv_out_2_sigma = nn.Linear(self.N_hidden, self.N_flat_canvas)
				self.V_canv_2 = nn.Linear(self.N_hidden, 1)


		if kwargs.get('nonlinear', 'relu') == 'tanh':
			self.nonlinear = nn.Tanh()
		else:
			self.nonlinear = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.beta_mu_epsilon = 0.01



	def forward(self, canv_target, canv_arg, op_OHE):

		op_OHE = op_OHE.unsqueeze(dim=0)

		if self.conv_layer_input:

			canv_target = canv_target.view(1, 1, self.N_side, self.N_side)
			canv_arg = canv_arg.view(1, 1, self.N_side, self.N_side)

			###########

			conv_1_target_out = self.nonlinear(self.conv1_target(canv_target))
			conv_2_target_out = self.nonlinear(self.conv2_target(conv_1_target_out))
			if self.use_pooling:
				conv_2_target_out = self.pool(conv_2_target_out)
			conv_3_target_out = self.nonlinear(self.conv3_target(conv_2_target_out))
			conv_target_flat = conv_3_target_out.reshape(-1, self.conv_out_flat_size)

			conv_1_arg_out = self.nonlinear(self.conv1_arg(canv_arg))
			conv_2_arg_out = self.nonlinear(self.conv2_arg(conv_1_arg_out))
			if self.use_pooling:
				conv_2_arg_out = self.pool(conv_2_arg_out)
			conv_3_arg_out = self.nonlinear(self.conv3_arg(conv_2_arg_out))
			conv_arg_flat = conv_3_arg_out.reshape(-1, self.conv_out_flat_size)

			########

			conv_1_target_V_out = self.nonlinear(self.conv1_target_V(canv_target))
			conv_2_target_V_out = self.nonlinear(self.conv2_target_V(conv_1_target_V_out))
			if self.use_pooling:
				conv_2_target_V_out = self.pool(conv_2_target_V_out)
			conv_3_target_V_out = self.nonlinear(self.conv3_target_V(conv_2_target_V_out))
			conv_target_V_flat = conv_3_target_V_out.reshape(-1, self.conv_out_flat_size)

			conv_1_arg_V_out = self.nonlinear(self.conv1_arg_V(canv_arg))
			conv_2_arg_V_out = self.nonlinear(self.conv2_arg_V(conv_1_arg_V_out))
			if self.use_pooling:
				conv_2_arg_V_out = self.pool(conv_2_arg_V_out)
			conv_3_arg_V_out = self.nonlinear(self.conv3_arg_V(conv_2_arg_V_out))
			conv_arg_V_flat = conv_3_arg_V_out.reshape(-1, self.conv_out_flat_size)

			conv_and_op = torch.cat((conv_target_flat, conv_arg_flat, op_OHE), dim=1)
			conv_and_op_V = torch.cat((conv_target_V_flat, conv_arg_V_flat, op_OHE), dim=1)

			x = self.nonlinear(self.layer_1(conv_and_op))
			y = self.nonlinear(self.layer_V(conv_and_op_V))

		else:
			canv_target_flat = canv_target.reshape(-1, self.N_flat_canvas)
			canv_arg_flat = canv_arg.reshape(-1, self.N_flat_canvas)
			input = torch.cat((canv_target_flat, canv_arg_flat, op_OHE), dim=1)

			if self.NN_type == 'linear':
				x = input
			else:
				x = self.nonlinear(self.layer_1(input))
				y = self.nonlinear(self.layer_V(input))
				if self.NN_type == 'two_layer':
					x = self.nonlinear(self.layer_2(x))


		canv_2_mu = self.beta_mu_epsilon + (1 - 2*self.beta_mu_epsilon)*self.sigmoid(self.canv_out_2_mu(x)).reshape(self.N_side, self.N_side)
		canv_2_sigma = self.sigmoid(self.canv_out_2_sigma(x) - 5.0).reshape(self.N_side, self.N_side)

		V_canv_2 = self.V_canv_2(y)

		return {
			'canv_2_mu' : canv_2_mu,
			'V_canv_2' : V_canv_2,
			'canv_2_sigma' : canv_2_sigma
		}




############### Saving/loading models

def save_policy_model(pm):
	save_models_optim(pm.policy_op, pm.policy_canv_1, pm.policy_canv_2, pm.policy_params, pm.optimizer)


def load_policy_model(pm, **kwargs):
	load_models_optim(pm.policy_op, pm.policy_canv_1, pm.policy_canv_2, pm.policy_params, pm.optimizer, **kwargs)


def save_models_optim(policy_op, policy_canv_1, policy_canv_2, policy_params, optimizer):

	save_dict = {
		'policy_op' : policy_op.state_dict(),
		'policy_canv_1' : policy_canv_1.state_dict(),
		'policy_canv_2' : policy_canv_2.state_dict(),
		'policy_params' : policy_params.state_dict(),
		'optimizer' : optimizer.state_dict(),
	}

	models_dir = 'saved_models'
	model_path = os.path.join(models_dir, 'policy_and_optim.model')

	if os.path.exists(model_path):
		print(f'\nFile {model_path} already exists! Removing and saving to it now...\n')
		os.remove(model_path)

	torch.save(save_dict, model_path)
	print('Done saving!\n')


def load_models_optim(policy_op, policy_canv_1, policy_canv_2, policy_params, optimizer, **kwargs):


	policy_dict = {
		'policy_op' : policy_op,
		'policy_canv_1' : policy_canv_1,
		'policy_canv_2' : policy_canv_2,
		'policy_params' : policy_params,
	}

	if kwargs.get('load_optim', True):
		policy_dict['optimizer'] = optimizer

	models_dir = 'saved_models'
	model_path = os.path.join(models_dir, 'policy_and_optim.model')
	if os.path.exists(model_path):
		print(f'\nLoading models and optim from {model_path}...\n')
		load_dict = torch.load(model_path)
		for name, v in policy_dict.items():
			v.load_state_dict(load_dict[name])
		print('Done!\n')

	else:
		print(f'\nFile {model_path} DNE! skipping.\n')


def save_whole_model_pickle(pm, **kwargs):

	models_dir = kwargs.get('save_dir', 'saved_models')
	model_fname = os.path.join(models_dir, 'pickle_policy_and_optim.pkl')

	if os.path.exists(model_fname):
		print(f'\nFile {model_fname} already exists! Removing and saving to it now...\n')
		os.remove(model_fname)

	with open(model_fname, 'wb') as f:
		pickle.dump({'policy_model' : pm}, f)


def load_whole_model_pickle(**kwargs):

	models_dir = kwargs.get('save_dir', 'saved_models')
	model_fname = os.path.join(models_dir, 'pickle_policy_and_optim.pkl')

	if not os.path.exists(model_fname):
		print(f'\nFile {model_fname} DNE! Exiting.\n')

	with open(model_fname, 'rb') as f:
		pm = pickle.load(f)

	return pm['policy_model']






#
