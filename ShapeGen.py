import torch
from torch.nn import ZeroPad2d
from torch.distributions import Bernoulli, Beta
from copy import deepcopy

import numpy as np
import traceback as tb

import plot_utils
import train_utils
import canv_utils as cu

class ShapeGen:



	def __init__(self, N_side_in, **kwargs):

		self.N_side = N_side_in
		self.canv_shape = (self.N_side, self.N_side)

		self.op_dict = {
			'union' : union,
			'rect' : self.primitive_rect
		}
		#'subtract' : subtract,

		self.op_str_list = list(self.op_dict.keys())
		#print(self.op_str_list)
		self.N_ops = len(self.op_str_list)
		self.N_non_primitive_ops = 1
		self.N_params = 4

		self.zero_pad = ZeroPad2d(1)
		#self.peaky_noise = Beta(0.03*torch.ones(self.canv_shape), 0.47*torch.ones(self.canv_shape))
		#self.peaky_noise = Beta(1*torch.ones(self.canv_shape), 8*torch.ones(self.canv_shape))
		self.peaky_noise = Beta(0.05*torch.ones(self.canv_shape), 0.45*torch.ones(self.canv_shape))

		self.canv_dist = kwargs.get('canv_dist', 'bernoulli')

		assert self.canv_dist in ['bernoulli', 'beta'], 'Canv dist must be either bernoulli or beta!'

		noise_methods = {
			'bernoulli' : 'bern',
			'beta' : 'peaky_blur',
		}

		self.noise_method = noise_methods[self.canv_dist]


	def is_primitive_op(self, op_ind):
		return self.op_str_list[op_ind] == 'rect'


	def op_str_to_OHE(self, op_str):

		op_OHE = torch.zeros(self.N_ops)
		op_OHE[self.op_str_list.index(op_str)] = 1.0
		return op_OHE


	def op_str_to_op_ind(self, op_str):

		return self.op_str_list.index(op_str)


	def op_ind_to_op_str(self, op_ind):

		return self.op_str_list[op_ind]


	def get_random_op_str(self):
		#op_str = self.op_str_list[np.random.randint(0, self.N_non_primitive_ops)]
		op_str = 'union'
		return op_str


	def is_equiv_to_primitive(self, canv):

		unique_dim_0 = np.unique(canv.sum(dim=0).numpy()).tolist()
		unique_dim_1 = np.unique(canv.sum(dim=1).numpy()).tolist()

		if (0 not in unique_dim_0) or (len(unique_dim_0) > 2):
			return False

		if (0 not in unique_dim_1) or (len(unique_dim_1) > 2):
			return False

		return True


	def corners_to_center_xy_wh(self, corners):

		'''

		Arg has form like that returned by get_random_shape_canv():
		(bot, top, left, right), and scaled to the size of the canv.

		Returns a rect with the form (center x, center y, w, h), scaled to
		[0, 1].

		'''

		return cu.corners_to_center_xy_wh(self.N_side, corners)


	def center_xy_wh_to_grid_corners(self, center_xy_wh):

		'''

		This just does the same thing that primitive_rect() does.

		'''

		return cu.center_xy_wh_to_grid_corners(self.N_side, center_xy_wh)



	def center_xy_wh_to_corner_xy_wh(self, center_xy_wh):

		'''

		Takes a rect with the form (center x, center y, w, h), scaled to
		[0, 1].

		Returns a rect with the form ((left x, bot y), w, h), scaled to
		[0, N_side], also with minor adjustments due to the grid of imshow().

		This is only for plotting, because it's what patches.rectangle() takes.

		'''

		return cu.center_xy_wh_to_corner_xy_wh(self.N_side, center_xy_wh)


	def primitive_rect(self, x, y, w, h):

		'''
		x, y, w, h are all in [0, 1], which gets scaled to the canvas size.

		x,y is the *center* of the rect.

		'''

		canvas = torch.zeros(self.N_side, self.N_side)

		bot_coord, top_coord, left_coord, right_coord = self.center_xy_wh_to_grid_corners([x, y, w, h])

		canvas[bot_coord:top_coord+1, left_coord:right_coord+1] = 1.0

		return canvas


	def primitive_circ(self, x, y, r):

		'''
		x, y, w, h are all in [0, 1], which gets scaled to the canvas size.

		x,y is the *center* of the rect.

		'''

		canvas = torch.zeros(self.N_side, self.N_side)

		bot_coord, top_coord, left_coord, right_coord = self.center_xy_wh_to_grid_corners(x, y, w, h)

		canvas[bot_coord:top_coord+1, left_coord:right_coord+1] = 1.0

		return canvas


	def combine_params_list(self, params_list):

		'''
		For combining a list of params in the form returned by get_random_shape_canv(),
		i.e., top/bottom/L/R.

		'''

		center_xy_wh_coords = [self.corners_to_center_xy_wh(p) for p in params_list]
		rects = [self.primitive_rect(*p) for p in center_xy_wh_coords]
		#print(type(rects), len(rects))
		combined = sum(rects).clamp(0.0, 1.0)

		return combined


	def add_noise_to_canv(self, canv, **kwargs):


		if self.noise_method == 'blur':
			canv = self.noise_blur(canv)

		elif self.noise_method == 'peaky':
			canv = self.noise_peaky(canv)

		elif self.noise_method == 'peaky_blur':
			prop_orig = np.random.rand()
			canv = canv*prop_orig + (1 - prop_orig)*self.noise_blur(self.noise_peaky(canv))

		elif self.noise_method == 'bern':

			p_subtract = 0.1*np.random.rand()
			p_add = 0.1*np.random.rand()
			#print(p_subtract, p_add)
			bern_noise_subtract = Bernoulli(p_subtract*torch.ones(self.canv_shape))
			bern_noise_add = Bernoulli(p_add*torch.ones(self.canv_shape))

			canv = canv - bern_noise_subtract.sample()
			canv = canv.clamp(0.0, 1.0)

			canv = canv + bern_noise_add.sample()
			canv = canv.clamp(0.0, 1.0)

		else:
			canv = self.noise_gaussian(canv, **kwargs)


		canv = canv.clamp(0.0, 1.0)
		return canv


	def noise_gaussian(self, canv, **kwargs):

		noise_sd = kwargs.get('noise_sd', 0.1)
		canv += noise_sd*torch.randn(canv.shape)

		canv = canv.clamp(0.0, 1.0)

		return canv


	def noise_blur(self, canv):

		padded = self.zero_pad(canv)

		bot_R = padded[2:, 2:]*(0.5 + torch.rand(canv.shape))
		top_R = padded[:-2, 2:]*(0.5 + torch.rand(canv.shape))
		bot_L = padded[2:, :-2]*(0.5 + torch.rand(canv.shape))
		top_L = padded[:-2, :-2]*(0.5 + torch.rand(canv.shape))

		prop_neighbors = 0.5
		prop_self = 1 - prop_neighbors

		canv = prop_self*canv + (prop_neighbors/4.0)*(bot_R + top_R + bot_L + top_L)

		canv = canv.clamp(0.0, 1.0)

		return canv


	def noise_peaky(self, canv):

		canv = canv + 1.0*self.peaky_noise.sample()

		canv = canv.clamp(0.0, 1.0)

		return canv


	def get_random_shape_canv(self, **kwargs):

		min_size = 2

		max_size = max(3, self.N_side - 5)

		canv_spec = torch.zeros(self.N_side, self.N_side)

		bot_coord = np.random.randint(0, self.N_side - 1 - min_size)
		left_coord = np.random.randint(0, self.N_side - 1 - min_size)

		top_coord = np.random.randint(bot_coord + min_size, min(bot_coord + max_size, self.N_side))
		right_coord = np.random.randint(left_coord + min_size, min(left_coord + max_size, self.N_side))

		#print(top_coord, bot_coord, left_coord, right_coord)

		canv_spec[bot_coord:top_coord+1, left_coord:right_coord+1] = 1.0
		canv_ideal = canv_spec.clone()

		if kwargs.get('with_noise', False):
			canv_spec = self.add_noise_to_canv(canv_spec, **kwargs)

		#print(canvas)

		return {
			'op_str' : 'rect',
			'canv_spec' : canv_spec,
			'params' : [bot_coord, top_coord, left_coord, right_coord],
			'canv_ideal' : canv_ideal,
		}


	def apply_op(self, op_str, canv_1, canv_2):

		op = self.op_dict[op_str]
		canv_out = op(canv_1, canv_2)
		return canv_out


	def get_op_sample(self, **kwargs):

		'''
		Gets a *valid* sample. Checks until it gets one that is actually
		solvable.

		'''

		while True:
			sample = self.produce_op_sample(**kwargs)
			if self.check_op_sample(sample):
				return sample


	def produce_op_sample(self, **kwargs):

		'''

		To make sure it's solvable, just check to make sure after applying the op,
		it's not the same as either of the input canv's.


		'''

		op_str = self.get_random_op_str()
		canv_1 = self.get_random_shape_canv()
		canv_2 = self.get_random_shape_canv()

		canv_spec = self.apply_op(op_str, canv_1['canv_spec'], canv_2['canv_spec'])
		canv_ideal = canv_spec.clone()
		if kwargs.get('with_noise', False):
			canv_spec = self.add_noise_to_canv(canv_spec, **kwargs)

		return  {
					'op_str' : op_str,
					'canv_1' : canv_1['canv_spec'],
					'canv_2' : canv_2['canv_spec'],
					'canv_1_params' : canv_1['params'],
					'canv_2_params' : canv_2['params'],
					'canv_spec' : canv_spec,
					'canv_ideal' : canv_ideal,
					'params_list' : [canv_1['params'], canv_2['params']]
				}




	def produce_compound_op_sample(self, **kwargs):

		'''
		New "branching" method, not recursive.
		'''

		force_op_sample = kwargs.get('force_op_sample', False)
		max_depth_ub = kwargs.get('max_depth_ub', 5)
		depth = kwargs.get('depth', 0)
		max_depth = np.random.randint(max_depth_ub)

		compound_canv = self.get_random_shape_canv()
		compound_canv['params_list'] = [compound_canv['params']]

		if kwargs.get('return_all_canvs', False):
			compound_canv['all_canvs_list'] = [compound_canv['canv_ideal']]

		while True:
			if force_op_sample:
				force_op_sample = False
			else:
				if depth >= max_depth:
					return compound_canv

				if np.random.randint(0,100) < 20:
					return compound_canv

			depth += 1

			attempts = 0
			attempts_lim = 3
			while True:

				op_str = self.get_random_op_str()
				prim_canv = self.get_random_shape_canv()
				prim_canv['params_list'] = [prim_canv['params']]

				canv_ideal = self.apply_op(op_str, compound_canv['canv_ideal'], prim_canv['canv_ideal'])
				params_list = compound_canv['params_list'] + prim_canv['params_list']

				op_dict = {
					'op_str' : op_str,
					'canv_ideal' : canv_ideal,
					'canv_1' : prim_canv['canv_spec'],
					'canv_2' : compound_canv['canv_spec'],
					'params_list' : params_list,
				}

				if kwargs.get('return_all_canvs', False):
					op_dict['all_canvs_list'] = compound_canv['all_canvs_list'] + [prim_canv['canv_ideal']]

				if kwargs.get('with_noise', False):
					canv_spec = self.add_noise_to_canv(canv_ideal, **kwargs)
				else:
					canv_spec = canv_ideal.clone()

				op_dict['canv_spec'] = canv_spec

				if self.check_compound_sample(op_dict) or attempts >= attempts_lim:
					compound_canv = op_dict
					break
				else:
					attempts += 1





	def check_compound_sample(self, sample):

		if len(sample['params_list']) == 1:
			return True

		params_list = sample['params_list']
		for p in params_list:

			p_canv = self.primitive_rect(*self.corners_to_center_xy_wh(p))


			other_params = [pp for pp in params_list if pp!=p]
			# This happens if there was a duplicate in the params list, which is
			# also something we don't want.
			if len(other_params) != (len(params_list)-1):
				return False

			others_combined_canv = self.combine_params_list(other_params)

			overlap = intersection(p_canv, others_combined_canv)
			if torch.all(torch.eq(overlap, p_canv)) or torch.all(torch.eq(overlap, others_combined_canv)):
				return False


		all_combined_canv = self.combine_params_list(params_list)
		if self.is_equiv_to_primitive(all_combined_canv):
			return False

		return True





	def check_op_sample(self, sample):


		# If the resulting canvas is blank
		if torch.max(sample['canv_ideal']) < 0.1:
			return False

		# If one canv is completely encompassed by the other.
		if torch.all(torch.eq(intersection(sample['canv_2'], sample['canv_1']), sample['canv_1'])):
			return False

		if torch.all(torch.eq(intersection(sample['canv_1'], sample['canv_1']), sample['canv_2'])):
			return False

		if self.is_equiv_to_primitive(sample['canv_ideal']):
			return False

		return True


	def get_sample(self, **kwargs):

		op_ind = np.random.randint(0, self.N_ops)

		if op_ind < self.N_non_primitive_ops:
			return self.get_op_sample()
		else:
			return self.get_random_shape_canv(**kwargs)






############################ Grid plotting

	def plot_primitives_grid(self, policy_model, **kwargs):

		'''
		Gets some prim examples, uses the policy model to try fitting to them, plots the fitted
		rects on top of them.

		'''

		N_rows = kwargs.get('N_rows', 3)
		N_cols = kwargs.get('N_cols', 6)

		prim_sample_grid = kwargs.get('prim_sample_grid', None)
		if prim_sample_grid is None:
			prim_sample_grid = [[self.get_random_shape_canv(**kwargs) for c in range(N_cols)] for r in range(N_rows)]

		canv_spec_grid = [[s['canv_spec'] for s in row] for row in prim_sample_grid]
		canv_ideal_grid = [[s['canv_ideal'] for s in row] for row in prim_sample_grid]

		nn_output_dict = [[policy_model.policy_params(s['canv_spec']) for s in row] for row in prim_sample_grid]
		nn_action_dict = [[train_utils.get_action_dict(s) for s in row] for row in nn_output_dict]
		params_grid = [[s['params'].squeeze().tolist() for s in row] for row in nn_action_dict]
		f1_grid = [[cu.F1_score(self.primitive_rect(*p), canv_ideal_grid[i][j]) for j,p in enumerate(r)] for i,r in enumerate(params_grid)]


		label_grid = [['F1 score = {:.3f}'.format(f) for f in row] for row in f1_grid]


		boxes_grid = [[[self.center_xy_wh_to_corner_xy_wh(s), self.center_xy_wh_to_corner_xy_wh(self.corners_to_center_xy_wh(self.center_xy_wh_to_grid_corners(s)))] for s in row] for row in params_grid]


		mean_score = np.mean(f1_grid)
		title_score = f'Mean score = {mean_score:.2f}'
		base_title = kwargs.get('base_title', None)
		if base_title is None:
			plot_title = title_score
		else:
			plot_title = base_title + '\n' + title_score

		plot_utils.plot_image_grid(canv_spec_grid, highlight_boxes=boxes_grid, label_grid=label_grid, plot_title=plot_title, **kwargs)

		return prim_sample_grid


	def plot_operations_grid(self, policy_model, **kwargs):

		'''
		Creates some simple 2-primitive operation canvs, uses the policy model
		to try and figure out the right canvs.


		'''


		N_cols = 10
		blank_canv = torch.zeros(self.get_op_sample()['canv_spec'].shape)

		op_sample_list = kwargs.get('op_sample_list', None)
		if op_sample_list is None:
			op_sample_list = [self.get_op_sample(**kwargs) for c in range(N_cols)]

		#op_sample_list = [self.produce_compound_op_sample() for c in range(N_cols)]
		canv_spec_list = [s['canv_spec'] for s in op_sample_list]
		canv_ideal_list = [s['canv_ideal'] for s in op_sample_list]
		canv_1_true_list = [s['canv_1'] for s in op_sample_list]
		canv_2_true_list = [s['canv_2'] for s in op_sample_list]

		canv_1_params_list = [s['canv_1_params'] for s in op_sample_list]
		canv_2_params_list = [s['canv_2_params'] for s in op_sample_list]

		target_op_OHE = [self.op_str_to_OHE(s['op_str']) for s in op_sample_list]

		op_action_dict_list = [train_utils.get_action_dict(policy_model.policy_op(c), **kwargs) for c in canv_spec_list]

		#canv_action_dict_list = [train_utils.get_action_dict(policy_canv(s['canv_spec'], t)) if s['op_str']!='rect' else None for s,t in zip(op_sample_list, target_op_OHE)]
		canv_1_action_dict_list = [train_utils.get_action_dict(policy_model.policy_canv_1(s['canv_spec'], t), **kwargs) if s['op_str']!='rect' else None for s,t in zip(op_sample_list, target_op_OHE)]

		canv_1_list = [n['canv_1'] if n is not None else blank_canv for n in canv_1_action_dict_list]

		canv_2_action_dict_list = [train_utils.get_action_dict(policy_model.policy_canv_2(s['canv_spec'], a, t), **kwargs) if s['op_str']!='rect' else None for s,t,a in zip(op_sample_list, target_op_OHE, canv_1_list)]

		canv_2_list = [n['canv_2'] if n is not None else blank_canv for n in canv_2_action_dict_list]

		canv_1_labels = ['canv_1' if n is not None else '' for n in canv_1_action_dict_list]
		canv_2_labels = ['canv_2' if n is not None else '' for n in canv_2_action_dict_list]

		target_op_strs = [s['op_str'] for s in op_sample_list]
		sampled_op_strs = [self.op_ind_to_op_str(n['op_ind']) for n in op_action_dict_list]

		#target_op_labels = ['using target:\n{}'.format(s) for s in target_op_strs]
		#sampled_op_labels = ['using sampled:\n{}'.format(s) for s in sampled_op_strs]



		canv_spec_op_list = [self.apply_op(s, c_1, c_2) if s != 'rect' else c_spec for s, c_1, c_2, c_spec in zip(target_op_strs, canv_1_list, canv_2_list, canv_spec_list)]
		#canv_sampled_op_list = [self.apply_op(s, c_1, c_2) if s != 'rect' else blank_canv for s, c_1, c_2 in zip(sampled_op_strs, canv_1_list, canv_2_list)]

		recon_score_list = [cu.F1_score(ideal, canv_op) for ideal, canv_op in zip(canv_ideal_list, canv_spec_op_list)]

		canv1_score_list = [cu.F1_score(canv_1, canv_1_true) if cu.F1_score(canv_1, canv_1_true) > cu.F1_score(canv_1, canv_2_true) else cu.F1_score(canv_1, canv_2_true) for canv_1_true, canv_2_true, canv_1 in zip(canv_1_true_list, canv_2_true_list, canv_1_list)]
		canv2_score_list = [cu.F1_score(canv_2, canv_1_true) if cu.F1_score(canv_2, canv_1_true) > cu.F1_score(canv_2, canv_2_true) else cu.F1_score(canv_2, canv_2_true) for canv_1_true, canv_2_true, canv_2 in zip(canv_1_true_list, canv_2_true_list, canv_2_list)]

		canv1_params_list = [canv_1_p if cu.F1_score(canv_1, canv_1_true) > cu.F1_score(canv_1, canv_2_true) else canv_2_p for canv_1_true, canv_2_true, canv_1, canv_1_p, canv_2_p in zip(canv_1_true_list, canv_2_true_list, canv_1_list, canv_1_params_list, canv_2_params_list)]
		canv2_params_list = [canv_1_p if cu.F1_score(canv_2, canv_1_true) > cu.F1_score(canv_2, canv_2_true) else canv_2_p for canv_1_true, canv_2_true, canv_2, canv_1_p, canv_2_p in zip(canv_1_true_list, canv_2_true_list, canv_2_list, canv_1_params_list, canv_2_params_list)]

		boxes_grid = [
			[],
			[[self.center_xy_wh_to_corner_xy_wh(self.corners_to_center_xy_wh(p))] for p in canv1_params_list],
			[[self.center_xy_wh_to_corner_xy_wh(self.corners_to_center_xy_wh(p))] for p in canv2_params_list],
			[],
		]

		op_labels = ['sampled: {}\nusing target:\n{}\nScore: {:.3f}'.format(s, t, score) for s, t, score in zip(sampled_op_strs, target_op_strs, recon_score_list)]

		label_grid = [
			target_op_strs,
			canv_1_labels,
			canv_2_labels,
			op_labels,
		]

		grid = [
			canv_spec_list,
			canv_1_list,
			canv_2_list,
			canv_spec_op_list,
		]

		mean_score_recon = np.mean(recon_score_list)
		mean_score_canv1 = np.mean(canv1_score_list)
		mean_score_canv2 = np.mean(canv2_score_list)

		title_score = f'Mean recon score = {mean_score_recon:.2f}, Mean canv_1 score = {mean_score_canv1:.2f}, Mean canv_2 score = {mean_score_canv2:.2f}, '
		base_title = kwargs.get('base_title', None)
		if base_title is None:
			plot_title = title_score
		else:
			plot_title = base_title + '\n' + title_score

		plot_utils.plot_image_grid(grid, label_grid=label_grid, plot_title=plot_title, highlight_boxes=boxes_grid, **kwargs)

		return op_sample_list


	def plot_compound_ops_grid(self, **kwargs):

		'''

		Creates some compound ops, and if a policy canv is passed, it will
		try to fit to them.


		'''

		N_cols = kwargs.get('N_cols', 10)
		blank_canv = torch.zeros(self.get_op_sample()['canv_spec'].shape)

		op_sample_list = [self.produce_compound_op_sample(force_op_sample=True, return_all_canvs=True, **kwargs) for c in range(N_cols)]

		#op_sample_list = [self.produce_compound_op_sample() for c in range(N_cols)]
		canv_spec_list = [s['canv_spec'] for s in op_sample_list]
		canv_ideal_list = [s['canv_ideal'] for s in op_sample_list]

		canv_params_list = [s['params_list'] for s in op_sample_list]
		combined_params_list = [self.combine_params_list(s['params_list']) for s in op_sample_list]



		eval_canv_1 = kwargs.get('eval_canv_1', False)
		eval_canv_2 = kwargs.get('eval_canv_2', False)

		if eval_canv_1:

			pm = kwargs.get('policy_model', None)
			assert pm is not None, 'Must supply a policy_model kwarg to eval!'

			target_op_OHE = self.op_str_to_OHE('rect')

			canv_1_list = []
			canv_1_params_list = []
			canv_2_list = []

			for i, target in enumerate(op_sample_list):
				output_dict = pm.policy_canv_1(target['canv_ideal'], target_op_OHE)
				all_canvs_list = target['all_canvs_list']
				all_log_probs = [train_utils.get_log_probs_of_samples(output_dict, canv_1=canv_1, **kwargs)['canv_1_log_prob'].item() for canv_1 in all_canvs_list]

				best_ind = np.argmax(all_log_probs)

				best_canv = train_utils.get_action_dict(output_dict, **kwargs)['canv_1']
				best_params = target['params_list'][best_ind]

				canv_1_list.append(best_canv)
				canv_1_params_list.append(best_params)
				#print([p for i,p in enumerate(target['params_list']) if i!=best_ind])
				combined_params_canv = self.combine_params_list([p for i,p in enumerate(target['params_list']) if i!=best_ind])
				canv_2_list.append(combined_params_canv)

			grid = [
				canv_ideal_list,
				canv_1_list,
			]

			label_grid = [
				['canv_spec' for _ in op_sample_list],
				['sampled canv_1' for _ in op_sample_list],
			]

			boxes_grid = [
				[[self.center_xy_wh_to_corner_xy_wh(self.corners_to_center_xy_wh(p)) for p in p_l] for p_l in canv_params_list],
				[[self.center_xy_wh_to_corner_xy_wh(self.corners_to_center_xy_wh(p_l))] for p_l in canv_1_params_list],
			]

		elif eval_canv_2:

			pm = kwargs.get('policy_model', None)
			assert pm is not None, 'Must supply a policy_model kwarg to eval!'

			target_op_OHE = self.op_str_to_OHE('rect')

			canv_1_list = []
			canv_1_params_list = []
			canv_2_list = []
			canv_2_params_list = []

			for i, target in enumerate(op_sample_list):
				output_dict = pm.policy_canv_2(target['canv_ideal'], target['canv_1'], target_op_OHE)
				canv_2 = train_utils.get_action_dict(output_dict, **kwargs)['canv_2']

				canv_1_list.append(target['canv_1'])
				canv_1_params_list.append(target['params_list'][-1])
				canv_2_list.append(canv_2)
				canv_2_params_list.append(target['params_list'][:-1])

			grid = [
				canv_ideal_list,
				canv_1_list,
				canv_2_list,
			]

			label_grid = [
				['canv_spec' for _ in op_sample_list],
				['input canv_1' for _ in op_sample_list],
				['sampled canv_2' for _ in op_sample_list],
			]

			boxes_grid = [
				[[self.center_xy_wh_to_corner_xy_wh(self.corners_to_center_xy_wh(p)) for p in p_l] for p_l in canv_params_list],
				[[self.center_xy_wh_to_corner_xy_wh(self.corners_to_center_xy_wh(p_l))] for p_l in canv_1_params_list],
				[[self.center_xy_wh_to_corner_xy_wh(self.corners_to_center_xy_wh(p)) for p in p_l] for p_l in canv_2_params_list],
			]

		else:

			canv_1_list = [s['canv_1'] if 'canv_1' in s.keys() else blank_canv for s in op_sample_list]
			canv_2_list = [s['canv_2'] if 'canv_2' in s.keys() else blank_canv  for s in op_sample_list]
			canv_1_params_list = [s['params_list'][-1] for s in op_sample_list]
			canv_2_params_list = [s['params_list'][:-1] for s in op_sample_list]

			grid = [
				canv_ideal_list,
				canv_1_list,
				canv_2_list,
			]

			label_grid = [
				['canv_spec' for _ in op_sample_list],
				['ex canv_1' for _ in op_sample_list],
				['ex canv_2' for _ in op_sample_list],
			]

			boxes_grid = [
				[[self.center_xy_wh_to_corner_xy_wh(self.corners_to_center_xy_wh(p)) for p in p_l] for p_l in canv_params_list],
				[[self.center_xy_wh_to_corner_xy_wh(self.corners_to_center_xy_wh(p_l))] for p_l in canv_1_params_list],
				[[self.center_xy_wh_to_corner_xy_wh(self.corners_to_center_xy_wh(p)) for p in p_l] for p_l in canv_2_params_list],
			]




		plot_utils.plot_image_grid(grid, label_grid=label_grid, highlight_boxes=boxes_grid, **kwargs)


	def plot_example_ops_grid(self, **kwargs):

		'''
		Plots example 2-primitive ops, but doesn't do any fitting.

		'''


		N_cols = kwargs.get('N_cols', 10)
		blank_canv = torch.zeros(self.get_op_sample()['canv_spec'].shape)

		op_sample_list = [self.get_op_sample(**kwargs) for c in range(N_cols)]

		#op_sample_list = [self.produce_compound_op_sample() for c in range(N_cols)]
		canv_spec_list = [s['canv_spec'] for s in op_sample_list]
		canv_ideal_list = [s['canv_ideal'] for s in op_sample_list]
		canv_1_true_list = [s['canv_1'] if 'canv_1' in s.keys() else blank_canv for s in op_sample_list]
		canv_2_true_list = [s['canv_2'] if 'canv_2' in s.keys() else blank_canv  for s in op_sample_list]

		canv_params_list = [s['params_list'] for s in op_sample_list]

		grid = [
			canv_ideal_list,
			canv_1_true_list,
			canv_2_true_list,
		]

		boxes_grid = [
			[[self.center_xy_wh_to_corner_xy_wh(self.corners_to_center_xy_wh(p)) for p in p_l] for p_l in canv_params_list],
			[],
			[],
			[],
			[],
		]

		plot_utils.plot_image_grid(grid, highlight_boxes=boxes_grid, **kwargs)



	def plot_example_compound_ops_grid(self, **kwargs):

		'''

		Creates some compound ops, and if a policy canv is passed, it will
		try to fit to them.


		'''

		N_rows = kwargs.get('N_rows', 3)
		N_cols = kwargs.get('N_cols', 8)
		blank_canv = torch.zeros(self.get_op_sample()['canv_spec'].shape)

		#op_sample_list = [self.produce_compound_op_sample(force_op_sample=True, return_all_canvs=True, **kwargs) for c in range(N_cols)]
		op_sample_grid = [[self.produce_compound_op_sample(force_op_sample=True, return_all_canvs=True, **kwargs) for c in range(N_cols)] for r in range(N_rows)]

		#op_sample_list = [self.produce_compound_op_sample() for c in range(N_cols)]
		canv_ideal_grid = [[s['canv_ideal'] for s in s_row] for s_row in op_sample_grid]

		canv_params_grid = [[s['params_list'] for s in s_row] for s_row in op_sample_grid]

		canv_boxes_grid = [[[self.center_xy_wh_to_corner_xy_wh(self.corners_to_center_xy_wh(p)) for p in p_l] for p_l in p_l_row] for p_l_row in canv_params_grid]

		plot_utils.plot_image_grid(canv_ideal_grid, highlight_boxes=canv_boxes_grid, **kwargs)



	def plot_inspect_op_grid(self, inspect_dict_list_in, **kwargs):

		'''
		For inspecting samples during PT that scored very low.

		'''


		inspect_dict_list = deepcopy(inspect_dict_list_in)
		max_N = 15
		inspect_dict_list = inspect_dict_list[:max_N]
		#print(inspect_dict_list)

		canv_spec_list = [s['canv_spec'] for s in inspect_dict_list]
		canv_ideal_list = [s['canv_ideal'] for s in inspect_dict_list]

		canv_spec_labels = ['log_prob = {:.2f},\ntarget op = {}'.format(d['log_prob'], d['target_op_str']) for d in inspect_dict_list]

		label_grid = [
			canv_spec_labels,
			[],
		]

		grid = [
			canv_spec_list,
			canv_ideal_list,
		]

		base_title = kwargs.get('base_title', None)
		if base_title is None:
			plot_title = ''
		else:
			plot_title = base_title + '\n' + title_score

		plot_utils.plot_image_grid(grid, label_grid=label_grid, plot_title=plot_title, **kwargs)


	def plot_inspect_params_grid(self, inspect_dict_list_in, **kwargs):

		'''
		For inspecting samples that scored really badly during PT.
		'''


		inspect_dict_list = deepcopy(inspect_dict_list_in)
		max_N = 10
		inspect_dict_list = inspect_dict_list[:max_N]
		#print(inspect_dict_list)

		canv_spec_list = [s['canv_spec'] for s in inspect_dict_list]
		canv_ideal_list = [s['canv_ideal'] for s in inspect_dict_list]

		canv_spec_labels = ['log_prob = {:.2f},\nmu = {}\nsigma = {}'.format(d['log_prob'], [f'{m:.2f}' for m in d['params_mu']], [f'{sig:.2f}' for sig in d['params_sigma']]) for d in inspect_dict_list]

		boxes_row = [[self.center_xy_wh_to_corner_xy_wh(s['params_sampled']), self.center_xy_wh_to_corner_xy_wh(self.corners_to_center_xy_wh(s['params']))] for s in inspect_dict_list]


		label_grid = [
			canv_spec_labels,
			[],
		]

		grid = [
			canv_spec_list,
			canv_ideal_list,
		]

		highlight_boxes = [
			boxes_row,
			[],
		]

		base_title = kwargs.get('base_title', None)
		if base_title is None:
			plot_title = ''
		else:
			plot_title = base_title + '\n' + title_score

		plot_utils.plot_image_grid(grid, label_grid=label_grid, plot_title=plot_title, highlight_boxes=highlight_boxes, **kwargs)






def union(canv_1, canv_2):
	return torch.max(canv_1, canv_2)

def intersection(canv_1, canv_2):
	return canv_1*canv_2

# Returns canv_1 - canv_2. Only removes what they have in common (i.e., min of 0).
def subtract(canv_1, canv_2):
	return torch.clamp(canv_1 - canv_2, 0)


def lst_to_str(lst):
	return ', '.join(['{:.2f}'.format(x) for x in lst])














#
