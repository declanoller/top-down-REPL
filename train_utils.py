
import torch
from torch.distributions import Categorical, Normal, Bernoulli, Beta
from torch import nn

import numpy as np
import traceback as tb
import time, os, itertools
from copy import deepcopy
from queue import Queue

import canv_utils as cu
import plot_utils

#np.set_printoptions(precision=4)


def pretrain(policy_model, sg, **kwargs):

	losses_dict = {}
	weights_dicts_all = []
	grads_dicts_all = []

	start_time = time.time()

	batch_size = kwargs.get('batch_size', 32)
	N_batches = kwargs.get('N_batches', 100)

	PT_op_inspect_thresh = kwargs.get('PT_op_inspect_thresh', None)
	PT_params_inspect_thresh = kwargs.get('PT_params_inspect_thresh', None)
	PT_canv_1_inspect_thresh = kwargs.get('PT_canv_1_inspect_thresh', None)
	PT_canv_2_inspect_thresh = kwargs.get('PT_canv_2_inspect_thresh', None)

	try:
		if kwargs.get('PT_op', True):
			print('\nTraining policy_op now...\n')
			section_start_time = time.time()


			if PT_op_inspect_thresh is not None:
				PT_op_inspect_samples = []

			losses_dict['log_probs_op'] = []
			losses_dict['loss_V_op'] = []
			loss_running_mean = []
			compound_op_N_nodes = np.zeros(8)

			N_batches_op = kwargs.get('N_batches_op', N_batches)

			for b in range(N_batches_op):

				loss = 0
				loss_V = 0
				for _ in range(batch_size):

					if kwargs.get('PT_compound', True):
						target = sg.produce_compound_op_sample(**kwargs)
						compound_op_N_nodes[len(target['params_list'])] += 1
					else:
						target = sg.get_sample(**kwargs)


					target_op_ind = torch.tensor(sg.op_str_to_op_ind(target['op_str']))
					cur_spec = target['canv_spec']
					#print(target_op_ind)
					op_output_dict = policy_model.policy_op(cur_spec)
					#op_output_dict = policy_model.policy_op(cur_spec)

					ideal_score_op = cu.F1_score(cur_spec, torch.tensor(target['canv_ideal']))
					loss_V += 0.5*(op_output_dict['V_op'] - ideal_score_op).pow(2)

					log_prob_dict = get_log_probs_of_samples(op_output_dict, op=target_op_ind, **kwargs)
					loss += -log_prob_dict['op_log_prob']

					losses_dict['log_probs_op'].append(log_prob_dict['op_log_prob'].item())

					if PT_op_inspect_thresh is not None:
						if log_prob_dict['op_log_prob'] <= PT_op_inspect_thresh:
							PT_op_inspect_samples.append({
								'canv_spec' : cur_spec,
								'canv_ideal' : target['canv_ideal'],
								'log_prob' : log_prob_dict['op_log_prob'].item(),
								'target_op_str' : target['op_str']
							})



				loss = loss/batch_size
				loss_V = loss_V/batch_size
				if kwargs.get('pretrain_V', True):
					loss_tot = loss + loss_V
				else:
					loss_tot = loss

				if np.isnan(loss.item()):
					print('\n\nLoss is nan. Breaking.\n\n')
					break

				losses_dict['loss_V_op'].append(loss_V.item())

				policy_model.optimizer_op.zero_grad()
				loss_tot.backward()
				#nn.utils.clip_grad_value_(nn_model.parameters(), 1.0)
				policy_model.optimizer_op.step()

				if kwargs.get('log_weights_grads', False):
					weights_dicts_all.append(policy_model.get_weights_info())
					grads_dicts_all.append(policy_model.get_grads_info())

				loss_running_mean.append(loss.item())
				if len(loss_running_mean) >= 100:
					loss_running_mean.pop(0)

				if b % max(1, N_batches_op // 20) == 0:
					print(f'Batch {b+1}/{N_batches_op} \t Loss = {loss.item():.1f} \t loss_running_mean = {np.mean(loss_running_mean):.2f}')

			print('Section took {:.2f} seconds\n'.format(time.time() - section_start_time))
			print(f'Breakdown of how many nodes: {compound_op_N_nodes}\n')


		if kwargs.get('PT_params', True):
			print('\nTraining policy_params now...\n')
			section_start_time = time.time()

			if PT_params_inspect_thresh is not None:
				PT_params_inspect_samples = []

			losses_dict['log_probs_params'] = []
			losses_dict['loss_V_params'] = []
			loss_running_mean = []

			N_batches_params = kwargs.get('N_batches_params', N_batches)

			for b in range(N_batches_params):

				loss = 0
				loss_V = 0
				for _ in range(batch_size):

					target = sg.get_random_shape_canv(**kwargs)

					target_op_ind = torch.tensor(sg.op_str_to_op_ind(target['op_str']))
					cur_spec = target['canv_spec']

					params = torch.tensor(sg.corners_to_center_xy_wh(target['params']), dtype=torch.float)
					params_output_dict = policy_model.policy_params(cur_spec)

					ideal_score_op = cu.F1_score(cur_spec, torch.tensor(target['canv_ideal']))
					loss_V += 0.5*(params_output_dict['V_params'] - ideal_score_op).pow(2)

					log_prob_dict = get_log_probs_of_samples(params_output_dict, params=params, **kwargs)
					loss += -log_prob_dict['params_log_prob']

					losses_dict['log_probs_params'].append(log_prob_dict['params_log_prob'].item())

					if PT_params_inspect_thresh is not None:
						if log_prob_dict['params_log_prob'] <= PT_params_inspect_thresh:
							PT_params_inspect_samples.append({
								'canv_spec' : cur_spec,
								'canv_ideal' : target['canv_ideal'],
								'log_prob' : log_prob_dict['params_log_prob'].item(),
								'target_op_str' : target['op_str'],
								'params' : target['params'],
								'params_mu' : params_output_dict['params_mu'].squeeze().tolist(),
								'params_sigma' : params_output_dict['params_sigma'].squeeze().tolist(),
								'params_sampled' : get_action_dict(params_output_dict, **kwargs)['params'].squeeze().tolist(),
							})


				loss = loss/batch_size
				loss_V = loss_V/batch_size
				if kwargs.get('pretrain_V', True):
					loss_tot = loss + loss_V
				else:
					loss_tot = loss

				if np.isnan(loss.item()):
					print('\n\nLoss is nan. Breaking.\n\n')
					break

				losses_dict['loss_V_params'].append(loss_V.item())

				policy_model.optimizer_params.zero_grad()
				loss_tot.backward()
				#nn.utils.clip_grad_value_(nn_model.parameters(), 1.0)
				policy_model.optimizer_params.step()

				if kwargs.get('log_weights_grads', False):
					weights_dicts_all.append(policy_model.get_weights_info())
					grads_dicts_all.append(policy_model.get_grads_info())

				loss_running_mean.append(loss.item())
				if len(loss_running_mean) >= 100:
					loss_running_mean.pop(0)

				if b % max(1, N_batches_params // 20) == 0:
					print(f'Batch {b+1}/{N_batches_params} \t Loss = {loss.item():.1f} \t loss_running_mean = {np.mean(loss_running_mean):.2f}')

				plot_prim_grid_fn = kwargs.get('plot_prim_grid_fn', None)
				if plot_prim_grid_fn:
					if b % max(1, N_batches_params // 50) == 0:
						plot_prim_grid_fn(policy_model, b)

			print('Section took {:.2f} seconds\n'.format(time.time() - section_start_time))


		if kwargs.get('PT_canv_1', True):
			print('\nTraining policy_canv_1 now...\n')
			section_start_time = time.time()

			losses_dict['log_probs_canv_1'] = []
			losses_dict['loss_V_canv_1'] = []
			loss_running_mean = []
			compound_op_N_nodes = np.zeros(8)

			N_batches_canv_1 = kwargs.get('N_batches_canv_1', N_batches)

			for b in range(N_batches_canv_1):

				loss = 0
				loss_V = 0
				for _ in range(batch_size):

					if kwargs.get('PT_compound', True):
						target = sg.produce_compound_op_sample(force_op_sample=True, return_all_canvs=True, **kwargs)
						compound_op_N_nodes[len(target['params_list'])] += 1
					else:
						target = sg.get_op_sample(**kwargs)

					target_op_ind = torch.tensor(sg.op_str_to_op_ind(target['op_str']))
					cur_spec = target['canv_spec']

					target_op_OHE = sg.op_str_to_OHE(target['op_str'])


					canv_1_output_dict = policy_model.policy_canv_1(cur_spec, target_op_OHE)

					# For training V
					ideal_score_op = cu.F1_score(cur_spec, torch.tensor(target['canv_ideal']))
					loss_V += 0.5*(canv_1_output_dict['V_canv_1'] - ideal_score_op).pow(2)

					if kwargs.get('PT_compound', True):
						all_canvs_list = target['all_canvs_list']
						all_log_probs = [get_log_probs_of_samples(canv_1_output_dict, canv_1=canv_1, **kwargs)['canv_1_log_prob'] for canv_1 in all_canvs_list]
						canv_1_log_prob = max(all_log_probs)

					else:
						log_prob_dict_1 = get_log_probs_of_samples(canv_1_output_dict, canv_1=target['canv_1'], **kwargs)
						log_prob_dict_2 = get_log_probs_of_samples(canv_1_output_dict, canv_1=target['canv_2'], **kwargs)

						if log_prob_dict_1['canv_1_log_prob'] > log_prob_dict_2['canv_1_log_prob']:
							canv_1_log_prob = log_prob_dict_1['canv_1_log_prob']
						else:
							canv_1_log_prob = log_prob_dict_2['canv_1_log_prob']

					loss += -canv_1_log_prob

					losses_dict['log_probs_canv_1'].append(canv_1_log_prob.item()/sg.N_side**2)

				loss = loss/batch_size
				loss_V = loss_V/batch_size
				if kwargs.get('pretrain_V', True):
					loss_tot = loss + loss_V
				else:
					loss_tot = loss

				if np.isnan(loss.item()):
					print('\n\nLoss is nan. Breaking.\n\n')
					break

				losses_dict['loss_V_canv_1'].append(loss_V.item())

				policy_model.optimizer_canv_1.zero_grad()
				loss_tot.backward()
				#nn.utils.clip_grad_value_(nn_model.parameters(), 1.0)
				policy_model.optimizer_canv_1.step()

				if kwargs.get('log_weights_grads', False):
					weights_dicts_all.append(policy_model.get_weights_info())
					grads_dicts_all.append(policy_model.get_grads_info())


				loss_running_mean.append(loss.item()/sg.N_side**2)
				if len(loss_running_mean) >= 50:
					loss_running_mean.pop(0)

				if b % max(1, N_batches_canv_1 // 20) == 0:
					print(f'Batch {b+1}/{N_batches_canv_1} \t Loss = {loss_running_mean[-1]:.1f} \t loss_running_mean = {np.mean(loss_running_mean):.2f}')

			print('Section took {:.2f} seconds\n'.format(time.time() - section_start_time))
			print(f'Breakdown of how many nodes: {compound_op_N_nodes}\n')


		if kwargs.get('PT_canv_2', True):
			print('\nTraining policy_canv_2 now...\n')
			section_start_time = time.time()

			losses_dict['log_probs_canv_2'] = []
			losses_dict['loss_V_canv_2'] = []
			loss_running_mean = []
			compound_op_N_nodes = np.zeros(8)

			N_batches_canv_2 = kwargs.get('N_batches_canv_2', N_batches)

			for b in range(N_batches_canv_2):

				loss = 0
				loss_V = 0
				for _ in range(batch_size):

					if kwargs.get('PT_compound', True):
						target = sg.produce_compound_op_sample(force_op_sample=True, **kwargs)
						compound_op_N_nodes[len(target['params_list'])] += 1
					else:
						target = sg.get_op_sample(**kwargs)

					target_op_ind = torch.tensor(sg.op_str_to_op_ind(target['op_str']))
					cur_spec = target['canv_spec']

					target_op_OHE = sg.op_str_to_OHE(target['op_str'])

					canv_2_output_dict = policy_model.policy_canv_2(cur_spec, target['canv_1'], target_op_OHE)

					ideal_score_op = cu.F1_score(cur_spec, torch.tensor(target['canv_ideal']))
					ideal_score_canv = 0.5*ideal_score_op + 0.5*1.0
					loss_V += 0.5*(canv_2_output_dict['V_canv_2'] - ideal_score_canv).pow(2)

					log_prob_dict = get_log_probs_of_samples(canv_2_output_dict, canv_2=target['canv_2'], **kwargs)

					loss += -log_prob_dict['canv_2_log_prob']

					losses_dict['log_probs_canv_2'].append(log_prob_dict['canv_2_log_prob'].item()/sg.N_side**2)


				loss = loss/batch_size
				loss_V = loss_V/batch_size
				if kwargs.get('pretrain_V', True):
					loss_tot = loss + loss_V
				else:
					loss_tot = loss

				if np.isnan(loss.item()):
					print('\n\nLoss is nan. Breaking.\n\n')
					break

				losses_dict['loss_V_canv_2'].append(loss_V.item())

				policy_model.optimizer_canv_2.zero_grad()
				loss_tot.backward()
				#nn.utils.clip_grad_value_(nn_model.parameters(), 1.0)
				policy_model.optimizer_canv_2.step()

				if kwargs.get('log_weights_grads', False):
					weights_dicts_all.append(policy_model.get_weights_info())
					grads_dicts_all.append(policy_model.get_grads_info())

				loss_running_mean.append(loss.item()/sg.N_side**2)
				if len(loss_running_mean) >= 50:
					loss_running_mean.pop(0)

				if b % max(1, N_batches_canv_2 // 20) == 0:
					print(f'Batch {b+1}/{N_batches_canv_2} \t Loss = {loss_running_mean[-1]:.1f} \t loss_running_mean = {np.mean(loss_running_mean):.2f}')

			print('Section took {:.2f} seconds\n'.format(time.time() - section_start_time))
			print(f'Breakdown of how many nodes: {compound_op_N_nodes}\n')

	except:
		print('\nSomething ended loop.\n')
		print(tb.format_exc())

	run_time = time.time() - start_time
	print('\n\nRun took {:.2f} seconds\n'.format(run_time))

	ret_dict = {
		'losses_dict' : losses_dict,
		'run_time' : run_time,
	}

	if PT_op_inspect_thresh is not None:
		ret_dict['PT_op_inspect_samples'] = PT_op_inspect_samples
	if PT_params_inspect_thresh is not None:
		ret_dict['PT_params_inspect_samples'] = PT_params_inspect_samples
	if PT_canv_1_inspect_thresh is not None:
		ret_dict['PT_canv_1_inspect_samples'] = PT_canv_1_inspect_samples
	if PT_canv_2_inspect_thresh is not None:
		ret_dict['PT_canv_2_inspect_samples'] = PT_canv_2_inspect_samples


	if kwargs.get('log_weights_grads', False):
		weights_dict = {}
		for k in weights_dicts_all[0].keys():
			weights_dict[k] = [l_d[k] for l_d in weights_dicts_all if l_d[k] is not None]

		grads_dict = {}
		for k in grads_dicts_all[0].keys():
			grads_dict[k] = [l_d[k] for l_d in grads_dicts_all if l_d[k] is not None]

		ret_dict['weights_dict'] = weights_dict
		ret_dict['grads_dict'] = grads_dict

	return ret_dict


def train(policy_model, sg, **kwargs):

	batch_size = kwargs.get('batch_size', 32)
	N_batches = kwargs.get('N_batches', 100)

	loss_dicts_all = []
	eval_dicts_all = []
	ep_stats_all = []
	weights_dicts_all = []
	grads_dicts_all = []

	loss_running_mean = []
	R_running_mean = []
	R_root_running_mean = []

	start_time = time.time()

	try:

		for b in range(N_batches):
			#loss_batch = 0
			R_batch = 0
			R_root_batch = 0
			losses_op = []
			losses_params = []
			losses_canv_1 = []
			losses_canv_2 = []

			for _ in range(batch_size):

				ret_dict = episode(policy_model, sg, **kwargs)
				loss_dict = ret_dict['loss_dict']
				R_batch += loss_dict['R_mean']
				R_root_batch += loss_dict['Root_node_Rtot']

				loss_dicts_all.append(loss_dict)
				ep_stats_all.append(ret_dict['ep_stats_dict'])


				losses_op += ret_dict['losses_op']
				losses_params += ret_dict['losses_params']
				losses_canv_1 += ret_dict['losses_canv_1']
				losses_canv_2 += ret_dict['losses_canv_2']


			if losses_op:
				losses_op_tot = torch.stack(losses_op).mean()
			else:
				losses_op_tot = 0
			if losses_params:
				losses_params_tot = torch.stack(losses_params).mean()
			else:
				losses_params_tot = 0
			if losses_canv_1:
				losses_canv_1_tot = torch.stack(losses_canv_1).mean()
			else:
				losses_canv_1_tot = 0
			if losses_canv_2:
				losses_canv_2_tot = torch.stack(losses_canv_2).mean()
			else:
				losses_canv_2_tot = 0

			loss_batch = 0
			if kwargs.get('RL_op', True):
				loss_batch += losses_op_tot
			if kwargs.get('RL_params', True):
				loss_batch += losses_params_tot
			if kwargs.get('RL_canv_1', True):
				loss_batch += losses_canv_1_tot
			if kwargs.get('RL_canv_2', True):
				loss_batch += losses_canv_2_tot


			R_batch = R_batch/batch_size
			R_root_batch = R_root_batch/batch_size

			policy_model.zero_all_grads()
			loss_batch.backward()

			if kwargs.get('clip_grads', True):
				policy_model.clip_all_grads()

			policy_model.step_all_optims()

			if kwargs.get('log_weights_grads', False):
				weights_dicts_all.append(policy_model.get_weights_info())
				grads_dicts_all.append(policy_model.get_grads_info())

			loss_running_mean.append(loss_batch.item())
			R_running_mean.append(R_batch)
			R_root_running_mean.append(R_root_batch)
			if len(loss_running_mean) >= 100:
				loss_running_mean.pop(0)
				R_running_mean.pop(0)
				R_root_running_mean.pop(0)

			if kwargs.get('RL_eval_episodes', True):
				ret_dict = episode(policy_model, sg, eval_mode=True, **kwargs)
				loss_dict = ret_dict['loss_dict']
				eval_dicts_all.append(loss_dict)

			if b % max(1, N_batches // 20) == 0:
				print(f'Batch {b+1}/{N_batches} \t R_root_running_mean = {np.mean(R_root_running_mean):.2f} \t R_running_mean = {np.mean(R_running_mean):.2f} \t R_batch = {R_batch:.2f} \t loss_batch = {loss_batch.item():.2f} \t loss_running_mean = {np.mean(loss_running_mean):.2f}')


	except:
		print('\n\nSomething ended the loop!\n')
		print(tb.format_exc())

	run_time = time.time() - start_time
	print('\n\nRun took {:.2f} seconds\n'.format(run_time))

	losses_dict = {}
	for k in loss_dicts_all[0].keys():
		losses_dict[k] = [l_d[k].item() if isinstance(l_d[k], torch.Tensor) else l_d[k] for l_d in loss_dicts_all if l_d[k] is not None]

	ep_stats_dict = {}
	for k in ep_stats_all[0].keys():
		ep_stats_dict[k] = list(itertools.chain.from_iterable([l_d[k].item() if isinstance(l_d[k], torch.Tensor) else l_d[k] for l_d in ep_stats_all if l_d[k] is not None]))

	if kwargs.get('RL_eval_episodes', True):
		eval_dict = {}
		for k in ['Root_node_Rtot', 'R_mean']:
			eval_dict[k] = [l_d[k].item() if isinstance(l_d[k], torch.Tensor) else l_d[k] for l_d in eval_dicts_all if l_d[k] is not None]


	ret_dict = {
		'losses_dict' : losses_dict,
		'ep_stats_dict' : ep_stats_dict,
		'eval_dict' : eval_dict,
		'run_time' : run_time,
	}

	if kwargs.get('log_weights_grads', False):
		weights_dict = {}
		for k in weights_dicts_all[0].keys():
			weights_dict[k] = [l_d[k] for l_d in weights_dicts_all if l_d[k] is not None]

		grads_dict = {}
		for k in grads_dicts_all[0].keys():
			grads_dict[k] = [l_d[k] for l_d in grads_dicts_all if l_d[k] is not None]

		ret_dict['weights_dict'] = weights_dict
		ret_dict['grads_dict'] = grads_dict


	return ret_dict


def episode(policy_model, sg, **kwargs):

	target = kwargs.get('target', sg.produce_compound_op_sample(force_op_sample=True)['canv_ideal'])
	blank_canv = torch.zeros(target.shape)

	all_nodes = []
	q = Queue()

	blank_node_dict = {
		'node_id' : None,
		'parent_id' : None,
		'spec_canv' : None,
		'depth' : None,
		'R_recon' : None,
		'R_children' : None,
		'R_children_tot' : None,
		'R_tot' : None,
		'R_canv_1' : None,
		'R_canv_2' : None,
		'action' : None,
		'action_ind' : None,
		'log_prob' : None,
		'log_prob_op' : None,
		'log_prob_params' : None,
		'log_prob_canv_1' : None,
		'log_prob_canv_2' : None,
		'V_op' : None,
		'V_params' : None,
		'V_canv_1' : None,
		'V_canv_2' : None,
		'params' : None,
		'expanded' : False,
		'ops_out' : None,
		'loss_op' : None,
		'loss_params' : None,
		'loss_canv_1' : None,
		'loss_canv_2' : None,
		'child_ids' : None,
		'recon_canv' : None,
		'forced' : False,
	}

	target_dict = blank_node_dict.copy()
	target_dict['node_id'] = 0
	target_dict['spec_canv'] = target
	target_dict['depth'] = 0
	target_dict['R_recon'] = 0
	target_dict['R_children'] = []

	next_node_id = 1

	q.put(target_dict)

	while not q.empty():

		cur_spec_dict = q.get()

		cur_spec = cur_spec_dict['spec_canv']

		# Get action to perform
		op_output_dict = policy_model.policy_op(cur_spec)

		cur_spec_dict['ops_out'] = op_output_dict['ops_out'].detach().flatten().numpy().tolist()
		cur_spec_dict['V_op'] = op_output_dict['V_op']

		if len(all_nodes) < kwargs.get('N_max_terms', 7):
			# If we're still expanding these nodes.
			op_action_dict = get_action_dict(op_output_dict, **kwargs)
			cur_spec_dict['log_prob_op'] = op_action_dict['op_log_prob']
			cur_spec_dict['expanded'] = True
			if kwargs.get('eval_mode', False):
				op_ind_sampled = int(np.argmax(cur_spec_dict['ops_out']))
			else:
				op_ind_sampled = op_action_dict['op_ind']

		else:
			if kwargs.get('force_rect', True):
				# Force it to choose rect
				rect_op_ind = torch.tensor(sg.op_str_to_op_ind('rect'))
				log_prob_dict = get_log_probs_of_samples(op_output_dict, op=rect_op_ind, **kwargs)

				cur_spec_dict['log_prob_op'] = log_prob_dict['op_log_prob']
				op_ind_sampled = rect_op_ind
				# We do this so that it will contribute to the recons (expanded=True),
				# but it won't be used to optimize on.
				cur_spec_dict['expanded'] = True
				cur_spec_dict['forced'] = True
			else:
				# In this case, it will just be a blank canv with no R. We do
				# False and False so it won't contribute to the grads (directly),
				# and it won't get assigned a canv/etc below.
				cur_spec_dict['expanded'] = False
				cur_spec_dict['forced'] = False


		if cur_spec_dict['expanded']:

			# If is primitive op
			if sg.is_primitive_op(op_ind_sampled):

				params_output_dict = policy_model.policy_params(cur_spec)

				params_action_dict = get_action_dict(params_output_dict, **kwargs)

				if kwargs.get('eval_mode', False):
					cur_spec_dict['params'] = params_output_dict['params_mu'].squeeze().tolist()
				else:
					cur_spec_dict['params'] = params_action_dict['params'].squeeze().tolist()

				cur_spec_dict['log_prob_params'] = params_action_dict['params_log_prob']
				cur_spec_dict['V_params'] = params_output_dict['V_params']

			else:

				target_op_OHE = sg.op_str_to_OHE(sg.op_ind_to_op_str(op_ind_sampled))

				canv_1_output_dict = policy_model.policy_canv_1(cur_spec, target_op_OHE)
				canv_1_action_dict = get_action_dict(canv_1_output_dict, **kwargs)

				canv_2_output_dict = policy_model.policy_canv_2(cur_spec, canv_1_action_dict['canv_1'], target_op_OHE)
				canv_2_action_dict = get_action_dict(canv_2_output_dict, **kwargs)

				if kwargs.get('eval_mode', False):
					children = [canv_1_output_dict['canv_1_mu'].detach().round(), canv_2_output_dict['canv_2_mu'].detach().round()]
				else:
					children = [canv_1_action_dict['canv_1'], canv_2_action_dict['canv_2']]

				cur_spec_dict['log_prob_canv_1'] = canv_1_action_dict['canv_1_log_prob']
				cur_spec_dict['log_prob_canv_2'] = canv_2_action_dict['canv_2_log_prob']

				cur_spec_dict['V_canv_1'] = canv_1_output_dict['V_canv_1']
				cur_spec_dict['V_canv_2'] = canv_2_output_dict['V_canv_2']

				# Add children to queue, add child ids to current dict's child id list.
				cur_spec_dict['child_ids'] = {}
				for i,c in enumerate(children):
					#assert (not c['is_terminal']), 'Cannot put a terminal node back in the queue! {}'.format(c)
					c_d = blank_node_dict.copy()
					c_d['node_id'] = next_node_id
					c_d['parent_id'] = cur_spec_dict['node_id']
					c_d['spec_canv'] = c
					c_d['depth'] = cur_spec_dict['depth'] + 1
					c_d['R_recon'] = 0
					c_d['R_children'] = []

					cur_spec_dict['child_ids'][i+1] = next_node_id
					next_node_id += 1

					q.put(c_d)

			# Update dict with actions info in either case
			cur_spec_dict['action'] = sg.op_ind_to_op_str(op_ind_sampled)
			cur_spec_dict['action_ind'] = op_ind_sampled

		# Add this dict to the completed ones
		all_nodes.append(cur_spec_dict)



	# Calculate true reconstructed canvases, and propagate rewards back up
	# towards the root canvas.
	alpha = kwargs.get('alpha', 1.0)
	for child_id in range(len(all_nodes) - 1, -1, -1):
		child_dict = all_nodes[child_id]

		# Get the recon canv for either rect or union op
		if child_dict['action'] == 'rect':
			recon_canv = sg.primitive_rect(*child_dict['params'])
		elif child_dict['action'] == 'union':
			assert len(child_dict['child_ids'])==2, 'Must have two children to apply op!'
			recon_canv = sg.apply_op(child_dict['action'], *[all_nodes[v]['recon_canv'] for v in child_dict['child_ids'].values()])
		else:
			recon_canv = blank_canv
		child_dict['recon_canv'] = recon_canv

		# Get reward based on how well the action applied to the children reconstructs the parent
		child_dict['R_recon'] = cu.F1_score(recon_canv, child_dict['spec_canv'])

		# If node has children, get their rewards and R_children_tot for this node.
		if child_dict['child_ids']:
			child_dict['R_children'] = {k : all_nodes[v]['R_tot'] for k,v in child_dict['child_ids'].items()}
			child_dict['R_children_tot'] = np.mean([v for v in child_dict['R_children'].values()])

		# Figure out R values to use for this node, depending on what type it is.
		if child_dict['action'] == 'rect':
			child_dict['R_tot'] = child_dict['R_recon']
		elif child_dict['action'] == 'union':
			child_dict['R_tot'] = alpha*child_dict['R_recon'] + (1 - alpha)*child_dict['R_children_tot']
			child_dict['R_canv_1'] = (alpha*child_dict['R_recon'] + (1 - alpha)*child_dict['R_children'][1])
			child_dict['R_canv_2'] = (alpha*child_dict['R_recon'] + (1 - alpha)*child_dict['R_children'][2])
		else:
			if kwargs.get('cutoff_node_R', 'zero') == 'V_op':
				child_dict['R_tot'] = child_dict['V_op'].item()
			else:
				child_dict['R_tot'] = 0



	# Only want to optimize for nodes that took an action (i.e., expanded)
	unforced_nodes = [n for n in all_nodes if (not n['forced'] and n['expanded'])]

	# Optimize
	losses_op = []
	losses_params = []
	losses_canv_1 = []
	losses_canv_2 = []

	opt_nodes = kwargs.get('optimize_nodes', 'all')
	for i,node in enumerate(unforced_nodes):

		if kwargs.get('no_adv', False):
			node['V_op'] = torch.tensor(0.0)
			node['V_params'] = torch.tensor(0.0)
			node['V_canv_1'] = torch.tensor(0.0)
			node['V_canv_2'] = torch.tensor(0.0)

		adv_op = node['R_tot'] - node['V_op']
		if kwargs.get('optimize_V', False):
			loss_op = -adv_op.detach().item()*node['log_prob_op'] + 0.5*adv_op.pow(2)
		else:
			loss_op = -adv_op.detach().item()*node['log_prob_op']

		losses_op.append(loss_op)
		node['adv_op'] = adv_op.detach().item()
		node['loss_op'] = loss_op.item()

		if node['action'] == 'rect':
			adv_params = node['R_tot'] - node['V_params']
			if kwargs.get('optimize_V', False):
				loss_params = -adv_params.detach().item()*node['log_prob_params'] + 0.5*adv_params.pow(2)
			else:
				loss_params = -adv_params.detach().item()*node['log_prob_params']

			losses_params.append(loss_params)
			node['adv_params'] = adv_params.detach().item()
			node['loss_params'] = loss_params.item()

		else:
			adv_canv_1 = node['R_canv_1'] - node['V_canv_1']
			adv_canv_2 = node['R_canv_2'] - node['V_canv_2']

			loss_canv_1 = -adv_canv_1.detach().item()*node['log_prob_canv_1']
			loss_canv_2 = -adv_canv_2.detach().item()*node['log_prob_canv_2']

			if kwargs.get('optimize_V', False):
				losses_canv_1.append(loss_canv_1 + 0.5*adv_canv_1.pow(2))
				losses_canv_2.append(loss_canv_2 + 0.5*adv_canv_2.pow(2))
			else:
				losses_canv_1.append(loss_canv_1)
				losses_canv_2.append(loss_canv_2)

			node['adv_canv_1'] = adv_canv_1.detach().item()
			node['loss_canv_1'] = loss_canv_1.item()

			node['adv_canv_2'] = adv_canv_2.detach().item()
			node['loss_canv_2'] = loss_canv_2.item()

		if opt_nodes == 'root_only':
			# If it's root_only, we only want the losses coming from the root
			# node and we can break here.
			break


	# Stats to return for plotting
	log_stats_dict = {}
	log_stats_dict['R_mean'] = [n['R_tot'] for n in unforced_nodes if n['R_tot'] is not None]
	log_stats_dict['R_recon_mean'] = [n['R_recon'] for n in unforced_nodes if n['R_recon'] is not None]
	log_stats_dict['R_recon_union'] = [n['R_recon'] for n in unforced_nodes if (n['R_recon'] is not None) and (n['action']=='union')]
	log_stats_dict['R_recon_rect'] = [n['R_recon'] for n in unforced_nodes if (n['R_recon'] is not None) and (n['action']=='rect')]
	log_stats_dict['R_children_mean'] = [n['R_children_tot'] for n in unforced_nodes if n['R_children_tot'] is not None]

	log_stats_dict['loss_op_mean'] = [n['loss_op'] for n in unforced_nodes if n['loss_op'] is not None]
	log_stats_dict['loss_params_mean'] = [n['loss_params'] for n in unforced_nodes if n['loss_params'] is not None]
	log_stats_dict['loss_canv_1_mean'] = [n['loss_canv_1'] for n in unforced_nodes if n['loss_canv_1'] is not None]
	log_stats_dict['loss_canv_2_mean'] = [n['loss_canv_2'] for n in unforced_nodes if n['loss_canv_2'] is not None]

	log_stats_dict['log_prob_op_mean'] = [n['log_prob_op'].item() for n in unforced_nodes if n['log_prob_op'] is not None]
	log_stats_dict['log_prob_params_mean'] = [n['log_prob_params'].item() for n in unforced_nodes if n['log_prob_params'] is not None]
	log_stats_dict['log_prob_canv_1_mean'] = [n['log_prob_canv_1'].item() for n in unforced_nodes if n['log_prob_canv_1'] is not None]
	log_stats_dict['log_prob_canv_2_mean'] = [n['log_prob_canv_2'].item() for n in unforced_nodes if n['log_prob_canv_2'] is not None]

	log_stats_dict['V_op_mean'] = [n['V_op'].detach().item() for n in unforced_nodes if n['V_op'] is not None]
	log_stats_dict['V_params_mean'] = [n['V_params'].detach().item() for n in unforced_nodes if n['V_params'] is not None]
	log_stats_dict['V_canv_1_mean'] = [n['V_canv_1'].detach().item() for n in unforced_nodes if n['V_canv_1'] is not None]
	log_stats_dict['V_canv_2_mean'] = [n['V_canv_2'].detach().item() for n in unforced_nodes if n['V_canv_2'] is not None]



	ep_stats_dict = deepcopy(log_stats_dict)

	for k, v in log_stats_dict.items():
		if v:
			log_stats_dict[k] = np.mean(v)
		else:
			log_stats_dict[k] = None


	loss_dict = {
					'N_nodes' : len(all_nodes),
					'Root_node_type' : all_nodes[0]['action_ind'],
					'Root_node_Rtot' : all_nodes[0]['R_tot'],
				}

	ret_dict =  {
					'loss_dict' : {**log_stats_dict, **loss_dict},
					'ep_stats_dict' : ep_stats_dict,
					'losses_op' : losses_op,
					'losses_params' : losses_params,
					'losses_canv_1' : losses_canv_1,
					'losses_canv_2' : losses_canv_2,
				}

	if kwargs.get('return_tree', False):
		ret_dict['tree'] = all_nodes

	return ret_dict


def repeat_single_ep_train(policy_model, sg, trees_dir, **kwargs):

	batch_size = kwargs.get('batch_size', 32)
	N_batches = kwargs.get('N_batches', 100)


	if kwargs.get('prim_only', False):
		target = sg.get_random_shape_canv(**kwargs)['canv_spec']
	elif kwargs.get('op_only', False):
		target = sg.get_op_sample(**kwargs)['canv_spec']
	else:
		target = sg.get_sample(**kwargs)['canv_spec']

	#target = sg.get_op_sample()['canv_spec']

	loss_dicts_all = []
	ep_stats_all = []
	weights_dicts_all = []
	grads_dicts_all = []

	loss_running_mean = []
	R_running_mean = []

	start_time = time.time()

	try:

		for b in range(N_batches):
			#loss_batch = 0
			R_batch = 0

			losses_op = []
			losses_params = []
			losses_canv_1 = []
			losses_canv_2 = []

			for _ in range(batch_size):

				ret_dict = episode(policy_model, sg, target=target, return_tree=True, **kwargs)
				loss_dict = ret_dict['loss_dict']
				R_batch += loss_dict['R_mean']

				loss_dicts_all.append(loss_dict)
				ep_stats_all.append(ret_dict['ep_stats_dict'])


				losses_op += ret_dict['losses_op']
				losses_params += ret_dict['losses_params']
				losses_canv_1 += ret_dict['losses_canv_1']
				losses_canv_2 += ret_dict['losses_canv_2']

			#loss_batch = loss_batch/batch_size
			if losses_op:
				losses_op_tot = torch.stack(losses_op).mean()
			else:
				losses_op_tot = 0
			if losses_params:
				losses_params_tot = torch.stack(losses_params).mean()
				#losses_params_tot = 0
			else:
				losses_params_tot = 0
			if losses_canv_1:
				losses_canv_1_tot = torch.stack(losses_canv_1).mean()
			else:
				losses_canv_1_tot = 0
			if losses_canv_2:
				losses_canv_2_tot = torch.stack(losses_canv_2).mean()
			else:
				losses_canv_2_tot = 0

			loss_batch = losses_op_tot + losses_params_tot + losses_canv_1_tot + losses_canv_2_tot
			R_batch = R_batch/batch_size

			if kwargs.get('optimize', True):

				policy_model.zero_all_grads()
				#ret_dict['loss_torch'].backward()
				loss_batch.backward()

				if kwargs.get('clip_grads', True):
					policy_model.clip_all_grads()



				policy_model.step_all_optims()

				if kwargs.get('log_weights_grads', False):
					weights_dicts_all.append(policy_model.get_weights_info())
					grads_dicts_all.append(policy_model.get_grads_info())



			plot_utils.plot_execution_tree(ret_dict['tree'], fname=os.path.join(trees_dir, f'train_tree_{b}.png'), show_plot=False, title_ops_out=True, title_adv_op=True)
			plot_utils.write_tree_to_json(ret_dict['tree'], os.path.join(trees_dir, f'train_tree_{b}.json'))

			if b % max(1, N_batches // 20) == 0:
				print(f'Batch {b+1}/{N_batches}, R_batch = {R_batch:.2f}')


	except:
		print('\n\nSomething ended the loop!\n')
		print(tb.format_exc())


	run_time = time.time() - start_time
	print('\n\nRun took {:.2f} seconds\n'.format(run_time))

	losses_dict = {}
	for k in loss_dicts_all[0].keys():
		losses_dict[k] = [l_d[k] for l_d in loss_dicts_all if l_d[k] is not None]

	ep_stats_dict = {}
	for k in ep_stats_all[0].keys():
		ep_stats_dict[k] = list(itertools.chain.from_iterable([l_d[k] for l_d in ep_stats_all if l_d[k] is not None]))


	ret_dict = {
		'losses_dict' : losses_dict,
		'ep_stats_dict' : ep_stats_dict,
		'run_time' : run_time,
	}


	if kwargs.get('log_weights_grads', False):
		weights_dict = {}
		for k in weights_dicts_all[0].keys():
			weights_dict[k] = [l_d[k] for l_d in weights_dicts_all if l_d[k] is not None]

		grads_dict = {}
		for k in grads_dicts_all[0].keys():
			grads_dict[k] = [l_d[k] for l_d in grads_dicts_all if l_d[k] is not None]

		ret_dict['weights_dict'] = weights_dict
		ret_dict['grads_dict'] = grads_dict

	return ret_dict






def get_action_dict(nn_out_dict, **kwargs):

	ret_dict = {}

	if 'ops_out' in nn_out_dict.keys():
		ret_dict['op_ind'], ret_dict['op_log_prob'] = op_dist_sample_log_prob(nn_out_dict)

	if 'canv_1_mu' in nn_out_dict.keys():
		ret_dict['canv_1'], ret_dict['canv_1_log_prob'] = canv_1_dist_sample_log_prob(nn_out_dict, **kwargs)

	if 'canv_2_mu' in nn_out_dict.keys():
		ret_dict['canv_2'], ret_dict['canv_2_log_prob'] = canv_2_dist_sample_log_prob(nn_out_dict, **kwargs)

	if 'params_mu' in nn_out_dict.keys():
		ret_dict['params'], ret_dict['params_log_prob'] = params_dist_sample_log_prob(nn_out_dict, **kwargs)

	return ret_dict


def get_log_probs_of_samples(nn_out_dict, **kwargs):
	ret_dict = {}

	if 'ops_out' in nn_out_dict.keys():
		_, ret_dict['op_log_prob'] = op_dist_sample_log_prob(nn_out_dict, **kwargs)

	if 'canv_1_mu' in nn_out_dict.keys():
		_, ret_dict['canv_1_log_prob'] = canv_1_dist_sample_log_prob(nn_out_dict, **kwargs)

	if 'canv_2_mu' in nn_out_dict.keys():
		_, ret_dict['canv_2_log_prob'] = canv_2_dist_sample_log_prob(nn_out_dict, **kwargs)

	if 'params_mu' in nn_out_dict.keys():
		_, ret_dict['params_log_prob'] = params_dist_sample_log_prob(nn_out_dict, **kwargs)

	return ret_dict



def op_dist_sample_log_prob(nn_out_dict, **kwargs):


	op_dist = Categorical(nn_out_dict['ops_out'])
	#print(nn_out_dict['ops_out'].detach().tolist())
	op = kwargs.get('op', None)
	if op is None:
		try:
			op = op_dist.sample()
		except:
			print('op dist entry < 0!! Here:')
			print(nn_out_dict['ops_out'])
			exit()

	log_prob = op_dist.log_prob(op).squeeze()

	return op, log_prob


def params_dist_sample_log_prob(nn_out_dict, **kwargs):

	if kwargs.get('params_dist', 'beta') == 'normal':
		params_dist = Normal(nn_out_dict['params_mu'], 0.001 + 0.5*nn_out_dict['params_sigma'])

	else:

		v = 20.0 + nn_out_dict['params_sigma']*5000.0
		a_params = nn_out_dict['params_mu']*v
		b_params = (1 - nn_out_dict['params_mu'])*v

		params_dist = Beta(a_params, b_params)

	params = kwargs.get('params', None)
	if params is None:
		params = params_dist.sample()

	if kwargs.get('log_prob_fn', 'mean') == 'mean':
		log_prob = params_dist.log_prob(params.clamp(0.01, 0.99)).mean()
		#log_prob = params_dist.log_prob(params).mean()
	else:
		log_prob = params_dist.log_prob(params.clamp(0.01, 0.99)).sum()

	return params, log_prob


def canv_1_dist_sample_log_prob(nn_out_dict, **kwargs):


	if kwargs.get('canv_dist', 'bernoulli') == 'bernoulli':
		canv_1_dist = Bernoulli(nn_out_dict['canv_1_mu'])
	else:
		v = 20.0 + nn_out_dict['canv_1_sigma']*5000.0
		a_1 = nn_out_dict['canv_1_mu']*v
		b_1 = (1 - nn_out_dict['canv_1_mu'])*v

		canv_1_dist = Beta(a_1, b_1)

	canv_1 = kwargs.get('canv_1', None)
	if canv_1 is None:
		canv_1 = canv_1_dist.sample()

	if kwargs.get('canv_dist', 'beta') == 'beta':
		canv_1 = canv_1.clamp(0.01, 0.99)

	if kwargs.get('log_prob_fn', 'mean') == 'mean':
		log_prob = canv_1_dist.log_prob(canv_1).mean()
	else:
		log_prob = canv_1_dist.log_prob(canv_1).sum()

	return canv_1, log_prob


def canv_2_dist_sample_log_prob(nn_out_dict, **kwargs):

	if kwargs.get('canv_dist', 'bernoulli') == 'bernoulli':
		canv_2_dist = Bernoulli(nn_out_dict['canv_2_mu'])
	else:
		v = 20.0 + nn_out_dict['canv_2_sigma']*5000.0
		a_1 = nn_out_dict['canv_2_mu']*v
		b_1 = (1 - nn_out_dict['canv_2_mu'])*v

		canv_2_dist = Beta(a_1, b_1)

	canv_2 = kwargs.get('canv_2', None)
	if canv_2 is None:
		canv_2 = canv_2_dist.sample()

	if kwargs.get('canv_dist', 'beta') == 'beta':
		canv_2 = canv_2.clamp(0.01, 0.99)

	if kwargs.get('log_prob_fn', 'mean') == 'mean':
		log_prob = canv_2_dist.log_prob(canv_2).mean()
	else:
		log_prob = canv_2_dist.log_prob(canv_2).sum()

	return canv_2, log_prob




def reward_recon(children, cur_spec, sg, op_ind):

	if sg.is_primitive_op(op_ind):
		return cu.F1_score(children[0], cur_spec)

	else:
		output = sg.apply_op(sg.op_str_list[op_ind], *children)
		return cu.F1_score(output, cur_spec)



def print_grads_info(policy_model):

	#nn.utils.clip_grad_value_(policy_model.policy_params.parameters(), 1.0)
	print('\n')
	for p in policy_model.policy_op.parameters():
		if p.grad is not None:
			print('Max and mean of policy_model.policy_op layer grads: {:.3f}\t{:.3f}'.format(torch.max(torch.abs(p.grad.data)), torch.mean(torch.abs(p.grad.data))))

	for p in policy_model.policy_params.parameters():
		if p.grad is not None:
			print('Max and mean of policy_model.policy_params layer grads: {:.3f}\t{:.3f}'.format(torch.max(torch.abs(p.grad.data)), torch.mean(torch.abs(p.grad.data))))

	for p in policy_model.policy_params.parameters():
		print('Max and mean of policy_model.policy_params layer weights: {:.3f}\t{:.3f}'.format(torch.max(torch.abs(p.data)), torch.mean(torch.abs(p.data))))



def index_to_OHE(ind, N_indices):

	OHE = torch.zeros(N_indices)
	OHE[ind] = 1.0
	return OHE
















#
