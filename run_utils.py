
import torch
from torch import nn


import numpy as np
import traceback as tb
import time, os, json
from functools import reduce
from operator import mul
from datetime import datetime

from NN import *
import plot_utils
import train_utils
from ShapeGen import ShapeGen
from copy import deepcopy

OUTPUT_DIR = 'output'

N_TREES = 15

def pretrain_procedure(**kwargs):

	# Create output dir, save params
	rel_fname = get_rel_fname(**kwargs)

	# Create run dir if there is none
	base_dir = kwargs.get('base_dir', OUTPUT_DIR)
	run_dir = os.path.join(base_dir, rel_fname)
	os.mkdir(run_dir)

	# Load defaults/save run params
	run_params = load_default_run_params_and_update('PT', **kwargs)
	save_run_params(run_dir, 'PT_passed', **kwargs)
	save_run_params(run_dir, 'PT_all', **run_params)

	N_side = kwargs.get('N_side', 30)
	N_hidden = kwargs.get('N_hidden', 500)
	ret_dict = None

	# Get ShapeGen object for creating shapes and such
	sg = ShapeGen(N_side, **run_params)

	# This is for if you want to load one that's already been trained. Otherwise,
	# it creates a new one.
	if kwargs.get('pretrain_load_model', False):
		model_dir = kwargs.get('model_dir', 'saved_models')
		model_fname = os.path.join(model_dir, 'pickle_policy_and_optim.pkl')
		assert os.path.exists(model_fname), 'Model pickle must exist to load! DNE.'
		pm = load_whole_model_pickle(save_dir=model_dir, **run_params)
		pm.reset_switches(**run_params)
	else:
		pm = PolicyModel(N_side, sg.N_ops, sg.N_params, N_hidden, **run_params)

	plot_prim_grid_fn = None
	if kwargs.get('PT_eval_grids', False):
		PT_eval_grids_dir = os.path.join(run_dir, 'PT_eval_grids')
		os.mkdir(PT_eval_grids_dir)
		prim_sample_grid = sg.plot_primitives_grid(pm, output_dir=PT_eval_grids_dir, with_noise=True, rel_fname=f'eval_prim_grid_PT_noise_0.png', base_title=f'PT batch 0')
		plot_prim_grid_fn = lambda pol_model, batch_iter: sg.plot_primitives_grid(pol_model, output_dir=PT_eval_grids_dir, with_noise=True, rel_fname=f'eval_prim_grid_PT_noise_{batch_iter}.png', base_title=f'PT batch {batch_iter}', prim_sample_grid=prim_sample_grid)

	# PT
	print('\nStarting pretraining...\n\n')
	ret_dict = train_utils.pretrain(pm, sg, N_batches=run_params.get('pretrain_batches', 100), plot_prim_grid_fn=plot_prim_grid_fn, **run_params)

	# Save model after run
	if run_params.get('save_model', True):
		save_whole_model_pickle(pm)
		if run_params.get('save_model_locally', True):
			save_whole_model_pickle(pm, save_dir=run_dir)

	# Plot/save losses from PT
	plot_losses(run_dir, ret_dict, 'PT', **run_params)
	if run_params.get('save_losses_data', True):
		save_losses_data(run_dir, ret_dict, 'PT', **run_params)

	# Plot primitive and recon grids
	prim_recon_dict = plot_prim_and_recon_grids(run_dir, 'PT', rel_fname, pm, sg, **run_params)

	# Plot execution trees
	plot_exec_trees(run_dir, 'PT', pm, sg, **run_params)

	# Display inspect samples
	plot_inspect_samples(run_dir, ret_dict, 'PT', sg, **run_params)

	pt_ret_dict = {
		'run_dir' : run_dir,
		'rel_fname' : rel_fname,
		'prim_sample_grid' : prim_recon_dict['prim_sample_grid'],
		'op_sample_list' : prim_recon_dict['op_sample_list'],
	}

	if ret_dict:
		pt_ret_dict['stats'] = {}
		if 'losses' in ret_dict['losses_dict'].keys():
			pt_ret_dict['stats']['min_PT_loss'] = min(ret_dict['losses_dict']['losses'])

		pt_ret_dict['stats']['run_time'] = ret_dict['run_time']



	return pt_ret_dict



def train_procedure(**kwargs_in):

	kwargs = kwargs_in.copy()

	N_side = kwargs.get('N_side', 30)
	batch_size = kwargs.get('batch_size', 32)
	N_hidden = kwargs.get('N_hidden', 500)

	# Create run_dir and rel_fname if needed
	run_dir = kwargs.get('run_dir', None)
	rel_fname = kwargs.get('rel_fname', None)
	if run_dir is None:
		print('\nCreating run_dir for train...\n')
		# Create output dir, save params
		rel_fname = get_rel_fname(**kwargs)
		run_dir = os.path.join(OUTPUT_DIR, rel_fname)
		os.mkdir(run_dir)


	# Load defaults/save run params
	run_params = load_default_run_params_and_update('RL', **kwargs)
	save_run_params(run_dir, 'RL_passed', **kwargs)
	save_run_params(run_dir, 'RL_all', **run_params)

	# Create ShapeGen obj
	sg = ShapeGen(N_side, **run_params)

	# Either load or create new model
	if run_params.get('load_model', True):
		assert os.path.exists(os.path.join('saved_models', 'pickle_policy_and_optim.pkl')), 'Model pickle must exist to load! DNE.'
		pm = load_whole_model_pickle(**run_params)
		pm.reset_all_optimizers(**run_params)
		pm.reset_switches(**run_params)
	else:
		pm = PolicyModel(N_side, sg.N_ops, sg.N_params, N_hidden, **run_params)

	# Option for evaluation before RL begins, so we can check the loaded model
	if run_params.get('before_RL_grids_trees', True):
		# Plot primitive and recon grids
		tree_dir_rel_fname = 'before_RL_exec_trees'
		before_RL_dir = os.path.join(run_dir, tree_dir_rel_fname)
		os.mkdir(before_RL_dir)
		prim_recon_dict = plot_prim_and_recon_grids(before_RL_dir, 'before_RL', rel_fname, pm, sg, **run_params, tree_dir_rel_fname=tree_dir_rel_fname)

		# Plot execution trees
		plot_exec_trees(before_RL_dir, 'before_RL', pm, sg, **run_params)


	# Main RL section. repeat_single is for diagnostics, where it will use
	# the same episode repeatedly.
	if run_params.get('repeat_single', False):
		trees_dir = os.path.join(run_dir, 'same_ep_exec_trees')
		if not os.path.exists(trees_dir):
			os.mkdir(trees_dir)

		ret_dict = train_utils.repeat_single_ep_train(pm, sg, N_batches=run_params.get('train_batches', 100), trees_dir=trees_dir, **run_params)
	else:
		ret_dict = train_utils.train(pm, sg, N_batches=run_params.get('train_batches', 100), **run_params)

	# Plot RL losses
	plot_losses(run_dir, ret_dict, 'RL', **run_params)
	if run_params.get('save_losses_data', True):
		save_losses_data(run_dir, ret_dict, 'RL', **run_params)

	# Plot primitive and recon grids
	prim_recon_dict = plot_prim_and_recon_grids(run_dir, 'RL', rel_fname, pm, sg, **run_params)

	# Plot execution trees
	plot_exec_trees(run_dir, 'RL', pm, sg, **run_params)

	losses_dict = ret_dict['losses_dict']

	rl_ret_dict = {
		'max_R_mean' : max(losses_dict['R_mean']),
		'avg_R_mean' : np.mean(losses_dict['R_mean']),
		'final_R_mean' : losses_dict['R_mean'][-1],
		'run_time' : ret_dict['run_time'],
	}
	if 'loss_tot' in losses_dict.keys():
		rl_ret_dict['min_loss_tot'] = min(losses_dict['loss_tot'])
		rl_ret_dict['final_loss_tot'] = losses_dict['loss_tot'][-1]

	return rl_ret_dict



def pretrain_train_procedure(**kwargs):

	pretrain_kwargs = kwargs.copy()
	train_kwargs = kwargs.copy()

	pt_ret_dict = pretrain_procedure(save_model=True, **pretrain_kwargs)
	rl_ret_dict = train_procedure(load_model=True, **train_kwargs, **pt_ret_dict)

	return {
		'rel_fname' : pt_ret_dict['rel_fname'],
		'pt_ret_dict' : pt_ret_dict['stats'],
		'rl_ret_dict' : rl_ret_dict,
	}



def multirun_PT_RL(run_dict_list):

	base_dir = os.path.join(OUTPUT_DIR, 'multirun_PT_RL_{}'.format(get_date_str()))
	os.mkdir(base_dir)

	results_dict = {}

	for run_kwargs in run_dict_list:

		print('\nNow running with:\n')
		print(run_kwargs)
		print('\n\n')

		pt_rl_ret_dicts = pretrain_train_procedure(base_dir=base_dir, **run_kwargs)

		results_dict[pt_rl_ret_dicts['rel_fname']] = pt_rl_ret_dicts



	all_results_fname = os.path.join(base_dir, 'all_results.json')
	with open(all_results_fname, 'w+') as f:
		json.dump(results_dict, f, indent=4)




def multirun_PT(run_dict_list):

	base_dir = os.path.join(OUTPUT_DIR, 'multirun_PT_{}'.format(get_date_str()))
	os.mkdir(base_dir)

	results_dict = {}

	for run_kwargs in run_dict_list:

		print('\nNow running with:\n')
		print(run_kwargs)
		print('\n\n')

		try:
			pt_ret_dicts = pretrain_procedure(base_dir=base_dir, **run_kwargs)


		except:
			print('\n\nRun failed, continuing...\n\n')
			print(tb.format_exc())



def load_model_replot(run_dir, **kwargs):

	pm, run_params = load_model_params(run_dir)

	sg = ShapeGen(run_params['N_side'], **run_params)

	rel_fname = ''
	plot_prim_and_recon_grids(run_dir, 'PT', rel_fname, pm, sg, **run_params)



###################### Saving/loading data

def get_date_str():
	# Returns the date and time for labeling output.
	return datetime.now().strftime('%d-%m-%Y_%H-%M-%S')


def get_rel_fname(**kwargs):

	fname_run_note = kwargs.get('fname_run_note', None)
	if fname_run_note is None:
		rel_fname = get_date_str()
	else:
		rel_fname = get_date_str() + '__{}'.format(fname_run_note.replace(' ', '_'))

	return rel_fname


def load_default_run_params_and_update(PT_RL_label, **kwargs):

	with open(f'default_params_{PT_RL_label}.json', 'r') as f:
		run_params = json.load(f)

	for k,v in kwargs.items():
		run_params[k] = v

	return run_params


def save_run_params(run_dir, PT_RL_label, **kwargs):

	run_params_fname = os.path.join(run_dir, f'run_params_{PT_RL_label}.json')
	with open(run_params_fname, 'w+') as f:
		json.dump(kwargs, f, indent=4)


def save_losses_data(run_dir, ret_dict, PT_RL_label, **kwargs):

	losses_fname = os.path.join(run_dir, f'losses_{PT_RL_label}.json')
	with open(losses_fname, 'w+') as f:
		json.dump(ret_dict, f, indent=4)


def load_model_params(run_dir):

	model_fname = os.path.join(run_dir, 'pickle_policy_and_optim.pkl')
	params_fname = os.path.join(run_dir, 'run_params_PT_all.json')

	assert os.path.exists(model_fname), 'Model pickle must exist to load! DNE.'
	assert os.path.exists(params_fname), 'Params file must exist to load! DNE.'

	with open(params_fname, 'r') as f:
		run_params = json.load(f)

	pm = load_whole_model_pickle(save_dir=run_dir, **run_params)
	pm.reset_switches(**run_params)

	return pm, run_params



############# Plotting

def plot_losses(run_dir, ret_dict, PT_RL_label, **kwargs):

	print('\nPlotting losses...')

	assert PT_RL_label in ['PT', 'RL'], 'PT_RL_label must be PT or RL'

	if PT_RL_label == 'PT':
		xlabel = 'Sample'
	else:
		xlabel = 'Episode'

	try:
		plot_utils.plot_losses(ret_dict['losses_dict'], limit_y_range=True, fname=os.path.join(run_dir, f'losses_{PT_RL_label}.png'), disable_log_scales=True, xlabel=xlabel)

		other_plots_dir = os.path.join(run_dir, 'other_plots')
		if not os.path.exists(other_plots_dir):
			os.mkdir(other_plots_dir)

		if PT_RL_label == 'RL':

			rewards_dict = {k:ret_dict['losses_dict'][k] for k in ['R_mean', 'Root_node_Rtot', 'R_recon_union', 'R_recon_rect']}
			plot_utils.plot_losses(rewards_dict, limit_y_range=True, fname=os.path.join(other_plots_dir, f'rewards_{PT_RL_label}.png'), disable_log_scales=True, xlabel=xlabel)

			key_list = [
				'loss_op_mean',
				'loss_params_mean',
				'loss_canv_1_mean',
				'loss_canv_2_mean',
				'log_prob_op_mean',
				'log_prob_params_mean',
				'log_prob_canv_1_mean',
				'log_prob_canv_2_mean',
				'V_op_mean',
				'V_params_mean',
				'V_canv_1_mean',
				'V_canv_2_mean',
			]
			loss_probs_V_dict = {k:ret_dict['losses_dict'][k] for k in key_list}
			plot_utils.plot_losses(loss_probs_V_dict, limit_y_range=True, fname=os.path.join(other_plots_dir, f'loss_prob_V_{PT_RL_label}.png'), disable_log_scales=True, xlabel=xlabel)



		plot_utils.plot_losses(ret_dict['losses_dict'], limit_y_range=False, fname=os.path.join(other_plots_dir, f'losses_log_{PT_RL_label}_nolog_noylim.png'), disable_log_scales=True, xlabel=xlabel)
		plot_utils.plot_losses(ret_dict['losses_dict'], limit_y_range=False, fname=os.path.join(other_plots_dir, f'losses_{PT_RL_label}_no_ylim.png'), disable_log_scales=False, xlabel=xlabel)

		if PT_RL_label=='RL' and kwargs.get('RL_eval_episodes', True):
			plot_utils.plot_losses(ret_dict['eval_dict'], limit_y_range=True, fname=os.path.join(run_dir, f'eval_R_{PT_RL_label}.png'), disable_log_scales=True, xlabel='Episode')
			plot_utils.plot_losses(ret_dict['eval_dict'], limit_y_range=False, fname=os.path.join(other_plots_dir, f'eval_R_{PT_RL_label}_no_ylim.png'), disable_log_scales=True, xlabel='Episode')

		if kwargs.get('log_weights_grads', False):
			plot_utils.plot_losses(ret_dict['weights_dict'], limit_y_range=True, fname=os.path.join(run_dir, f'weights_{PT_RL_label}.png'), disable_log_scales=False)
			plot_utils.plot_losses(ret_dict['grads_dict'], limit_y_range=True, fname=os.path.join(run_dir, f'grads_{PT_RL_label}.png'), disable_log_scales=False)

			plot_utils.plot_losses(ret_dict['weights_dict'], limit_y_range=False, fname=os.path.join(other_plots_dir, f'weights_{PT_RL_label}_no_ylim.png'), disable_log_scales=False)
			plot_utils.plot_losses(ret_dict['grads_dict'], limit_y_range=False, fname=os.path.join(other_plots_dir, f'grads_{PT_RL_label}_no_ylim.png'), disable_log_scales=False)


	except:
		print(f'Plotting {PT_RL_label} loss didnt work, continuing...')
		print(tb.format_exc())


def plot_exec_trees(run_dir, PT_RL_label, pm, sg, **kwargs):

	print('\nPlotting execution trees...')

	tree_dir_rel_fname = kwargs.get('tree_dir_rel_fname', 'exec_trees')
	trees_dir = os.path.join(run_dir, tree_dir_rel_fname)
	if not os.path.exists(trees_dir):
		os.mkdir(trees_dir)

	for i in range(N_TREES):
		ep_ret_dict = train_utils.episode(pm, sg, return_tree=True, eval_mode=True, **kwargs)
		plot_utils.plot_execution_tree(ep_ret_dict['tree'], fname=os.path.join(trees_dir, f'{PT_RL_label}_tree_{i}.png'), show_plot=False)
		plot_utils.write_tree_to_json(ep_ret_dict['tree'], os.path.join(trees_dir, f'{PT_RL_label}_tree_{i}.json'))


def plot_prim_and_recon_grids(run_dir, PT_RL_label, rel_fname, pm, sg, **kwargs):

	print('\nPlotting primitive and recon grids...')

	temp_kwargs = deepcopy(kwargs)

	if 'with_noise' in temp_kwargs.keys():
		temp_kwargs.pop('with_noise')
	if 'rel_fname' in temp_kwargs.keys():
		temp_kwargs.pop('rel_fname')

	prim_sample_grid = sg.plot_primitives_grid(pm, output_dir=run_dir, rel_fname=f'prim_grid_{PT_RL_label}.png', base_title=f'{PT_RL_label}\n{rel_fname}', **temp_kwargs)
	sg.plot_primitives_grid(pm, output_dir=run_dir, with_noise=True, rel_fname=f'prim_grid_{PT_RL_label}_noise.png', base_title=f'{PT_RL_label}\n{rel_fname}', **temp_kwargs)

	if 'prim_sample_grid' in temp_kwargs.keys():
		temp_kwargs.pop('prim_sample_grid')
	if 'op_sample_list' in temp_kwargs.keys():
		temp_kwargs.pop('op_sample_list')

	op_sample_list = sg.plot_operations_grid(pm, output_dir=run_dir, rel_fname=f'recon_grid_{PT_RL_label}.png', base_title=f'{PT_RL_label}\n{rel_fname}', **temp_kwargs)
	sg.plot_operations_grid(pm, output_dir=run_dir, with_noise=True, rel_fname=f'recon_grid_{PT_RL_label}_noise.png', base_title=f'{PT_RL_label}\n{rel_fname}', **temp_kwargs)

	sg.plot_compound_ops_grid(output_dir=run_dir, rel_fname=f'compound_ops_{PT_RL_label}_ex.png', base_title=f'{PT_RL_label}\n{rel_fname}', **temp_kwargs)
	sg.plot_compound_ops_grid(policy_model=pm, eval_canv_1=True, output_dir=run_dir, rel_fname=f'compound_ops_{PT_RL_label}_canv_1.png', base_title=f'{PT_RL_label}\n{rel_fname}\npolicy_canv_1', **temp_kwargs)
	sg.plot_compound_ops_grid(policy_model=pm, eval_canv_2=True, output_dir=run_dir, rel_fname=f'compound_ops_{PT_RL_label}_canv_2.png', base_title=f'{PT_RL_label}\n{rel_fname}\npolicy_canv_2', **temp_kwargs)

	return {
		'prim_sample_grid' : prim_sample_grid,
		'op_sample_list' : op_sample_list,
	}


def plot_inspect_samples(run_dir, ret_dict, PT_RL_label, sg, **kwargs):


	if 'PT_op_inspect_samples' in ret_dict.keys():
		print('Num PT_op inspect samps: ', len(ret_dict['PT_op_inspect_samples']))
		sg.plot_inspect_op_grid(ret_dict['PT_op_inspect_samples'], output_dir=run_dir, rel_fname=f'inspect_op_grid_{PT_RL_label}.png')

	if 'PT_params_inspect_samples' in ret_dict.keys():
		print('Num PT_params inspect samps: ', len(ret_dict['PT_params_inspect_samples']))
		sg.plot_inspect_params_grid(ret_dict['PT_params_inspect_samples'], output_dir=run_dir, rel_fname=f'inspect_params_grid_{PT_RL_label}.png')














#
