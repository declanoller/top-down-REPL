
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import traceback as tb
import os, json
import warnings
import matplotlib.patches as patches
import torch


import canv_utils as cu

warnings.filterwarnings("ignore", category=UserWarning)

plot_labels = {
	'R_mean' : '$R$',
	'Root_node_Rtot' : '$R_{root}$',
	'R_recon_mean' : '$R_{recon}$',
	'R_recon_union' : '$R_{recon, union}$',
	'R_recon_rect' : '$R_{recon, rect}$',
	'R_children_mean' : '$R_{children}$',
	'loss_op_mean' : '$Loss_{op}$',
	'loss_params_mean' : '$Loss_{params}$',
	'loss_canv_1_mean' : '$Loss_{canvas 1}$',
	'loss_canv_2_mean' : '$Loss_{canvas 2}$',
	'log_prob_op_mean' : r'$\mathrm{log}(\pi_{op})$',
	'log_prob_params_mean' : r'$\mathrm{log}(\pi_{params})$',
	'log_prob_canv_1_mean' : r'$\mathrm{log}(\pi_{canv 1})$',
	'log_prob_canv_2_mean' : r'$\mathrm{log}(\pi_{canv 2})$',
	'log_probs_op' : r'$\mathrm{log}(\pi_{op})$',
	'log_probs_params' : r'$\mathrm{log}(\pi_{params})$',
	'log_probs_canv_1' : r'$\mathrm{log}(\pi_{canv 1})$',
	'log_probs_canv_2' : r'$\mathrm{log}(\pi_{canv 2})$',
	'V_op_mean' : r'$V_{op}$',
	'V_params_mean' : r'$V_{params}$',
	'V_canv_1_mean' : r'$V_{canv 1}$',
	'V_canv_2_mean' : r'$V_{canv 2}$',
	'N_nodes' : '$N_{nodes}$',
	'loss_V_op' : '$Loss_{V, op}$',
	'loss_V_params' : '$Loss_{V, params}$',
	'loss_V_canv_1' : '$Loss_{V, canv 1}$',
	'loss_V_canv_2' : '$Loss_{V, canv 2}$',

}


def plot_single_axis(ax, data, col, ylabel, **kwargs):

	data = np.array(data)
	data = data[~np.isnan(data)]

	#ax.plot(data, '-o', markersize=4, color=cols[i])
	ax.plot(data, 'o', markersize=3, alpha=0.3, color=col)
	smoothed = smooth_data(data)
	ax.plot(*smoothed, '-', color='black')
	ax.set_xlabel(kwargs.get('xlabel', 'Episode/batch'), fontsize=13)
	ax.set_ylabel(plot_labels.get(ylabel, ylabel), fontsize=22)

	if np.all(np.array(data) > 0):
		if not kwargs.get('disable_log_scales', False):
			ax.set_yscale('log')

	if kwargs.get('limit_y_range', True) and kwargs.get('disable_log_scales', False):
		y_margin = 0.3*smoothed[1].ptp()
		ax.set_ylim(np.min(smoothed[1]) - y_margin, np.max(smoothed[1]) + y_margin)



def plot_losses(loss_dict, **kwargs):

	cols = [
		'dodgerblue',
		'tomato',
		'seagreen',
		'orchid',
		'orange',
		'cyan',
		'forestgreen',
		'gold',
		'pink',
	]

	key_list = list(loss_dict.keys())
	N_plots = len(key_list)

	# N_cols, N_rows
	col_row_dict = {
		1 : (1, 1),
		2 : (2, 1),
		3 : (3, 1),
		4 : (2, 2),
		5 : (3, 2),
		6 : (3, 2),
		7 : (4, 2),
		8 : (4, 2),
		9 : (4, 3),
		10 : (4, 3),
		11 : (4, 3),
		12 : (4, 3),
		13 : (5, 3),
		14 : (5, 3),
		15 : (5, 3),
		16 : (6, 3),
		17 : (6, 3),
		18 : (6, 3),
		19 : (6, 4),
		20 : (6, 4),
		21 : (6, 4),
		22 : (6, 4),
		23 : (6, 4),
		24 : (6, 4),
	}

	if N_plots in col_row_dict.keys():
		N_cols, N_rows = col_row_dict[N_plots]
	else:
		N_cols = np.ceil(np.sqrt(N_plots)).astype(int)
		N_rows = np.ceil(N_plots/N_cols).astype(int)

	plt.close('all')

	if N_plots > 1:
		plot_w = 3.5
	else:
		plot_w = 7
	plot_h = plot_w

	plt.close('all')
	fig, axes = plt.subplots(N_rows, N_cols, figsize=(plot_w*N_cols, plot_h*N_rows))

	if N_plots > 1:
		for i,ax in enumerate(axes.flatten()):
			if i >= N_plots:
				break

			k = key_list[i]
			data = loss_dict[k]
			col = cols[i % len(cols)]
			plot_single_axis(ax, data, col, k, **kwargs)

	else:
		ax = axes
		i = 0
		k = key_list[i]
		data = loss_dict[k]
		col = cols[i % len(cols)]
		plot_single_axis(ax, data, col, k, **kwargs)


	plt.tight_layout()
	fname = kwargs.get('fname', 'losses.png')
	plt.savefig(fname)

	if kwargs.get('show_plot', False):
		plt.show()


def plot_batch_results(input_batch, output_batch, **kwargs):

	N_test = kwargs.get('N_test', 10)

	#input_imgs = img_list[:N_test]

	N_rows = 5

	plt.close('all')
	s = 8
	fig, axes = plt.subplots(N_rows, N_test, figsize=(8*(N_test/N_rows), 8))

	for i in range(N_test):

		'''axes[0][i].axis('off')
		axes[1][i].axis('off')
		axes[2][i].axis('off')
		axes[3][i].axis('off')
		axes[4][i].axis('off')'''

		axes[0][i].set_xticks([])
		axes[1][i].set_xticks([])
		axes[2][i].set_xticks([])
		axes[3][i].set_xticks([])
		axes[4][i].set_xticks([])

		axes[0][i].set_yticks([])
		axes[1][i].set_yticks([])
		axes[2][i].set_yticks([])
		axes[3][i].set_yticks([])
		axes[4][i].set_yticks([])

		axes[0][i].imshow(input_batch['canv_1s'][i].detach().numpy().squeeze(), cmap='OrRd')
		axes[1][i].imshow(input_batch['canv_2s'][i].detach().numpy().squeeze(), cmap='OrRd')
		axes[2][i].imshow(input_batch['canv_ops'][i].detach().numpy().squeeze(), cmap='OrRd')
		axes[3][i].imshow(output_batch['canv_1_out'][i].detach().numpy().squeeze(), cmap='OrRd')
		axes[4][i].imshow(output_batch['canv_2_out'][i].detach().numpy().squeeze(), cmap='OrRd')




	plt.subplots_adjust(left=0.01, bottom=0.01, right=1-0.01, top=1-0.01, hspace=0.0)


	fname = kwargs.get('fname', 'recon_batch.png')
	plt.savefig(fname)


	if kwargs.get('show_plot', True):
		plt.show()



def plot_execution_tree(tree, **kwargs):

	'''
	Code for drawing images on nodes from:
	https://stackoverflow.com/questions/53967392/creating-a-graph-with-images-as-nodes
	https://gist.github.com/munthe/7de513dc886917860f7b960a51c95e10

	note that order of tight_layout(), set_aspect(), etc, is critical (or the
	nodes don't end up in the right places).


	'''

	####################################
	# Create graph
	####################################

	G = nx.DiGraph()
	N_nodes = len(tree)

	# Add nodes to graph
	[G.add_node(i) for i in range(N_nodes)]

	# Add edges to graph
	for i,node in enumerate(tree):
		parent_id = node['parent_id']
		if parent_id is not None:
			G.add_edge(parent_id, i)

	# Use dot layout to get positions
	pos = nx.nx_agraph.graphviz_layout(G, prog='dot')

	#print('\npos: ', pos)

	if len(pos) > 1:
		pos_x = [p[0] for p in pos.values()]
		pos_y = [p[1] for p in pos.values()]

		range_x = max(pos_x) - min(pos_x)
		range_y = max(pos_y) - min(pos_y)

		asp_rat_hw = range_y/range_x
	else:
		asp_rat_hw = 0.5

	plt.close('all')
	fig_w = 6
	fig = plt.figure(figsize=(fig_w, fig_w*asp_rat_hw))
	#ax = plt.subplot(111)
	#fig, ax = plt.subplots(figsize=(fig_w, fig_w*asp_rat_hw))
	#ax_main = fig.add_axes([0.05, 0.0, 0.9, 0.95])
	ax_main = fig.add_axes([0.0, 0.0, 1, 1])

	ax_main.axis('off')
	'''LRB_margin = 0.03
	top_margin = 0.15
	plt.subplots_adjust(left=LRB_margin, bottom=LRB_margin, right=1-LRB_margin, top=1-top_margin, wspace=0.0, hspace=0.0)'''

	# Draw edges/arrows from parent to children
	nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax_main)

	# If is primitive tree, zoom to single node
	if len(tree) == 1:
		node_plot_size = 0.5
		node_pos = pos[0]
		single_w = 20
		x_lims = (node_pos[0] - single_w, node_pos[0] + single_w)
		y_lims = (node_pos[1] - single_w, node_pos[1] + single_w)
		ax_main.set_xlim(x_lims)
		ax_main.set_ylim(y_lims)

		min_node_node_dist = node_plot_size*np.ptp(x_lims)

	else:

		max_tree_depth = max([n['depth'] for n in tree])
		#print('max tree depth = ', max_tree_depth)

		depth_counts = [0 for d in range(max_tree_depth + 1)]
		depth_min_max = {d:{'min' : 100, 'max' : -100} for d in range(max_tree_depth + 1)}
		depth_pos_x = {d:[] for d in range(1, max_tree_depth + 1)}
		for i,n in enumerate(tree):

			depth_counts[n['depth']] += 1

			if n['depth']==0:
				continue

			depth_pos_x[n['depth']].append(pos[i][0])

		max_depth_width = max(depth_counts)

		min_deltas = []
		for d, positions_x in depth_pos_x.items():
			pos_x = np.array(sorted(positions_x))
			deltas = pos_x[1:] - pos_x[:-1]
			min_deltas.append(min(deltas))

		min_node_node_dist = min(min_deltas)/2

		x_lims = ax_main.get_xlim()
		y_lims = ax_main.get_ylim()

		ax_main.set_xlim(x_lims[0] - min_node_node_dist, x_lims[1] + min_node_node_dist)
		ax_main.set_ylim(y_lims[0] - min_node_node_dist, y_lims[1] + min_node_node_dist)

		x_lims = ax_main.get_xlim()

		#node_plot_size = 0.15 # this is the node canv plot size, as a fraction of the full plot width

		#node_plot_size = (range_x/np.ptp(x_lims))/(2*max_depth_width)
		node_plot_size = 0.9*min_node_node_dist/np.ptp(x_lims)
		#print('node_plot_size: ', node_plot_size)

		#print('x_lims: ', x_lims)

	h = fig.axes[0].get_ylim()[1] - fig.axes[0].get_ylim()[0]

	data_to_display_trans = ax_main.transData.transform
	display_to_fig_trans = fig.transFigure.inverted().transform

	# Create network labels
	labels = {i : n['action'] for i,n in enumerate(tree)}
	# Draw networkx labels, below node canv plots
	y_shift = node_plot_size*h
	pos_higher = {k:(v[0], v[1] - 0.8*min_node_node_dist) for k,v in pos.items()}
	nx.draw_networkx_labels(G, pos_higher, labels, font_size=12)



	####################################
	# Draw node canvases
	####################################

	plot_recon_canvs = False
	if all(['recon_canv' in node.keys() for node in tree]) and kwargs.get('plot_recon_canvs', True):
		plot_recon_canvs = True

	for i,n in enumerate(G):
		xx, yy = data_to_display_trans(pos[n]) # data coords -> display coords
		xa, ya = display_to_fig_trans((xx, yy)) # display coords -> figure coords

		if (not plot_recon_canvs) or (node['recon_canv'] is None):

			node_canv_ax = plt.axes([xa - node_plot_size/2, ya - node_plot_size/2, node_plot_size, node_plot_size])
			node_canv_ax.set_aspect('equal')
			node_canv_ax.set_xticks([])
			node_canv_ax.set_yticks([])

			node = tree[i]

			title_label = get_node_title_label(node, **kwargs)

			node_canv_ax.set_title(title_label, fontsize=8)
			node_canv_ax.imshow(node['spec_canv'], cmap='OrRd')

			if node['params'] is not None:
				if isinstance(node['spec_canv'], torch.Tensor):
					N_side = node['spec_canv'].detach().numpy().shape[0]
				else:
					N_side = len(node['spec_canv'])

				rect = patches.Rectangle(*cu.center_xy_wh_to_corner_xy_wh(N_side, node['params']), linewidth=1.5, edgecolor='dodgerblue', facecolor='none', linestyle='solid')
				node_canv_ax.add_patch(rect)

				snap_rect_params = cu.center_xy_wh_to_corner_xy_wh(N_side, cu.corners_to_center_xy_wh(N_side, cu.center_xy_wh_to_grid_corners(N_side, node['params'])))
				snap_rect = patches.Rectangle(*snap_rect_params, linewidth=1.5, edgecolor='tomato', facecolor='none', linestyle='dashed')
				node_canv_ax.add_patch(snap_rect)

		else:

			node = tree[i]
			#node_canv_ax = plt.axes([0, ya - node_plot_size/2, node_plot_size, node_plot_size])
			node_canv_ax = plt.axes([xa - node_plot_size, ya - node_plot_size/2, node_plot_size, node_plot_size])
			node_canv_ax.set_anchor((1, 0))
			node_canv_ax.set_aspect('equal')
			node_canv_ax.set_xticks([])
			node_canv_ax.set_yticks([])


			title_label = get_node_title_label(node, **kwargs)

			node_canv_ax.set_title(title_label, fontsize=9)
			node_canv_ax.imshow(node['spec_canv'], cmap='OrRd')

			if node['params'] is not None:
				if isinstance(node['spec_canv'], torch.Tensor):
					N_side = node['spec_canv'].detach().numpy().shape[0]
				else:
					N_side = len(node['spec_canv'])

				rect = patches.Rectangle(*cu.center_xy_wh_to_corner_xy_wh(N_side, node['params']), linewidth=1.5, edgecolor='dodgerblue', facecolor='none', linestyle='solid')
				node_canv_ax.add_patch(rect)

				snap_rect_params = cu.center_xy_wh_to_corner_xy_wh(N_side, cu.corners_to_center_xy_wh(N_side, cu.center_xy_wh_to_grid_corners(N_side, node['params'])))
				snap_rect = patches.Rectangle(*snap_rect_params, linewidth=1.5, edgecolor='tomato', facecolor='none', linestyle='dashed')
				node_canv_ax.add_patch(snap_rect)


			######

			node_canv_ax = plt.axes([xa, ya - node_plot_size/2, node_plot_size, node_plot_size])
			node_canv_ax.set_anchor((0, 0))
			node_canv_ax.set_aspect('equal')
			node_canv_ax.set_xticks([])
			node_canv_ax.set_yticks([])


			#node_canv_ax.set_title('recon_canv', fontsize=8)
			node_canv_ax.imshow(node['recon_canv'], cmap='BuPu')



	fname = kwargs.get('fname', 'execution_tree.png')
	plt.savefig(fname)
	if kwargs.get('show_plot', True):
		plt.show()



def write_tree_to_json(tree, fname):

	tree_dict = {i : t for i,t in enumerate(tree)}

	for k,v in tree_dict.items():
		if 'spec_canv' in v.keys():
			#v.pop('spec_canv')
			v['spec_canv'] = v['spec_canv'].tolist()

		for kk,vv in v.items():
			if isinstance(vv, torch.Tensor):
				vv_sq = vv.detach().squeeze()
				if vv_sq.shape:
					v[kk] = vv_sq.tolist()
				else:
					v[kk] = vv_sq.item()


	with open(fname, 'w+') as f:
		json.dump(tree_dict, f, indent=4)





def smooth_data(in_dat):

	'''
	Useful for smoothing data, when you have a ton of points and want fewer,
	or when it's really noisy and you want to see the general trend.

	Expects in_dat to just be a long list of values. Returns a tuple of
	the downsampled x and y, where the x are the indices of the y values,
	so you can easily plot the smoothed version on top of the original.
	'''

	hist = np.array(in_dat)
	N_avg_pts = min(100, len(hist)) #How many points you'll have in the end.

	avg_period = max(1, len(hist) // max(1, N_avg_pts))

	downsampled_x = avg_period*np.array(range(N_avg_pts))
	hist_downsampled_mean = np.array([hist[i*avg_period:(i+1)*avg_period].mean() for i in range(N_avg_pts)])
	return downsampled_x, hist_downsampled_mean




def plot_image_grid(img_grid, **kwargs):

	'''
	Assumes you pass a list of lists of images. Assumes each of the sublists
	are the same size.


	'''

	cols = [
		'dodgerblue',
		'tomato',
		'seagreen',
		'orchid',
		'orange',
		'cyan',
	]

	linestyles = ['solid', 'dashed', 'dotted']

	label_grid = kwargs.get('label_grid', None)
	highlight_boxes = kwargs.get('highlight_boxes', None)

	N_rows = len(img_grid)
	N_cols = len(img_grid[0])

	#print(f'Plotting a {N_rows} by {N_cols} grid...')

	w_max = 15
	h_max = 9

	if label_grid:
		w_to_h_factor = 1.5

		if N_rows*w_to_h_factor >= N_cols:
			ax_h = h_max/N_rows
			ax_w = ax_h/w_to_h_factor
			#print(f'Plot is height limited, using ax_w = {ax_w:.1f}, ax_h = {ax_h:.1f}')
		else:
			ax_w = w_max/N_cols
			ax_h = ax_w*w_to_h_factor
			#print(f'Plot is width limited, using ax_w = {ax_w:.1f}, ax_h = {ax_h:.1f}')

		plot_w = ax_w*N_cols
		plot_h = ax_h*N_rows
		#print(f'Plot dims: w = {plot_w:.1f}, h = {plot_h:.1f}')

	else:

		if N_cols/N_rows < w_max/h_max:
			ax_h = h_max/N_rows
			ax_w = ax_h
			#print(f'Plot is height limited, using ax_w = {ax_w:.2f}, ax_h = {ax_h:.2f}')
		else:
			ax_w = w_max/N_cols
			ax_h = ax_w
			#print(f'Plot is width limited, using ax_w = {ax_w:.2f}, ax_h = {ax_h:.2f}')

		plot_w = ax_w*N_cols
		plot_h = ax_h*N_rows
		#print(f'Plot dims: w = {plot_w:.2f}, h = {plot_h:.2f}')

	plt.close('all')
	fig, axes = plt.subplots(N_rows, N_cols, figsize=(plot_w, plot_h))

	plt.suptitle(kwargs.get('plot_title', ''), fontsize=14)

	#fig, axes = plt.subplots(N_rows, N_cols, figsize=(8*N_cols/N_rows, 8))


	for i in range(N_rows):
		for j in range(N_cols):
			#axes[i][j].axis('off')

			N = len(img_grid[i][j])

			axes[i][j].imshow(img_grid[i][j], cmap='OrRd')

			axes[i][j].set_xlim(-0.5, N - 0.5)
			axes[i][j].set_ylim(-0.5, N - 0.5)


			axes[i][j].set_xticks(np.arange(0, N) - 0.5)
			axes[i][j].set_yticks(np.arange(0, N) - 0.5)
			axes[i][j].set_xticklabels([])
			axes[i][j].set_yticklabels([])
			axes[i][j].set_aspect('equal')
			axes[i][j].grid(color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

			if highlight_boxes is not None:
				box_row = highlight_boxes[i]
				if box_row:
					box_list = box_row[j]
					for col_ind,b in enumerate(box_list):
						rect = patches.Rectangle(*b, linewidth=2, edgecolor=cols[col_ind % len(cols)], facecolor='none', linestyle=linestyles[col_ind % len(linestyles)])
						axes[i][j].add_patch(rect)

			if label_grid:
				label_row = label_grid[i]
				if label_row:
					axes[i][j].set_title(label_row[j], fontsize=7)


	if label_grid:
		plt.subplots_adjust(left=0.01, bottom=0.01, right=1-0.01, top=1-0.1, wspace=0.0, hspace=0.0)
	else:
		plt.subplots_adjust(left=0.01, bottom=0.01, right=1-0.01, top=1-0.01, wspace=0.0, hspace=0.0)

	if kwargs.get('save_plot', True):
		output_dir = kwargs.get('output_dir', './')
		rel_fname = kwargs.get('rel_fname', 'img_grid.png')
		fname = kwargs.get('fname', os.path.join(output_dir, rel_fname))
		plt.savefig(fname)

	if kwargs.get('show_plot', False):
		plt.show()




def draw_canvas(canv):

	#print(canv)

	s = 4
	plt.figure(figsize=(s, s))
	plt.imshow(canv.detach().numpy(), cmap='OrRd')

	#plt.axis('equal')

	#plt.tight_layout()
	plt.show()


def get_node_title_label(node, **kwargs):


	title_label = 'Node {}\n'.format(node['node_id'])
	title_label += r'$R_{tot} = $' + '{:.2f}'.format(node['R_tot'])
	#title_label += '\nR_recon = {:.2f}'.format(node['R_recon'])

	if node['R_children']:
		#R_children_strs = [f'{x:.2f}' for x in node['R_children'].values()]
		#title_label += '\nR_children = {}'.format(R_children_strs)
		pass


	if node['R_children_tot'] is not None:
		#title_label += '\nR_children_tot = {:.2f}'.format(node['R_children_tot'])
		pass

	#if node['V_op'] is not None:
	if False:
		if isinstance(node['V_op'], torch.Tensor):
			title_label += '\nV_op = {:.2f}'.format(node['V_op'].item())
		else:
			title_label += '\nV_op = {:.2f}'.format(node['V_op'])



	if kwargs.get('title_ops_out', False):
		ops_out_strs = [f'{x:.3f}' for x in node['ops_out']]
		title_label += '\npi_op = {}'.format(ops_out_strs)
	if kwargs.get('title_adv_op', False):
		if 'adv_op' in node.keys():
			title_label += '\nadv_op = {:.3f}'.format(node['adv_op'])

	return title_label







#
