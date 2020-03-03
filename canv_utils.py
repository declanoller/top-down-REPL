import torch

import numpy as np


mse_fn = torch.nn.MSELoss()

def corners_to_center_xy_wh(N_side, corners):

	'''

	Arg has form like that returned by get_random_shape_canv():
	(bot, top, left, right), and scaled to the size of the canv.

	Returns a rect with the form (center x, center y, w, h), scaled to
	[0, 1].

	'''

	bot, top, left, right = corners

	bot /= N_side
	top /= N_side
	left /= N_side
	right /= N_side

	h = top - bot
	w = right - left

	c_x = 0.5*(left + right)
	c_y = 0.5*(bot + top)

	return [c_x, c_y, w, h]



def center_xy_wh_to_grid_corners(N_side, center_xy_wh):

	'''

	This just does the same thing that primitive_rect() does.

	'''

	x, y, w, h = center_xy_wh

	x = np.clip(x, 0, 1)
	y = np.clip(y, 0, 1)
	w = np.clip(w, 0, 1)
	h = np.clip(h, 0, 1)

	bot_coord = np.round(N_side*(y - h/2)).astype(int)
	top_coord = np.round(N_side*(y + h/2)).astype(int)

	left_coord = np.round(N_side*(x - w/2)).astype(int)
	right_coord = np.round(N_side*(x + w/2)).astype(int)

	bot_coord = np.clip(bot_coord, 0, N_side - 1)
	top_coord = np.clip(top_coord, 0, N_side - 1)
	left_coord = np.clip(left_coord, 0, N_side - 1)
	right_coord = np.clip(right_coord, 0, N_side - 1)

	return [bot_coord, top_coord, left_coord, right_coord]



def center_xy_wh_to_corner_xy_wh(N_side, center_xy_wh):

	'''

	Takes a rect with the form (center x, center y, w, h), scaled to
	[0, 1].

	Returns a rect with the form ((left x, bot y), w, h), scaled to
	[0, N_side], also with minor adjustments due to the grid of imshow().

	This is only for plotting, because it's what patches.rectangle() takes.

	'''

	c_x, c_y, w, h = center_xy_wh

	return [(N_side*(c_x - w/2) - 0.5, N_side*(c_y - h/2) - 0.5), N_side*w + 1, N_side*h + 1]



def F1_score(c1, c2):
	return (2*(c1*c2).sum()/(c1.sum() + c2.sum())).item()


def mse(c1, c2):
	return mse_fn(c1, c2)



















#
