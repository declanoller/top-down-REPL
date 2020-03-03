
import train_utils
from run_utils import *



################### default

run_params = {
    'fname_run_note' : 'PT, default',

    'model_dir' : '/home/declan/Documents/code/top-down-REPL/output/big_run_output/27-02-2020_01-42-59__PT,_default',
    'pretrain_load_model' : True,

    'PT_op_inspect_thresh' : -4,
    'PT_params_inspect_thresh' : -10,
    'PT_canv_1' : False,
    'PT_canv_2' : False,

    'save_losses_data' : False,
    'save_model_locally' : False,

    'pretrain_batches' : 1000,
    'batch_size' : 32,
    'N_side' : 12,
    'N_hidden' : 1024,
}
pretrain_procedure(**run_params)




#
