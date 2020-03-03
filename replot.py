import train_utils
import run_utils
from ShapeGen import ShapeGen


#run_dir = '/home/declan/Documents/code/top-down-REPL/output/big_run_output/27-02-2020_01-42-59__PT,_default'
#run_dir = '/home/declan/Documents/code/top-down-REPL/output/big_run_output/27-02-2020_08-08-34__PT,_no_noise'
#run_utils.load_model_replot(run_dir)


sg = ShapeGen(20)
sg.plot_example_compound_ops_grid(show_plot=True, N_rows=4, N_cols=3)





















#
