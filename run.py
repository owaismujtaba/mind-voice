
import pdb
from src.utils import create_logger, load_config
from src.analysis.p100 import run_p100_pipeline
from src.analysis.n100 import run_n100_pipeline
from src.analysis.motor import MotorPipeline
from src.analysis.snr import run_snr

import pdb


def main():
    config = load_config('config.yaml')
    ########### Analysis
    if config['analysis'].get('p100', {}).get('run_analysis', False):
        logger = create_logger('p100_analysis')
        run_p100_pipeline(config, logger)
        
    if config['analysis'].get('n100', {}).get('run_analysis', False):
        logger = create_logger('n100_analysis')
        run_n100_pipeline(config, logger)
    
    if config['analysis'].get('motor', {}).get('run_analysis', False):
        logger = create_logger('motor_analysis')
        pipe = MotorPipeline(config, logger)
        pipe.run()
    if config['analysis'].get('snr', {}).get('run_analysis', False):
        logger = create_logger('snr_analysis')
        run_snr(config, logger)
        
    ########### Decoding   
    if config['decoding'].get('run_decoding', False):
        logger = create_logger('decoding')
        from src.decoding.decoding import run_decoding_pipeline
        run_decoding_pipeline(config, logger)
        

    ########### P100
    if config['plotting'].get('peak_mean_amplitude_p100', False):
        logger = create_logger('plotting')
        from src.vis.peak_mean_erp_p100 import plot_p100_mean_peak
        plot_p100_mean_peak(config, logger)
        
    if config['plotting'].get('grand_p100', False):
        logger = create_logger('plotting')
        from src.vis.grand_erp_p100 import plot_p100_grand
        plot_p100_grand(config, logger)
    
    #############  Motor
    if config['plotting'].get('motror_covert_overt_rest', False):
        logger = create_logger('plotting')
        from src.vis.motor import plot_motor_average
        plot_motor_average(config, logger)
    
    #############  Decoding Plots
    if config['plotting'].get('accuracy', False):
        logger = create_logger('plotting')
        from src.vis.accuracy import plot_accuracy
        plot_accuracy(logger)
        
    if config['plotting'].get('metrics', False):
        logger = create_logger('plotting')
        from src.vis.metrics import plot_metrics
        plot_metrics(logger)
        
    if config['plotting'].get('cm', False):
        logger = create_logger('plotting')
        from src.vis.confusion_matrix import plot_confusion_matrix
        plot_confusion_matrix(logger)
        
    if config['plotting'].get('per_class_metrics', False):
        logger = create_logger('plotting')
        from src.vis.per_class_metrics import display_classwise
        display_classwise(logger)
        
    if config['plotting'].get('motor_box', False):
        logger = create_logger('plotting')
        from src.vis.motor import plot_motor_box
        plot_motor_box(logger=logger)
        
    if config['plotting'].get('motor_covert_overt_rest', False):
        logger = create_logger('plotting')
        from src.vis.motor import plot_motor_average
        plot_motor_average(logger=logger)
        
        
    ########### N100
    if config['plotting'].get('grand_n100', False):
        logger = create_logger('grand_n100')
        from src.vis.grand_erp_n100 import plot_n100_grand
        plot_n100_grand(config, logger)
        
        
    if config['plotting'].get('peak_mean_amplitude_n100', False):
        logger = create_logger('peak_mean_n100')
        from src.vis.peak_mean_erp_n100 import plot_n100_mean_peak
        plot_n100_mean_peak(config, logger)
        
    if config['plotting'].get('snr_plot', False):
        logger = create_logger('peak_mean_n100')
        from src.vis.snr import plot_snr
        plot_snr(config, logger)
        
    
        
    
    
            

if __name__ == "__main__":
    main()
