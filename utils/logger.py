from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

def create_logger(log_dir, data_type):
    root_output_dir = Path(log_dir)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    
    log_file = 'main_{}_{}.log'.format(time_str, data_type)
    final_log_file = root_output_dir / log_file

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(log_dir) / ('tensorboard_main_' + time_str + '_' + data_type)

    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(tensorboard_log_dir)

def create_logger_autoencoder(log_dir, data_type):
    root_output_dir = Path(log_dir)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    
    log_file = 'autoencoder_{}_{}.log'.format(time_str, data_type)
    final_log_file = root_output_dir / log_file

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(log_dir) / ('tensorboard' + '_autoencoder_' + time_str + '_' + data_type)

    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(tensorboard_log_dir)