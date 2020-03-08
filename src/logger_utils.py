import logging
import os
import pathlib
import sys
import time


class Logger:
    def __init__(self, experiment_type: str, output_root: str):
        """
        Initialize logger class
        :param experiment_type: the experiment type- use for saving string of the outputs/
        :param output_root: the directory to which the output will be saved.
        """

        # Create logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        self.logger = logger
        self.json_file_name = None
        self.results_dict = {}

        self.unique_time = time.strftime("%Y%m%d_%H%M%S")
        self.output_folder = os.path.join(output_root, '%s_%s' % (experiment_type, self.unique_time))
        pathlib.Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        self.define_log_file(os.path.join(self.output_folder, 'log_%s_%s.log' %
                                          (experiment_type, self.unique_time)))

    def define_log_file(self, log_file_name: str):
        """
        create log file to be save into hard disk
        :param log_file_name: the name of the log file
        :return:
        """
        fh = logging.FileHandler(log_file_name)
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)

    def info(self, string_to_print: str):
        """
        print and save to log file in logger style info
        :param string_to_print: string that will be display in the log
        :return:
        """
        self.logger.info(string_to_print)
