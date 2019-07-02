import json
import os
import pickle

from data_utilities import DataParameters
from experimnet_utilities import Experiment, ExperimentParameters
from general_utilies import exp_df_dict_saver
from logger_utilities import Logger
from pnml_utilities import PNMLParameters
from pnml_utilities import twice_universality


# Create parameters
def execute_experiment(params):
    data_params = DataParameters()
    pnml_params = PNMLParameters()
    exp_params = ExperimentParameters()

    # Set parameters from dict
    for key, value in params['exp_params'].items():
        setattr(exp_params, key, value)
    for key, value in params['pnml_params'].items():
        setattr(pnml_params, key, value)
    for key, value in params['data_params'].items():
        setattr(data_params, key, value)

    # Create logger and save params to output folder
    logger = Logger(experiment_type=exp_params.experiment_name, output_root=exp_params.output_dir_base)
    logger.info('OutputDirectory: %s' % logger.output_folder)
    with open(os.path.join(logger.output_folder, 'params.json'), 'w', encoding='utf8') as outfile:
        outfile.write(json.dumps(params, indent=4, sort_keys=True))

    logger.info('%s' % data_params)
    logger.info('%s' % pnml_params)
    logger.info('%s' % exp_params)

    exp_h = Experiment(exp_params, data_params, pnml_params)
    if exp_params.exp_type == 'poly':
        exp_h.execute_poly_degree_search()
    elif exp_params.exp_type == 'lambda':
        exp_h.execute_lambda_search()
    regret_df = exp_h.get_regret_df()
    exp_df_dict = exp_h.get_exp_df_dict()
    x_train, y_train = exp_h.get_train()

    # Twice universal
    logger.info('Execute TU.')
    twice_df = twice_universality(exp_df_dict)
    exp_df_dict['Twice'] = twice_df

    # Save results
    logger.info('Save results.')
    regret_df.to_pickle(os.path.join(logger.output_folder, 'regret_df.pkl'))
    exp_df_dict_saver(exp_df_dict, logger.output_folder)

    trainset_dict = {'x_train': x_train, 'y_train': y_train}
    with open(os.path.join(logger.output_folder, 'trainset_dict.pkl'), "wb") as f:
        pickle.dump(trainset_dict, f)
        f.close()
    logger.info('Finished. Save to: %s' % logger.output_folder)


if __name__ == "__main__":
    with open("params.json") as file:
        params_from_file = json.load(file)

    execute_experiment(params_from_file)
