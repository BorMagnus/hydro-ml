import os
import sys
from typing import List

import ray
import streamlit as st
from ray import tune

from src.train import train_model

# Get the absolute path of the grandparent directory (Hydro-ML)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
# Add the root directory to sys.path
sys.path.append(root_dir)


dataframe = st.session_state['df']
file_name = st.session_state['file_name']


def get_datetime_and_target_variables(dataframe):
    """Get datetime and target variables from dataframe."""
    datetime_variable = st.selectbox(
        'Select date variable:',
        list(dataframe.columns))

    target_variable_options = list(dataframe.columns)
    target_variable_options.remove(datetime_variable)

    target_variable = st.selectbox(
        'Select target variable:',
        target_variable_options)

    return datetime_variable, target_variable


def get_variables_for_forecast(dataframe, datetime_variable, target_variable):
    """Get variables to be used in forecast."""
    variables_options = list(dataframe.columns)
    variables_options.remove(datetime_variable)
    variables_options.remove(target_variable)

    variables_set = []
    with st.expander("Select variables"):
        with st.form(key='grid_search_form'):
            num_var_sets = st.number_input('How many feature sets do you want to create?', min_value=1, value=1)

            for i in range(num_var_sets):
                variables = st.multiselect(
                    f'Select variables for feature set {i + 1}:',
                    variables_options,
                    variables_options,
                    key=f'feature_set_{i}'
                )
                variables_set.append(variables)

            submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            st.session_state['form_submitted'] = True
            st.session_state['variables_set'] = variables_set
        elif 'form_submitted' not in st.session_state:
            st.session_state['form_submitted'] = False
            st.session_state['variables_set'] = [variables_options]

        
def get_training_parameters_form():
    with st.expander("Select parameters"):
        with st.form(key='training_parameters_form'):
            sequence_length = st.number_input('Sequence length:', min_value=1, value=25, max_value=72)
            batch_size = st.number_input('Batch size:', min_value=1, value=256, max_value=512)
            hidden_size = st.number_input('Hidden size:', min_value=1, value=32, max_value=64)
            num_layers = st.number_input('Number of layers:', min_value=1, value=2, max_value=4)
            learning_rate = st.number_input('Learning rate:', min_value=0.0, value=1e-4, step=1e-4, format="%f")
            weight_decay = st.number_input('Weight decay:', min_value=0.0, value=0.0, step=1e-4, format="%f")
            num_epochs = st.number_input('Number of epochs:', min_value=1, value=100, max_value=200)

            submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            st.session_state['training_parameters_submitted'] = True
            st.session_state['training_parameters'] = {
                'sequence_length': sequence_length,
                'batch_size': batch_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'num_epochs': num_epochs,
            }
        elif 'training_parameters_submitted' not in st.session_state:
            st.session_state['training_parameters_submitted'] = False
            st.session_state['training_parameters'] = {
                "sequence_length": tune.choice([25]),
                "batch_size": tune.choice([256, 512]),
                "hidden_size": tune.choice([32, 64]),
                'num_layers': tune.choice([2, 3, 4]),
                "learning_rate": tune.loguniform(1e-4, 1e-1),
                "weight_decay": tune.choice([0, 0.001, 0.0001]),
                'num_epochs': tune.choice([50, 100, 150, 200]),
            }

        return st.session_state['training_parameters']


def select_models() -> List[str]:
    """Select models to use."""
    st.write('Select Models to use:')
    models = []
    if st.checkbox('FCN'):
        models.append('FCN')
    if st.checkbox('FCNTemporalAttention'):
        models.append('FCNTemporalAttention')
    if st.checkbox('LSTM'):
        models.append('LSTM')
    if st.checkbox('LSTMTemporalAttention'):
        models.append('LSTMTemporalAttention')
    #if st.checkbox('LSTMSpatialAttention'):
    #    models.append('LSTMSpatialAttention')
    if st.checkbox('LSTMSpatialTemporalAttention'):
        models.append('LSTMSpatialTemporalAttention')

    return models


def create_config(datetime_variable, target_variable, models, training_parameters):
    """Create configuration for training."""
    config = {
        "data_file": file_name,
        "datetime":  datetime_variable,
        
        "data": {
            "target_variable": target_variable,
            "sequence_length": training_parameters['sequence_length'],
            "batch_size": training_parameters['batch_size'],
            "variables": tune.grid_search(st.session_state['variables_set'])
        },

        "model": tune.grid_search(models),
        "model_arch": {
            "input_size": None,
            "hidden_size": training_parameters['hidden_size'],
            'num_layers': training_parameters['num_layers'],
            "output_size": 1
        },

        "training": {
            "learning_rate": training_parameters['learning_rate'],
            "weight_decay": training_parameters['weight_decay'],
        },

        'num_epochs': training_parameters['num_epochs'],
    }

    return config


def train(config):
    """Run the training with given configuration."""
    def init_ray():
        try:
            ray.init()
        except:
            ray.shutdown()
            ray.init()

    init_ray()

    reporter = tune.CLIReporter(
        parameter_columns={
            "weight_decay": "w_decay",
            "learning_rate": "lr",
            "num_epochs": "num_epochs"
        },
        metric_columns=[
            "train_loss", "val_loss", "test_loss", "training_iteration"
        ])
    exp_name = "inflow_forecasting"
    local_dir="../ray_results/"
    analysis = tune.run(
        train_model, # TODO: partial(train_cifar, data_dir=data_dir),
        resources_per_trial={"cpu": 12, "gpu": 1},
        config=config,
        num_samples=1, #TODO: Set n_samples
        #scheduler=scheduler, #TODO: Find a scheduler
        progress_reporter=reporter,
        name=exp_name, # #TODO: Set name for experiment
        local_dir=local_dir,
        metric='val_loss',
        mode='min',
    )

    st.session_state['analysis'] = analysis

    # Shutdown Ray after training
    ray.shutdown()


def main():

    dataframe = st.session_state['df']
    datetime_variable, target_variable = get_datetime_and_target_variables(dataframe)
    get_variables_for_forecast(dataframe, datetime_variable, target_variable)
    models = select_models()
    training_parameters = get_training_parameters_form()
    config = create_config(datetime_variable, target_variable, models, training_parameters)
    
    if st.button('Train'):
        with st.spinner('Training...'):
            train(config)

        st.success('Done!')
        results = st.session_state['analysis']
        st.header('Training results')
        df = results.results_df
        st.write(df[['config/model', 'train_loss', 'val_loss', 'test_loss', 'time_total_s', 'config/data/variables']].sort_values('test_loss'))
        st.write(df)

if __name__ == "__main__":
    st.title('Training')

    main()