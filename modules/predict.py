import json
import os
from datetime import datetime

import dill
import pandas as pd

# Укажем путь к файлам проекта:
# -> $PROJECT_PATH при запуске в Airflow
# -> иначе - директория уровнем выше при локальном запуске
path = os.environ.get('PROJECT_PATH', '..')


def load_current_model():
    model_filename = f'{path}/data/models/cars_pipe_{datetime.now().strftime("%Y%m%d%H%M")}.pkl'
    with open(model_filename, 'rb') as file:
        model = dill.load(file)

    return model


def prepare_dataframe() -> pd.DataFrame:
    test_dir = f'{path}/data/test'

    samples = []
    for filename in os.listdir(test_dir):
        with open(f'{test_dir}/{filename}') as json_file:
            samples.append(json.load(json_file))

    test_df = pd.DataFrame.from_records(samples)
    return test_df


def make_prediction_df(predictions, ids) -> pd.DataFrame:
    predictions_df = pd.DataFrame()
    predictions_df['car_id'] = ids
    predictions_df['pred'] = predictions

    return predictions_df


def predict():
    model = load_current_model()
    test_df = prepare_dataframe()

    predictions = model.predict(test_df)

    predictions_df = make_prediction_df(predictions, test_df['id'])
    predictions_df.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()
