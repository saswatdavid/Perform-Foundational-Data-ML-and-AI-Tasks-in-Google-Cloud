## QUEST: Perform Foundational Data, ML, and AI Tasks in Google Cloud

## LAB: AI Platform: Qwik Start
## GSP076

git clone https://github.com/GoogleCloudPlatform/training-data-analyst

import os

%%bash
mkdir data
gsutil -m cp gs://cloud-samples-data/ml-engine/census/data/* data/

%%bash
export TRAIN_DATA=$(pwd)/data/adult.data.csv
export EVAL_DATA=$(pwd)/data/adult.test.csv

%%bash
head data/adult.data.csv

%%bash
mkdir -p trainer
touch trainer/__init__.py

##########

%%writefile trainer/util.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import urllib
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf

# Storage directory
DATA_DIR = os.path.join(tempfile.gettempdir(), 'census_data')

# Download options.
DATA_URL = (
    'https://storage.googleapis.com/cloud-samples-data/ai-platform/census'
    '/data')
TRAINING_FILE = 'adult.data.csv'
EVAL_FILE = 'adult.test.csv'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)

# These are the features in the dataset.
# Dataset information: https://archive.ics.uci.edu/ml/datasets/census+income
_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

# This is the label (target) we want to predict.
_LABEL_COLUMN = 'income_bracket'

# These are columns we will not use as features for training. There are many
# reasons not to use certain attributes of data for training. Perhaps their
# values are noisy or inconsistent, or perhaps they encode bias that we do not
# want our model to learn. For a deep dive into the features of this Census
# dataset and the challenges they pose, see the Introduction to ML Fairness
# Notebook: https://colab.research.google.com/github/google/eng-edu/blob
# /master/ml/cc/exercises/intro_to_fairness.ipynb
UNUSED_COLUMNS = ['fnlwgt', 'education', 'gender']

_CATEGORICAL_TYPES = {
    'workclass': pd.api.types.CategoricalDtype(categories=[
        'Federal-gov', 'Local-gov', 'Never-worked', 'Private', 'Self-emp-inc',
        'Self-emp-not-inc', 'State-gov', 'Without-pay'
    ]),
    'marital_status': pd.api.types.CategoricalDtype(categories=[
        'Divorced', 'Married-AF-spouse', 'Married-civ-spouse',
        'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'
    ]),
    'occupation': pd.api.types.CategoricalDtype([
        'Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial',
        'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct',
        'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv',
        'Sales', 'Tech-support', 'Transport-moving'
    ]),
    'relationship': pd.api.types.CategoricalDtype(categories=[
        'Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried',
        'Wife'
    ]),
    'race': pd.api.types.CategoricalDtype(categories=[
        'Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'
    ]),
    'native_country': pd.api.types.CategoricalDtype(categories=[
        'Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic',
        'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece',
        'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong',
        'Hungary',
        'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos',
        'Mexico',
        'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines',
        'Poland',
        'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand',
        'Trinadad&Tobago', 'United-States', 'Vietnam', 'Yugoslavia'
    ]),
    'income_bracket': pd.api.types.CategoricalDtype(categories=[
        '<=50K', '>50K'
    ])
}


def _download_and_clean_file(filename, url):
    """Downloads data from url, and makes changes to match the CSV format.

    The CSVs may use spaces after the comma delimters (non-standard) or include
    rows which do not represent well-formed examples. This function strips out
    some of these problems.

    Args:
      filename: filename to save url to
      url: URL of resource to download
    """
    temp_file, _ = urllib.request.urlretrieve(url)
    with tf.io.gfile.GFile(temp_file, 'r') as temp_file_object:
        with tf.io.gfile.GFile(filename, 'w') as file_object:
            for line in temp_file_object:
                line = line.strip()
                line = line.replace(', ', ',')
                if not line or ',' not in line:
                    continue
                if line[-1] == '.':
                    line = line[:-1]
                line += '\n'
                file_object.write(line)
    tf.io.gfile.remove(temp_file)


def download(data_dir):
    """Downloads census data if it is not already present.

    Args:
      data_dir: directory where we will access/save the census data
    """
    tf.io.gfile.makedirs(data_dir)

    training_file_path = os.path.join(data_dir, TRAINING_FILE)
    if not tf.io.gfile.exists(training_file_path):
        _download_and_clean_file(training_file_path, TRAINING_URL)

    eval_file_path = os.path.join(data_dir, EVAL_FILE)
    if not tf.io.gfile.exists(eval_file_path):
        _download_and_clean_file(eval_file_path, EVAL_URL)

    return training_file_path, eval_file_path


def preprocess(dataframe):
    """Converts categorical features to numeric. Removes unused columns.

    Args:
      dataframe: Pandas dataframe with raw data

    Returns:
      Dataframe with preprocessed data
    """
    dataframe = dataframe.drop(columns=UNUSED_COLUMNS)

    # Convert integer valued (numeric) columns to floating point
    numeric_columns = dataframe.select_dtypes(['int64']).columns
    dataframe[numeric_columns] = dataframe[numeric_columns].astype('float32')

    # Convert categorical columns to numeric
    cat_columns = dataframe.select_dtypes(['object']).columns
    dataframe[cat_columns] = dataframe[cat_columns].apply(lambda x: x.astype(
        _CATEGORICAL_TYPES[x.name]))
    dataframe[cat_columns] = dataframe[cat_columns].apply(lambda x: x.cat.codes)
    return dataframe


def standardize(dataframe):
    """Scales numerical columns using their means and standard deviation to get
    z-scores: the mean of each numerical column becomes 0, and the standard
    deviation becomes 1. This can help the model converge during training.

    Args:
      dataframe: Pandas dataframe

    Returns:
      Input dataframe with the numerical columns scaled to z-scores
    """
    dtypes = list(zip(dataframe.dtypes.index, map(str, dataframe.dtypes)))
    # Normalize numeric columns.
    for column, dtype in dtypes:
        if dtype == 'float32':
            dataframe[column] -= dataframe[column].mean()
            dataframe[column] /= dataframe[column].std()
    return dataframe


def load_data():
    """Loads data into preprocessed (train_x, train_y, eval_y, eval_y)
    dataframes.

    Returns:
      A tuple (train_x, train_y, eval_x, eval_y), where train_x and eval_x are
      Pandas dataframes with features for training and train_y and eval_y are
      numpy arrays with the corresponding labels.
    """
    # Download Census dataset: Training and eval csv files.
    training_file_path, eval_file_path = download(DATA_DIR)

    # This census data uses the value '?' for missing entries. We use
    # na_values to
    # find ? and set it to NaN.
    # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv
    # .html
    train_df = pd.read_csv(training_file_path, names=_CSV_COLUMNS,
                           na_values='?')
    eval_df = pd.read_csv(eval_file_path, names=_CSV_COLUMNS, na_values='?')

    train_df = preprocess(train_df)
    eval_df = preprocess(eval_df)

    # Split train and eval data with labels. The pop method copies and removes
    # the label column from the dataframe.
    train_x, train_y = train_df, train_df.pop(_LABEL_COLUMN)
    eval_x, eval_y = eval_df, eval_df.pop(_LABEL_COLUMN)

    # Join train_x and eval_x to normalize on overall means and standard
    # deviations. Then separate them again.
    all_x = pd.concat([train_x, eval_x], keys=['train', 'eval'])
    all_x = standardize(all_x)
    train_x, eval_x = all_x.xs('train'), all_x.xs('eval')

    # Reshape label columns for use with tf.data.Dataset
    train_y = np.asarray(train_y).astype('float32').reshape((-1, 1))
    eval_y = np.asarray(eval_y).astype('float32').reshape((-1, 1))

    return train_x, train_y, eval_x, eval_y


%%writefile trainer/model.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def input_fn(features, labels, shuffle, num_epochs, batch_size):
    """Generates an input function to be used for model training.

    Args:
      features: numpy array of features used for training or inference
      labels: numpy array of labels for each example
      shuffle: boolean for whether to shuffle the data or not (set True for
        training, False for evaluation)
      num_epochs: number of epochs to provide the data for
      batch_size: batch size for training

    Returns:
      A tf.data.Dataset that can provide data to the Keras model for training or
        evaluation
    """
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features))

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def create_keras_model(input_dim, learning_rate):
    """Creates Keras Model for Binary Classification.

    The single output node + Sigmoid activation makes this a Logistic
    Regression.

    Args:
      input_dim: How many features the input has
      learning_rate: Learning rate for training

    Returns:
      The compiled Keras model (still needs to be trained)
    """
    Dense = tf.keras.layers.Dense
    model = tf.keras.Sequential(
        [
            Dense(100, activation=tf.nn.relu, kernel_initializer='uniform',
                  input_shape=(input_dim,)),
            Dense(75, activation=tf.nn.relu),
            Dense(50, activation=tf.nn.relu),
            Dense(25, activation=tf.nn.relu),
            Dense(1, activation=tf.nn.sigmoid)
        ])

    # Custom Optimizer:
    # https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
    optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)

    # Compile Keras model
    model.compile(
        loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


%%writefile trainer/task.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

from . import model
from . import util

import tensorflow as tf


def get_args():
    """Argument parser.

    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting '
             'models')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=20,
        help='number of times to go through the data, default=20')
    parser.add_argument(
        '--batch-size',
        default=128,
        type=int,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--learning-rate',
        default=.01,
        type=float,
        help='learning rate for gradient descent, default=.01')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    args, _ = parser.parse_known_args()
    return args


def train_and_evaluate(args):
    """Trains and evaluates the Keras model.

    Uses the Keras model defined in model.py and trains on data loaded and
    preprocessed in util.py. Saves the trained model in TensorFlow SavedModel
    format to the path defined in part by the --job-dir argument.

    Args:
      args: dictionary of arguments - see get_args() for details
    """

    train_x, train_y, eval_x, eval_y = util.load_data()

    # dimensions
    num_train_examples, input_dim = train_x.shape
    num_eval_examples = eval_x.shape[0]

    # Create the Keras Model
    keras_model = model.create_keras_model(
        input_dim=input_dim, learning_rate=args.learning_rate)

    # Pass a numpy array by passing DataFrame.values
    training_dataset = model.input_fn(
        features=train_x.values,
        labels=train_y,
        shuffle=True,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size)

    # Pass a numpy array by passing DataFrame.values
    validation_dataset = model.input_fn(
        features=eval_x.values,
        labels=eval_y,
        shuffle=False,
        num_epochs=args.num_epochs,
        batch_size=num_eval_examples)

    # Setup Learning Rate decay.
    lr_decay_cb = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: args.learning_rate + 0.02 * (0.5 ** (1 + epoch)),
        verbose=True)

    # Setup TensorBoard callback.
    tensorboard_cb = tf.keras.callbacks.TensorBoard(
        os.path.join(args.job_dir, 'keras_tensorboard'),
        histogram_freq=1)

    # Train model
    keras_model.fit(
        training_dataset,
        steps_per_epoch=int(num_train_examples / args.batch_size),
        epochs=args.num_epochs,
        validation_data=validation_dataset,
        validation_steps=1,
        verbose=1,
        callbacks=[lr_decay_cb, tensorboard_cb])

    export_path = os.path.join(args.job_dir, 'keras_export')
    tf.keras.models.save_model(keras_model, export_path)
    print('Model exported to: {}'.format(export_path))



if __name__ == '__main__':
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        args = get_args()
        tf.compat.v1.logging.set_verbosity(args.verbosity)
        train_and_evaluate(args)


##########

%%bash
MODEL_DIR=output
gcloud ai-platform local train \
    --module-name trainer.task \
    --package-path trainer/ \
    --job-dir $MODEL_DIR \
    -- \
    --train-files $TRAIN_DATA \
    --eval-files $EVAL_DATA \
    --train-steps 1000 \
    --eval-steps 100


%%bash
ls output/keras_export/


from trainer import util
_, _, eval_x, eval_y = util.load_data()

prediction_input = eval_x.sample(5)
prediction_targets = eval_y[prediction_input.index]


print(prediction_input)


import json

with open('test.json', 'w') as json_file:
  for row in prediction_input.values.tolist():
    json.dump(row, json_file)
    json_file.write('\n')


%%bash
cat test.json


%%bash

gcloud ai-platform local predict \
    --model-dir output/keras_export/ \
    --json-instances ./test.json


%%bash
export PROJECT=$(gcloud config list project --format "value(core.project)")
echo "Your current GCP Project Name is: "${PROJECT}

PROJECT = "q1l1"  # Replace with your project name
BUCKET_NAME=PROJECT+"-aiplatform"
REGION="us-central1"

os.environ["PROJECT"] = PROJECT
os.environ["BUCKET_NAME"] = BUCKET_NAME
os.environ["REGION"] = REGION
os.environ["TFVERSION"] = "2.1"
os.environ["PYTHONVERSION"] = "3.7"

%%bash
if ! gsutil ls | grep -q gs://${BUCKET_NAME}; then
    gsutil mb -l ${REGION} gs://${BUCKET_NAME}
fi
gsutil cp -r data gs://$BUCKET_NAME/data

%%bash
export TRAIN_DATA=gs://$BUCKET_NAME/data/adult.data.csv
export EVAL_DATA=gs://$BUCKET_NAME/data/adult.test.csv

%%bash
gsutil cp test.json gs://$BUCKET_NAME/data/test.json

%%bash
export TEST_JSON=gs://$BUCKET_NAME/data/test.json


%%bash
JOB_ID=census_$(date -u +%y%m%d_%H%M%S)
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_ID
gcloud ai-platform jobs submit training $JOB_ID \
    --job-dir $OUTPUT_PATH \
    --runtime-version $TFVERSION \
    --python-version $PYTHONVERSION \
    --module-name trainer.task \
    --package-path trainer/ \
    --region $REGION \
    -- \
    --train-files $TRAIN_DATA \
    --eval-files $EVAL_DATA \
    --train-steps 1000 \
    --eval-steps 100 \
    --verbosity DEBUG

os.environ["JOB_ID"] = "census_210220_182019" # Replace with your job id

os.environ["MODEL_NAME"] = "census"

%%bash
gcloud ai-platform models create $MODEL_NAME --regions=$REGION

%%bash
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_ID
MODEL_BINARIES=$OUTPUT_PATH/keras_export/
gcloud ai-platform versions create v1 \
--model $MODEL_NAME \
--origin $MODEL_BINARIES \
--runtime-version $TFVERSION \
--python-version $PYTHONVERSION \
--region=global

%%bash
gcloud ai-platform models list

%%bash
gcloud ai-platform predict \
--model $MODEL_NAME \
--version v1 \
--json-instances ./test.json \
--region global