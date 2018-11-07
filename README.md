# NYC Taxi Challenge

This is a little exercise that I did to get familiar with ML and TensorFlow
again.

# Problem Statement

Given a taxi trip, determine if a tip will be paid and if a tip is paid, what
will be the expected tip amount.

# Details

The details of the entire journey is documented on my website at
http://jkschin.com/2018/11/02/nyc-taxi.html.

# Data

2017 GYC Green Taxi Data: https://data.cityofnewyork.us/Transportation/2017-Green-Taxi-Trip-Data/5gj9-2kzx

We used the entire 2017 dataset and sample 80% for the training data, and 20%
for the test data.

# Usage

## Initial Setup

1. Clone this repository. You can run a prediction as is once this is done.
   Refer to "how do I make a prediction" below for more information.
2. Download the dataset from the link above.
3. Put it in the `nyc_taxi` directory.
4. Rename it as `original.csv`
5. If you haven't already, use `pip` to install TensorFlow. You can refer to the
   TensorFlow website for more details.
   
P.S. I realized there were some package dependencies that were not included. I should have setup a `virtualenv` and done `pip freeze`.

## How do I generate the datasets?

1. `python clean_data.py`
2. Be sure the initial setup is done before you do this.

## How do I train the model again?

1. Open `nyc_taxi.py` and change `Line 154` to `Train = True`.
2. `python nyc_taxi.py`

## How do I make a prediction on the model?

**It isn't necessary to train a model first as I have included the models in
this repository. This should run as is.**

1. First note that this only works on a single example. It will not work as
   expected if the CSV has more than 2 rows (header and data). The fields
   `tip_amount` and `total_amount` have to be included, but won't be used. 
2. Change the data in `test_1.csv`, if necessary.
3. `python nyc_taxi.py`
4. The code should run as is if TensorFlow is already installed, and the sample
   output below will be shown.

## Sample Output

A 2.479710 tip is predicted
Since we have the answers, the actual tip that was given was 2.660000.

