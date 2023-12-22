# MESSENGER Magnetospheric Classification Using LSTM Neural Network

## Overview
This repository hosts Python scripts for supervised learning using an LSTM neural network to classify magnetospheric data from the MESSENGER spacecraft. The methodology and findings are detailed in ([Smith, Jackman et al. 2023](#)). Data used included
[MESSENGER magnetomster data](https://pds-ppi.igpp.ucla.edu/search/view/?id=pds://PPI/mess-mag-calibrated), and manually labeled magnetospheric crossings used for training are available in ([Sun et al. 2020](https://doi.org/10.1029/2019JA027490)).

## Model Training
To train the LSTM neural network model with the MESSENGER mission data, follow these steps:

1. **Data Preprocessing**
   - Run the preprocessing script located at `data_prep/messenger_preprocessing_pipeline.py`.
   - This script processes the MESSENGER CSVs and Sun labels for model training.
   - Be sure to include the relevant data files in the data directory.

2. **Model Training**
   - Use the RNN training script located at `ml_models/messenger_rnn.py`.
   - This script trains the LSTM neural network model with the preprocessed data.

3. **Result Visualisation**
   - Visualise the training results using the script `ml_models/visuals.py`.
   - This script aids in visualising the outcomes as included in the associated research paper.

## Using the Model for New Labels

## Getting Started
### Prerequisites
- Python 3.x
- Required Python packages (see `requirements.txt`)

### Installation
Clone the repository:
```bash
git clone https://github.com/ksmith9/messenger_ml_mag_classification.git

