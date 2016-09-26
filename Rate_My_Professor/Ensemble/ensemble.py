# Ensemble Learning - MDST RateMyProfessors Challenge

import numpy as np
import pandas as pd

# INSTRUCTIONS: Replace "files" and "scores" with a list of submission
# files and their corresponding leaderboard scores. Ideally, choose
# submissions that are very different from eachother. You won't see
# any benefit if you ensemble a few regression models with slightly
# different hyperparameters.
files = ['16065.csv', 'LSTM_single_best_14756.csv', 'average_6_single_layer_NN.csv']
scores = [1.42982, 1.42697, 1.47473]

# Save location - feel free to change this
outfile = 'submit_ensemble.csv'

# RMSE of the all zero benchmark - we'll use this later
eps_0 = 7.95683

# Read the submission files
n = 69356
models = np.zeros((n, len(scores)))
for i in range(3):
    models[:, i] = pd.read_csv(files[i],usecols=['quality'],header=0)['quality'].get_values()

# Optimize weights
scores = np.array(scores)
YtX = (np.sum(np.power(models, 2), 0) + n*np.power(eps_0, 2) -
       n*np.power(scores, 2))/2;
weights = np.linalg.solve(np.dot(models.transpose(), models), YtX)
print "Weights:", weights

# Make predictions and trim
predictions = np.dot(models, weights.transpose())
predictions[predictions > 10] = 10
predictions[predictions < 2] = 2

# Save the new submission file
table = pd.read_csv(files[1])
table.quality = predictions
table.to_csv(outfile, index=False)
