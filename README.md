# Exploring truthfulness direction via SAEs

In this preject we use the Inference-Time Intervention technique to find truthfulness direction in the space of SAE activatetions of a model. We try to interpret it with the infrastructure available for SAE interpretation.

## What files are here

There are the following scripts to run inference-time intervention on Llama-2-7b and Gemma-2-2b:
- get_accuracies_*model*: scripts for saving probing accuracy for all the heads in the model.
- get_steering_direction_*model*: scripts for calculating the streering directions for all the heads/layers of the model.
- hyperparameters_sweep_*model*: scripts for searching for the best hyperparameters of intervention. Require two previous scripts to be run first.
- test_*model*: scripts for testing models with the hyperparemeters hardcoded in the script (need to be changed to the ones found in step before) on the testing dataset.

Also the repository has the following notebooks:
- gemma_sae.ipynb: for playing with SAE activations and vectors, and plotting something about them.
- plot_stuff.ipynb: for plotting other graphs.

Feel free to use this code in any (unharmful) way you like if you find it useful.