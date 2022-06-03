## Preparation

Extract data.zip to the same folder with scripts.

## Trainig

To run training with cuda:

` python train.py cuda `

To run training with cpu:

` python train.py `

During the run program outputs some information about current hyperparameters and losses, accuracy. \ 

Program will save the current best performing network and overwrite the previous one during all the run.\
As a result, after program finishes we have the best network saved in 'model_state_dict'  file.

## Testing

To run testing with cuda:

`python test.py cuda`

To run testing with cpu:

`python test.py`

Testing should be done only after training program completed! Because it needs to first load saved best network.

#### Additional scripts

* Sanity check:

`python sanity_check.py`

* Plot training and validation losses of best network (should run training first):

`python plot_best.py`
