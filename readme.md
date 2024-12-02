To run these files:

1. Run the parse.R file. It will take a while to download, so don't stop it if it seems to hang. It will output files into ```/data``` (the .RData files), ```/series``` (the series for the 2020-2023 seasons that will be trained on), and ```/forecast``` (the data for the 2023 and 2024 seasons needed to predict the 2024 season).
1. Open the terminal and navigate to the ```/train``` directory.
1. To assess each model, run build.py in the terminal. Its interface is ```python build.py <model> <td>|<fg> <off>|<def>```. The options for ```<model>``` are:
    - ```lsr```: Least-Squares Regression
    - ```mpl```: Multi-Layer Perceptron
    - ```rnn```: Recurrent Neural Network
    - ```lstm```: Long Short-Term Memory
    - ```trans```: Transformer
1. To predict games for the 2024 NFL season, navigate to the ```/model``` directory.

These models were run with Python 3.12.4 and R 4.4.2.
