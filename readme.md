This repository contains the code and data for the paper `Enhancing the Coverage of SemRep Using a Relation Classification Approach`.

# Instructions
This project is implemented with PyTorch. 
To run this code, we recommend Python version >= 3.9. To install the packages needed to run the code, run `pip install -r requirements.txt`.

## Data
There are three versions of data available for download from this repository, corresponding to the data used in the paper, and they are split into Train/Dev/Test.
`./EntityNameData` directory contains data where entities were formatted as entity names;
`./EntityTypeData` directory contains data where entities were formatted as semantic types;
`./EntityGroupData` directory contains data where entities were formatted as semantic groups;

## Run the code for training
Download the data and place all Python scripts in the same folder. 
To run the default configuration: `python main.py --do_train`. You can also choose to run the model with or without pretraining by changing the argument `do_pretrain` in the `main.py`.
You can specify your data for pretraining, training, and testing in the `main.py`. Place them in the same folder as your code.

## Run the code for test
`python predict.py`
