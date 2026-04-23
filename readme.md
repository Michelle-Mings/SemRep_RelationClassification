This repository contains the code and data for the paper `Enhancing the Coverage of SemRep Using a Relation Classification Approach`.

**Update:** The code has been updated for compatibility with recent versions of the `transformers` library and PyTorch (2.6+), including `safetensors` support and new PubMedBERT HuggingFace local folder loading.

# Instructions
This project is implemented with PyTorch. 
To install the required environment, run `conda env create -f semrep_env.yml`. 

## Data
There are three versions of data available for download from this repository, corresponding to the data used in the paper, and they are split into Train/Dev/Test.
`./EntityNameData` directory contains data where entities were formatted as entity names;

`./EntityTypeData` directory contains data where entities were formatted as semantic types;

`./EntityGroupData` directory contains data where entities were formatted as semantic groups.

## Run the code for training
Download the data and place all Python scripts in the same folder.

The `--model_name_or_path` argument specifies which BERT model architecture to use. It accepts either a HuggingFace model name (downloaded automatically) or a path to a locally downloaded model folder — useful if you prefer to download the model in advance rather than relying on automatic downloads. The default is `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`.

**Without pretraining** (fine-tune directly):
```
python main.py --do_train
python main.py --do_train --model_name_or_path "path/to/local/pubmedbert"
```

**With pretraining** (run contrastive pretraining first, then fine-tune):
```
python main.py --do_train --do_pretrain
python main.py --do_train --do_pretrain --model_name_or_path "path/to/local/pubmedbert"
```

You can specify your data for pretraining, training, and testing in `main.py`.

## Run the code for test

To replicate the results reported in the paper, first download the trained model from Google Drive:

#### [Download the trained model, test set results (reported in Table 3 in the paper), and large-scale assessment results from Google Drive](https://drive.google.com/drive/u/0/folders/1QpSVC9RX62uPmfv3r1Y1HSSf_PStU2Ie)

The downloaded folder contains two subfolders: `EntityType-model` and `EntityType-pretrainedmodel`. Point `--model_dir` to `EntityType-model`:
```
python predict.py --model_dir "path/to/EntityType-model" --output_file "your/output/file/path" --input_file EntityTypeData/test.tsv
```

## Evaluate the predictions

After running prediction, update the `file1` variable in `result_analysis.py` to point to your output file, then run:
```
python result_analysis.py
```
This reports micro-F1, macro-F1, precision, recall, and a per-label classification report.

