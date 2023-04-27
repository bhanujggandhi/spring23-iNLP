# Word2Vec

## Directory Structure

```
2022201068_assignment4
├── 2022201068_assignment4.pdf
├── models
│   ├── multinli
│   │   ├── elmo_finetuned_mnli.pth
│   │   ├── test_data_mnli.pickle
│   │   └── vocab_mnli.pickle
│   └── sst
│       ├── finetuned_elmo_sst.pth
│       ├── pretrained_elmo_sst.pth
│       ├── test_data_sst.pickle
│       └── vocab_sst.pickle
├── multinli.py
├── readme.md
└── sst.py
```

> **_NOTE:_** As the submission format requires no command line arguments, I have hardcoded the file names and the formats. Please follow the directory stucture in order for the code to work.

- `multinli.py`: This is the main python file in which multi nli elmo model code is provided. By default model will run in predict state. To enable train mode, you need to turn `MODE=True` inside the file.

- `sst.py`: This is the main python file in which sst elmo model code is prvovided. By default model will run in predict state. To enable train mode, you need to turn `MODE=True` inside the file.

> **_NOTE:_** Below folders and the files in them can be found at the `google-drive` link present at the end of the readme.

- `models`: This directory contains all the models. It has pretrained models according to the directiry sturcture.

- `svd_model` and `cbow_model`: Contains trained model as well as vocabulary stored in order for python scripts to predict

## Steps to execute

- Run the below commands

```sh
$ python sst.py
$ python multinli.py
```

[Link for the files](https://drive.google.com/drive/folders/1SOKwwFvbbDyKMlf1WFsUIY3-eevIaMua?usp=sharing)
