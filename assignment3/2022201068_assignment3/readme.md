# Word2Vec

## Directory Structure

```
2022201068_assignment3
├── cbow.py
├── cbow_model
│   ├── model_cbow.pt
│   └── vocab_cbow.pt
├── co-occurrence.py
├── datasubsets
│   ├── subdata_cbow.json
│   └── subdata_svd.json
├── readme.md
└── svd_model
    ├── ipca_matrix.npy
    └── vocab.pkl
```

> **_NOTE:_** As the submission format requires no command line arguments, I have hardcoded the file names and the formats. Please follow the directory stucture in order for the code to work.

- `cbow.py`: This is the main python file in which CBOW neural model code is prvovided. By default model will run in predict state. To enable train mode, you need to turn `TRAIN=True` inside the file. When the file runs, it will run for 5 words stored in the list, we can change them in order to predict more.

- `co-occurrence.py`: This is the main python file in which Co-occurrence model code is prvovided. By default model will run in predict state. To enable train mode, you need to turn `TRAIN=True` inside the file. When the file runs, it will run for 5 words stored in the list, we can change them in order to predict more.

> **_NOTE:_** Below folders and the files in them can be found at the `google-drive` link present at the end of the readme.

- `datasubsets`: This directory contains all the dataset files which are a subset of original dataset provided which had huge number of sentences. I have takes 100000 amazon reviews to work with.

- `svd_model` and `cbow_model`: Contains trained model as well as vocabulary stored in order for python scripts to predict

## Steps to execute

- Run the below commands

```sh
$ python cbow.py
$ python co-occurrence.py
```

- The output will be the **PCA plots** of the word list.
- Terminal will show top 10 nearest words based on the cosine similarity

[Link for the files](https://drive.google.com/drive/folders/1of9JnhbB_gbhyuoo3LROudhWc_5v6kgg?usp=share_link)
