# Neural POS Tagger

## Directory Structure

```
2022201068_assignment2
├── 2022201068_assignment2_report.pdf
├── README.md
├── UD_English-Atis
│   ├── en_atis-ud-dev.conllu
│   ├── en_atis-ud-test.conllu
│   └── en_atis-ud-train.conllu
├── pos_tagger.py
├── ptmode.pth
├── requirements.txt
├── tag_vocab.json
└── word_vocab.json
```

> **_NOTE:_** As the submission format requires no command line arguments, I have hardcoded the file names and the formats. Please follow the directory stucture in order for the code to work.

- `UD_English-Atis`: This directory contains all the dataset files which are is conllu format. This placement of the files must be in this format with exact same filename and directory name in order for the code to run.

- `pos_tagger.py`: This is the main python file which takes in the sentence and provides tags in the specified format. This file also loads the model `ptmode.pth` which is also hardcoded.

- `ptmode.pth`: This is the saved _pytorch model_ file, which is loaded in _pos_tagger.py_, make sure the file must be present in the same directory as mentioned with the same name.

- `tag_vocab.json` and `word_vocab.json`: These are the vocabulary JSON files. I have fixed the random state of the model to replicate the results, but if we change that state, we need to load vocabulary also to the _pos_tagger.py_ in which loading JSON code is commented.

## Steps to execute

- Run the below commands

```sh
$ pip install requirements.txt
$ python pos_tagger.py
```

- The output will be the **_Classfication Report_** of the test dataset and an infinite loop to enter the new sentences and get the output as POS Tags.
- Press 0 and enter to exit the program.
