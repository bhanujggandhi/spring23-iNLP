import re
import numpy as np
import random


# Tokenizer function returns clean data
def tokenize(text):
    text = re.sub(
        r'(Mr\.|Mrs\.|Ms\.)[a-zA-Z]*', '<TITLE>', text)
    text = re.sub(r'https?:\/\/\S+\b(?!\.)?', '<URL>', text)
    text = re.sub(r'@\w+', '<MENTION>', text)
    text = re.sub(r'#\w+', '<HASHTAG>', text)
    text = re.sub(r'\S*[\w\~\-]\@[\w\~\-]\S*', r'<EMAIL>', text)

    text = re.sub(r'([a-zA-Z]+)n[\'’]t', r'\1 not', text)
    text = re.sub(r'([iI])[\'’]m', r'\1 am', text)
    text = re.sub(r'([a-zA-Z]+)[\'’]s', r'\1 is', text)

    text = re.sub(r'\*{2,}.*?\*{2,}', '', text, flags=re.DOTALL)

    text = re.sub(r"_(.*?)_", r"\1", text)
    text = text.split()
    text = " ".join(text)
    text = "<SOS> " + text + " <EOS>\n"

    text = re.sub(r'[^\w\s<>]', ' ', text)
    text = text.lower()

    return text


# Open file
with open("Ulysses - James Joyce.txt", 'r') as fp:
    text = fp.readlines()

# ----------------------------
# build corpus
preprocessText = list()
for t in text:
    if t != "\n" or t != '':
        pt = tokenize(t)
        temp = pt.split()
        if len(temp) != 2:
            preprocessText.append(pt)

with open("clean-corpus.txt", "w+") as f:
    f.writelines(preprocessText)


# Build vocabulary
def vocabBuilder(texts):
    vocab = set()
    for txt in texts:
        vocab.update([re.sub(r'\s', '', ele) for ele in re.split(" ", txt)])
    return list(vocab)


vocab = vocabBuilder(preprocessText)

# Train test split


def train_test_split(text, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = range(len(text))
    test_indices = random.sample(population=indices, k=test_size)

    train_data, test_data = [], []
    for i in range(len(preprocessText)):
        if i in test_indices:
            test_data.append(preprocessText[i])
        else:
            train_data.append(preprocessText[i])

    return train_data, test_data


train_data, test_data = train_test_split(preprocessText, 1000)

print(len(test_data))
