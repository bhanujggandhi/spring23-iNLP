import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import math

import sys

# Tokenizer function returns clean data
def clean_text(text):
    import re

    text = re.sub(r"(Mr\.|Mrs\.|Ms\.)[a-zA-Z]*", "<TITLE>", text)
    text = re.sub(r"https?:\/\/\S+\b(?!\.)?", "<URL>", text)
    text = re.sub(r"@\w+", "<MENTION>", text)
    text = re.sub(r"#\w+", "<HASHTAG>", text)
    text = re.sub(r"\S*[\w\~\-]\@[\w\~\-]\S*", r"<EMAIL>", text)

    text = re.sub(r"([a-zA-Z]+)n[\'’]t", r"\1 not", text)
    text = re.sub(r"([iI])[\'’]m", r"\1 am", text)
    text = re.sub(r"([a-zA-Z]+)[\'’]s", r"\1 is", text)

    text = re.sub(r"\*{2,}.*?\*{2,}", "", text, flags=re.DOTALL)

    text = re.sub(r"_(.*?)_", r"\1", text)
    text = re.sub(r"[.!?]+", ". ", text)
    text = re.sub(r"[^\w\s<>.]", " ", text)

    text = text.lower()
    text = text.split()
    text = " ".join(text)
    return text


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights):

        # Initialising the parent class
        super().__init__()

        # Setting up the hyperparameter
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Declaring embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Declaring LSTM Layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout_rate, batch_first=True)

        # Dropout Layer
        self.dropout = nn.Dropout(dropout_rate)

        # Linear Layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Tie Weights
        if tie_weights:
            if embedding_dim != hidden_dim:
                print("In order to tie weights, embedding dimension and hidden dimenstion should be same")
                sys.exit(1)
            self.embedding.weight = self.fc.weight

        # Initialising weights as generally pytorch initialises it randomly
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.weight.data.uniform_(-1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim))
        self.fc.bias.data.zero_()
        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(self.embedding_dim, self.hidden_dim).uniform_(
                -1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim)
            )
            self.lstm.all_weights[i][1] = torch.FloatTensor(self.hidden_dim, self.hidden_dim).uniform_(
                -1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim)
            )

    def forward(self, src, hidden):
        # Embedding tensor
        embedding = self.dropout(self.embedding(src))

        # LSTM Matrix
        output, hidden = self.lstm(embedding, hidden)
        output = self.dropout(output)

        # Linear output
        prediction = self.fc(output)
        return prediction, hidden

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell


def batchify(corpus, vocab, batch_size):
    tensor_data = list()
    for sent in corpus:
        tokens = sent.split()
        if tokens[-1][-1] == ".":
            tokens.append("<eos>")
        temptokens = []
        for word in tokens:
            try:
                if word != "<eos>" and word_count[word] >= 3:
                    temptokens.append(vocab[word])
                else:
                    temptokens.append(vocab["<unk>"])
            except KeyError:
                temptokens.append(vocab["<unk>"])
        tensor_data.extend(temptokens)

    # Convert 1D array to torch tensor
    tensor_data = torch.LongTensor(tensor_data)

    # No. of batches
    num_batches = tensor_data.shape[0] // batch_size

    # Drop extra values
    tensor_data = tensor_data[: num_batches * batch_size]

    # Convert to 2D tensor of size batch x num_batches
    tensor_data = tensor_data.view(batch_size, num_batches)
    return tensor_data


def create_sequence(tensor_data, seq_len, idx):
    X = tensor_data[:, idx : idx + seq_len]
    y = tensor_data[:, idx + 1 : idx + seq_len + 1]
    return X, y


def evaluate(model, data, criterion, batch_size, seq_len, device):
    epoch_loss = 0
    model.eval()
    num_batches = data.shape[-1]
    data = data[:, : num_batches - (num_batches - 1) % seq_len]
    num_batches = data.shape[-1]

    hidden = model.init_hidden(batch_size, device)

    # Disable gradient calculation while inference
    with torch.no_grad():
        for idx in range(0, num_batches - 1, seq_len):
            hidden = model.detach_hidden(hidden)
            src, target = create_sequence(data, seq_len, idx)
            src, target = src.to(device), target.to(device)
            batch_size = src.shape[0]

            prediction, hidden = model(src, hidden)
            prediction = prediction.reshape(batch_size * seq_len, -1)
            target = target.reshape(-1)

            loss = criterion(prediction, target)
            epoch_loss += loss.item() * seq_len
    return epoch_loss / num_batches


def calculate_perp(data, vocab_dict, model_file_path):
    data = clean_text(data)
    seq_len = len(data.split())
    data = data.split(". ")
    data = batchify(data, vocab_dict, 1)
    # print(data)
    model.load_state_dict(torch.load(model_file_path, map_location=device))
    test_loss = evaluate(model, data, criterion, 1, seq_len, device)
    pp = math.exp(test_loss)
    print(f"Perplexity: {math.exp(test_loss):.3f}")
    print(f"Probability: {(1 / pp) ** seq_len}")


vocab_file = input("Input vocabulary json file\n")
word_count_file = input("Input word_count json file\n")


import json

with open(vocab_file, "r") as f:
    vocab_dict = json.load(f)

with open(word_count_file, "r") as f:
    word_count = json.load(f)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
vocab_size = len(vocab_dict)
embedding_dim = 1150
hidden_dim = 1150
num_layers = 2
dropout_rate = 0.5
tie_weights = True
lr = 1e-3


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    model = LSTM(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    mode = False

    if mode == True:
        with open("Ulysses - James Joyce.txt", "r") as f:
            corpus = f.read()

        corpus = clean_text(corpus)
        text = []
        corpus = corpus.split(". ")
        for sent in corpus:
            split = sent.split(" ")
            split = [s for s in split if len(s) > 0]
            if len(split) > 0:
                sent = " ".join(split)
            text.append(sent)

        text_len = len(text)

        train_split = int(0.7 * text_len)
        dev_split = int(0.15 * text_len)

        train_data = text[:train_split]
        dev_data = text[train_split : train_split + dev_split]
        test_data = text[train_split + dev_split :]

        print("Saving Train Perplexity")
        train_perps = []
        for sent in train_data:
            pp = calculate_perp(sent, vocab_dict, sys.argv[1])
            train_perps.append(pp)

        with open("2022201068_LM6_train.txt", "w") as f:
            for i, sent in enumerate(train_data):
                f.write(f"{sent} \t {train_perps[i]}")

        print("Saving Test Perplexity")
        test_perps = []
        for sent in train_data:
            pp = calculate_perp(sent, vocab_dict, sys.argv[1])
            test_perps.append(pp)

        with open("2022201068_LM6_test.txt", "w") as f:
            for i, sent in enumerate(test_data):
                f.write(f"{sent} \t {test_perps[i]}")

    while True:
        print("*" * 40)
        sent = input("Enter the sentence\n")
        if sent == "0":
            break
        pp = calculate_perp(sent, vocab_dict, sys.argv[1])
        print("*" * 40)
