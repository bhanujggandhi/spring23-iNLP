# IMPORTS
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from conllu import parse
from sklearn.metrics import classification_report
from tqdm import tqdm

# For replicating the results
torch.manual_seed(1)

# mode to set on training or evalauting the saved model
mode = False

# Import the dataset files, I have used the english version of the dataset
with open("UD_English-Atis/en_atis-ud-train.conllu") as f:
    train_data = parse(f.read())
with open("UD_English-Atis/en_atis-ud-train.conllu") as f:
    dev_data = parse(f.read())
with open("UD_English-Atis/en_atis-ud-train.conllu") as f:
    test_data = parse(f.read())


# Utility to convert datasets to the sequences to feed into the model
def prepare_datasets(dataset):
    mod_data = []
    for idx in range(len(dataset)):
        tempword = []
        temptag = []
        for jdx in range(len(dataset[idx])):
            tempword.append(dataset[idx][jdx]["form"])
            temptag.append(dataset[idx][jdx]["upos"])

        mod_data.append([tempword, temptag])
    return mod_data


# Utitility to convert words to sequences
def tag_to_ix(tag, ix):
    try:
        return torch.tensor(ix[tag], dtype=torch.long)
    except KeyError:
        return torch.tensor(0, dtype=torch.long)


def sequence_to_idx(sequence, ix):
    ans = []
    for s in sequence:
        try:
            ans.append(ix[s])
        except KeyError:
            ans.append(0)
    return torch.tensor(ans, dtype=torch.long)


# Converting dataset to sequences
mod_train_data = prepare_datasets(train_data)
mod_test_data = prepare_datasets(test_data)
mod_dev_data = prepare_datasets(dev_data)

# Creating word_vocabulary
words_list = [sublist[0] for sublist in mod_train_data]
word_vocab = torchtext.vocab.build_vocab_from_iterator(words_list, min_freq=2)
word_vocab.insert_token("<unk>", 0)
word_vocab.set_default_index(word_vocab["<unk>"])

tags_list = [sublist[1] for sublist in mod_train_data]
tag_vocab = torchtext.vocab.build_vocab_from_iterator(tags_list)
tag_vocab.insert_token("<unk>", 0)
tag_vocab.set_default_index(tag_vocab["<unk>"])

print(f"Unique words: {len(word_vocab)}")
print(f"Unique tags: {len(tag_vocab)}")


# LSTM Model Declaration
class LSTMTagger(nn.Module):
    def __init__(self, word_embedding_dim, word_hidden_dim, vocab_size, tagset_size, dropout_fac):
        super(LSTMTagger, self).__init__()
        self.word_hidden_dim = word_hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, word_embedding_dim)
        self.lstm = nn.LSTM(word_embedding_dim, word_hidden_dim, num_layers=2, bidirectional=True)

        self.hidden2tag = nn.Linear(word_hidden_dim * 2, tagset_size)

        self.dropout = nn.Dropout(dropout_fac)

    def forward(self, sentence):
        embeds = self.dropout(self.word_embeddings(sentence))
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


# Setting up the right available device
device = torch.device("cpu")
# if(torch.backends.mps.is_available()):
#     device = torch.device("mps")
if torch.cuda.is_available():
    device = torch.device("cuda")

# Hyperparameters
WORD_EMBEDDING_DIM = 64
WORD_HIDDEN_DIM = 64
EPOCHS = 20
BIDIRECTIONAL = True
DROPOUT = 0.1

# Initialising the model
model = LSTMTagger(WORD_EMBEDDING_DIM, WORD_HIDDEN_DIM, len(word_vocab), len(tag_vocab), DROPOUT).to(device)

# Define the loss function as the Cross Entropy Loss
loss_function = nn.CrossEntropyLoss()

# We will be using an Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.005)

if mode == True:
    # Cold checking of the model before training
    with torch.no_grad():
        sent = "Mary had a little lamb".lower().split()
        sentence = torch.tensor(sequence_to_idx(sent, word_vocab), dtype=torch.long).to(device)

        tag_scores = model(sentence)
        _, indices = torch.max(tag_scores, 1)
        ret = []
        for i in range(len(indices)):
            for key, value in tag_vocab.get_stoi().items():
                if indices[i] == value:
                    ret.append((sent[i], key))
        print("*" * 5 + "Cold Starting the Model without training" + "*" * 5)
        print(ret)
        print("*" * 10)

    print("*" * 5 + "Training Started" + "*" * 5)
    train_accuracy_list = []
    train_loss_list = []
    valid_loss_list = []
    valid_accuracy_list = []
    max_valid_acc = -np.inf

    for epoch in range(EPOCHS):
        acc = 0
        loss = 0
        for sentence_tag, tag in tqdm(mod_train_data, desc="Training: ", leave=False):
            sentence = torch.tensor(sequence_to_idx(sentence_tag, word_vocab), dtype=torch.long).to(device)
            targets = torch.tensor(sequence_to_idx(tag, tag_vocab), dtype=torch.long).to(device)

            model.zero_grad()

            tag_scores = model(sentence)

            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
            loss += loss.item()
            _, indices = torch.max(tag_scores, 1)

            acc += torch.mean(torch.tensor(targets == indices, dtype=torch.float))

        loss = loss / len(mod_train_data)
        acc = acc / len(mod_train_data)
        train_loss_list.append(float(loss))
        train_accuracy_list.append(float(acc))

        valid_loss = 0
        valid_acc = 0
        for sentence, tag in mod_dev_data:
            sentence = torch.tensor(sequence_to_idx(sentence, word_vocab), dtype=torch.long).to(device)
            targets = torch.tensor(sequence_to_idx(tag, tag_vocab), dtype=torch.long).to(device)

            target = model(sentence)
            loss = loss_function(target, targets)
            valid_loss = loss.item() + valid_loss

            _, indices = torch.max(target, 1)

            valid_acc += torch.mean(torch.tensor(targets == indices, dtype=torch.float))

        print(f"Epoch {epoch+1} \t\t Training Loss: {loss} \t\t Training Acc: {acc}")
        print(f"Validation Loss: {valid_loss / len(mod_dev_data)} \t\t Validation Acc: {valid_acc/ len(mod_dev_data)}")
        valid_loss_list.append(float(valid_loss))
        valid_accuracy_list.append(float(valid_acc))

        if max_valid_acc < valid_acc:
            print(f"Validation Acc Increased({max_valid_acc:.6f}--->{valid_acc:.6f}) \t Saving The Model")
            max_valid_acc = valid_acc
            # Saving State Dict
            torch.save(model.state_dict(), "saved_model.pth")

    plt.plot(train_accuracy_list, c="red", label="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    plt.plot(train_loss_list, c="blue", label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    plt.plot(valid_loss_list, c="green", label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    plt.plot(valid_accuracy_list, c="yellow", label="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    from sklearn.metrics import accuracy_score

    with torch.no_grad():
        acc = []
        wrong = 0
        for sent, tags in mod_dev_data:
            sentence = torch.tensor(sequence_to_idx(sent, word_vocab), dtype=torch.long).to(device)

            tag_scores = model(sentence)
            _, indices = torch.max(tag_scores, 1)
            ret = []
            for i in range(len(indices)):
                for key, value in tag_vocab.get_stoi().items():
                    if indices[i] == value:
                        wrong += 1
                        ret.append(key)
            acc.append(accuracy_score(tags, ret))

        print(f"Accuracy on test dataset: {np.mean(acc)}")


model.load_state_dict(torch.load("last_model.pth", map_location=device))

# import json

# with open("model_state_most_acc/word_vocab.json", "r") as fp:
#     word_vocab = json.load(fp)

# with open("model_state_most_acc/tag_vocab.json", "r") as fp:
#     tag_vocab = json.load(fp)


def find_pos_tags(sent: str):
    with torch.no_grad():
        # Preprocessing
        sent = re.sub(r"[^\w\s]", "", sent).lower()
        sent = sent.split()

        sentence = torch.tensor(sequence_to_idx(sent, word_vocab), dtype=torch.long).to(device)

        tag_scores = model(sentence)
        _, indices = torch.max(tag_scores, 1)
        ret = []
        for i in range(len(indices)):
            for key, value in tag_vocab.get_stoi().items():
                if indices[i] == value:
                    ret.append(f"{sent[i]}\t{key}")

        for p in ret:
            print(p)


test_tag_list = [sublist[1] for sublist in mod_test_data]
flat_tags_test = [item for sublist in test_tag_list for item in sublist]

ret = []
with torch.no_grad():
    acc = []
    wrong = 0
    for sent, tags in mod_test_data:
        sentence = torch.tensor(sequence_to_idx(sent, word_vocab), dtype=torch.long).to(device)

        tag_scores = model(sentence)
        _, indices = torch.max(tag_scores, 1)
        for i in range(len(indices)):
            for key, value in tag_vocab.get_stoi().items():
                if indices[i] == value:
                    wrong += 1
                    ret.append(key)

print(classification_report(flat_tags_test, ret))


while True:
    sent = input("Please enter a sentence\n")
    if sent == "0":
        break
    find_pos_tags(sent)
