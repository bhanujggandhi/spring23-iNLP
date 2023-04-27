import pickle
import re

import contractions
import nltk
import numpy as np
import torch
import torch.nn as nn
import torchtext
import unidecode
from bs4 import BeautifulSoup
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import (
    classification_report,
    roc_curve,
    roc_auc_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from matplotlib import pyplot as plt


# nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()

# nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stop_words.remove("not")
stop_words.remove("no")
stop_words.remove("nor")
stop_words.remove("now")
stop_words.remove("do")
stop_words.remove("does")


MAX_LENGTH = 42
BATCH_SIZE = 64
MODE = False


def preprocess_sentence(sent: str) -> str:
    # Remove HTML
    soup = BeautifulSoup(sent, "html.parser")
    sent = soup.get_text(separator=" ")

    # Remove whitespaces
    sent = sent.strip()
    sent = " ".join(sent.split())

    # Lowercase
    sent = sent.lower()

    # Remove accent characters
    sent = unidecode.unidecode(sent)

    # Expand the contractions
    sent = contractions.fix(sent)

    # Remove punctuations and non-ASCII characters
    sent = re.sub(r"[^\w\s]", "", sent)
    sent = re.sub(r"[^\x00-\x7f]", "", sent)

    # Remove stopwords and lemmatize
    sent = " ".join(
        [lemmatizer.lemmatize(word) for word in sent.split() if word not in stop_words]
    )

    return sent


class MNLIDataset(Dataset):
    def __init__(self, premise, hypothesis, label):
        self.premise = premise
        self.hypothesis = hypothesis
        self.label = label

    def __getitem__(self, index):
        premise = self.premise[index]
        hypothesis = self.hypothesis[index]
        label = self.label[index]
        return torch.tensor(premise), torch.tensor(hypothesis), torch.tensor(label)

    def __len__(self):
        return len(self.label)


class ELMo(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout, embeddings):
        super(ELMo, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings)

        self.layer_1 = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.layer_2 = nn.LSTM(
            input_size=2 * hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, X):
        embeddings = self.embedding(X)

        lstm1_output, _ = self.layer_1(embeddings)

        lstm2_output, _ = self.layer_2(lstm1_output)
        lstm2_output = self.dropout(lstm2_output)

        output = self.linear(lstm2_output)
        output = torch.transpose(output, 1, 2)
        return output


class Sentiment_Classifier_MultiNLI(nn.Module):
    def __init__(self, embedding_size):
        super(Sentiment_Classifier_MultiNLI, self).__init__()

        self.s1 = nn.Parameter(torch.ones(1))
        self.s2 = nn.Parameter(torch.ones(1))
        self.s3 = nn.Parameter(torch.ones(1))

        self.gamma = nn.Parameter(torch.ones(1))
        self.linear = nn.Linear(2 * embedding_size, 3)

    def forward(self, premise, hypothesis):
        embeddings_p = elmo.embedding(premise)
        lstm1_output_p, _ = elmo.lstm1(embeddings_p)
        lstm2_output_p, _ = elmo.lstm2(lstm1_output_p)
        output_p = self.gamma * (
            self.s1 * embeddings_p + self.s2 * lstm1_output_p + self.s3 * lstm2_output_p
        )
        out1 = output_p.mean(dim=1)

        embeddings_h = elmo.embedding(hypothesis)
        lstm1_output_h, _ = elmo.lstm1(embeddings_p)
        lstm2_output_h, _ = elmo.lstm2(lstm1_output_h)
        output_h = self.gamma * (
            self.s1 * embeddings_h + self.s2 * lstm1_output_h + self.s3 * lstm2_output_h
        )
        out2 = output_h.mean(dim=1)

        out = torch.cat([out1, out2], dim=-1).to(torch.float32)
        output = self.linear(out)
        return output


dataset = load_dataset("multi_nli")

print(f"Train Data: {len(dataset['train']['premise'])}")
print(f"Validation Data: {len(dataset['validation_matched']['premise'])}")

if MODE == True:
    X_train_sent1 = []
    X_train_sent2 = []
    y_train = []
    for data in dataset["train"]:
        if data["label"] == "-":
            continue
        X_train_sent1.append(preprocess_sentence(data["premise"]))
        X_train_sent2.append(preprocess_sentence(data["hypothesis"]))
        y_train.append(data["label"])

    X_test_sent1 = []
    X_test_sent2 = []
    y_test = []

    for data in dataset["validation_matched"]:
        if data["label"] == "-":
            continue
        X_test_sent1.append(preprocess_sentence(data["premise"]))
        X_test_sent2.append(preprocess_sentence(data["hypothesis"]))
        y_test.append(data["label"])

    # Creating word_vocabulary
    word_train_sent1 = [nltk.word_tokenize(x) for x in X_train_sent1]
    word_train_sent2 = [nltk.word_tokenize(x) for x in X_train_sent2]
    word_train_sent1.extend(word_train_sent2)
    word_vocab = torchtext.vocab.build_vocab_from_iterator(word_train_sent1, min_freq=5)
    word_vocab.insert_token("<unk>", 0)
    word_vocab.insert_token("<pad>", 1)
    word_vocab.set_default_index(word_vocab["<unk>"])

    print(f"Unique words: {len(word_vocab)}")

    def sent_to_sequence(dataset, word_vocab, max_length):
        text_sequence = []
        for data in dataset:
            curr_sequence = [word_vocab[word] for word in nltk.word_tokenize(data)]
            padding_seq = curr_sequence[:max_length] + [1] * max(
                0, max_length - len(curr_sequence)
            )
            text_sequence.append(padding_seq)

        return np.array(text_sequence)

    train_sequence_1 = sent_to_sequence(X_train_sent1, word_vocab, MAX_LENGTH)
    train_sequence_2 = sent_to_sequence(X_train_sent2, word_vocab, MAX_LENGTH)

    test_sequence_1 = sent_to_sequence(X_test_sent1, word_vocab, MAX_LENGTH)
    test_sequence_2 = sent_to_sequence(X_test_sent2, word_vocab, MAX_LENGTH)

    X_train1, X_val1, X_train2, X_val2, y_train, y_val = train_test_split(
        train_sequence_1, train_sequence_2, y_train, test_size=0.2, random_state=777
    )

    train_dataset = MNLIDataset(X_train1, X_train2, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    val_dataset = MNLIDataset(X_val1, X_val2, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_dataset = MNLIDataset(test_sequence_1, test_sequence_2, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    embedding_dict = {}

    file = open("./glove.6B/glove.6B.100d.txt", "r", encoding="utf-8")

    for line in file:
        line_list = line.split()
        word = line_list[0]
        embeddings = np.asarray(line_list[1:], dtype=float)

        embedding_dict[word] = embeddings

    file.close()

    embed_matrix = np.zeros((len(word_vocab), 200))

    for word, ind in word_vocab.get_stoi().items():
        embedding_vector = embedding_dict.get(word)

        if embedding_vector is not None:
            embed_matrix[ind] = embedding_vector

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_matrix_torch = torch.from_numpy(embed_matrix)
    embed_matrix_torch = embed_matrix_torch.to(device)
    elmo = (
        ELMo(len(word_vocab), 200, 100, 0.1, embed_matrix_torch, device)
        .double()
        .to(device)
    )

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(elmo.parameters(), lr=0.01)

    # Train the model
    elmo.train()

    for epoch in range(5):
        total_loss = 0
        for premise, hypothesis, label in tqdm(train_dataloader):

            optimizer.zero_grad()
            inp = premise[:, :-1].to(device)
            targ = premise[:, 1:].to(device)
            output = elmo(inp)
            loss = criterion(output, targ).to(device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            optimizer.zero_grad()
            inp = hypothesis[:, :-1].to(device)
            targ = hypothesis[:, 1:].to(device)
            output = elmo(inp)
            loss = criterion(output, targ).to(device)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Training Loss: {total_loss/len(train_dataloader)}")

    torch.save(elmo, "pretrained_elmo_nli.pth")

    model2 = Sentiment_Classifier_MultiNLI(200).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model2.parameters(), lr=0.01)

    for epoch in range(10):
        total_loss = 0
        for i, (premise, hypothesis, label) in enumerate(tqdm(train_dataloader)):
            model2.train()

            premise, hypothesis, label = (
                premise.to(device),
                hypothesis.to(device),
                label.to(device),
            )

            optimizer.zero_grad()
            outputs = model2(premise, hypothesis)

            batch_loss = criterion(outputs, label)
            total_loss += batch_loss.item()

            batch_loss.backward()
            optimizer.step()

        print(f"Validation Loss {total_loss/len(val_dataloader)}")

    torch.save(model2, "elmo_finetuned_mnli.pth")

    with open("vocab_mnli.pickle", "wb") as f:
        pickle.dump(word_vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open("test_data_mnli.pickle", "wb") as f:
        pickle.dump(test_dataloader, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("./models/multinli/test_data_mnli.pickle", "rb") as f:
    test_dataloader = pickle.load(f)

elmo = torch.load("./models/multinli/pretrained_elmo_nli.pth", map_location="cpu")
model2 = torch.load("./models/multinli/elmo_finetuned_mnli.pth", map_location="cpu")


y_true = []
y_pred = []

model2.eval()

with torch.no_grad():
    for premise, hypothesis, targets in tqdm(test_dataloader):
        outputs = model2(premise, hypothesis)

        y_true += targets.tolist()
        y_pred += torch.argmax(outputs, dim=1).long().tolist()


print("Classification Report:")
print(classification_report(y_true, y_pred))
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
