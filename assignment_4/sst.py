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
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stop_words.remove("not")
stop_words.remove("no")
stop_words.remove("nor")
stop_words.remove("now")
stop_words.remove("do")
stop_words.remove("does")


MAX_LENGTH = 29
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


class SSTDataset(Dataset):
    def __init__(self, sentence, label):
        self.sentence = sentence
        self.label = label

    def __getitem__(self, index):
        sentence = self.sentence[index]
        label = self.label[index]
        return torch.tensor(sentence), torch.tensor(label)

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


class Sentiment_Classifier(nn.Module):
    def __init__(self, embedding_size):
        super(Sentiment_Classifier, self).__init__()

        self.s1 = nn.Parameter(torch.ones(1))
        self.s2 = nn.Parameter(torch.ones(1))
        self.s3 = nn.Parameter(torch.ones(1))
        self.alpha = nn.Parameter(torch.ones(1))

        self.linear = nn.Linear(embedding_size, 2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, sentence):
        embeddings = elmo.embedding(sentence)
        out_1, _ = elmo.lstm1(embeddings)
        out_2, _ = elmo.lstm2(out_1)

        s_sum = self.s1 + self.s2 + self.s3

        output = self.alpha * (
            self.s1 / s_sum * embeddings
            + self.s2 / s_sum * out_1
            + self.s3 / s_sum * out_2
        ).to(torch.float32)

        output = self.linear(output)
        output = output.mean(dim=1)
        output = self.sigmoid(output)

        return output


dataset = load_dataset("sst")

print(f"Train Data: {len(dataset['train']['sentence'])}")
print(f"Validation Data: {len(dataset['validation']['sentence'])}")
print(f"Validation Data: {len(dataset['test']['sentence'])}")

if MODE == True:
    X_train = []
    y_train = []
    i = 0
    for data in dataset["train"]:
        X_train.append(preprocess_sentence(data["sentence"]))
        y_train.append(data["label"])

    X_val = []
    y_val = []
    i = 0
    for data in dataset["validation"]:
        X_val.append(preprocess_sentence(data["sentence"]))
        y_val.append(data["label"])

    X_test = []
    y_test = []
    for data in dataset["test"]:
        X_test.append(preprocess_sentence(data["sentence"]))
        y_test.append(data["label"])

    # Creating word_vocabulary
    word_train = [nltk.word_tokenize(x) for x in X_train]
    word_vocab = torchtext.vocab.build_vocab_from_iterator(word_train, min_freq=2)
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

    train_sequence = sent_to_sequence(X_train, word_vocab, MAX_LENGTH)
    val_sequence = sent_to_sequence(X_val, word_vocab, MAX_LENGTH)
    test_sequence = sent_to_sequence(X_test, word_vocab, MAX_LENGTH)

    train_dataset = SSTDataset(train_sequence, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    val_dataset = SSTDataset(val_sequence, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_dataset = SSTDataset(test_sequence, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    embedding_dict = {}

    file = open("./glove.6B/glove.6B.100d.txt", "r", encoding="utf-8")

    for line in file:
        line_list = line.split()
        word = line_list[0]
        embeddings = np.asarray(line_list[1:], dtype=float)

        embedding_dict[word] = embeddings

    file.close()

    embed_matrix = np.zeros((len(word_vocab), 100))

    for word, ind in word_vocab.get_stoi().items():
        embedding_vector = embedding_dict.get(word)

        if embedding_vector is not None:
            embed_matrix[ind] = embedding_vector

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_matrix_torch = torch.from_numpy(embed_matrix)
    embed_matrix_torch = embed_matrix_torch.to(device)
    elmo = (
        ELMo(len(word_vocab), 100, 50, 0.1, embed_matrix_torch, device)
        .double()
        .to(device)
    )

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(elmo.parameters(), lr=0.01)

    elmo.train()
    val_prev = np.inf
    for epoch in range(20):
        total_train_loss = 0
        total_loss = 0
        for sentence, label in tqdm(train_dataloader):
            inp = sentence[:, :-1].to(device)
            targ = sentence[:, 1:].to(device)
            optimizer.zero_grad()
            output = elmo(inp)
            loss = criterion(output, targ)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        print(total_train_loss / len(train_dataloader))

        with torch.no_grad():
            for sent, lab in tqdm(val_dataloader):
                input_val = sent[:, :-1].to(device)
                traget_val = sent[:, 1:].to(device)
                opt = elmo(input_val)

                loss_val = criterion(opt, traget_val)
                total_loss += loss_val.item()

        if val_prev < total_loss / len(val_dataloader):
            break
        else:
            val_prev = total_loss / len(val_dataloader)
        print(f"Validation Loss: {total_loss/len(val_dataloader)}")

    sentiment_model = Sentiment_Classifier(100).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(sentiment_model.parameters(), lr=0.01)

    for epoch in range(20):
        total_loss = 0
        total_loss_train = 0
        prev_val = np.inf
        for i, (sentence, label) in enumerate(tqdm(train_dataloader)):
            sentiment_model.train()
            sentence, label = sentence.to(device), label.to(device)

            optimizer.zero_grad()
            outputs = sentiment_model(sentence)

            label = (label >= 0.5).long()
            batch_loss = criterion(outputs, label)
            total_loss_train += batch_loss.item()

            batch_loss.backward()
            optimizer.step()

        print(f"Train Loss: {total_loss_train/len(train_dataloader)}")
        with torch.no_grad():
            sentiment_model.eval()
            for inputs, targets in tqdm(test_dataloader):
                model_input = inputs.to(device)
                targets = targets.to(device)
                out = sentiment_model(model_input)

                targets = (targets >= 0.5).long()
                loss = criterion(out, targets)
                total_loss += loss.item()

        if prev_val < total_loss:
            break
        else:
            prev_val = total_loss

        print(f"Validation Loss {total_loss/len(val_dataloader)}")

    torch.save(sentiment_model, "finetuned_elmo_sst.pth")

    with open("vocab_sst.pickle", "wb") as f:
        pickle.dump(word_vocab, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open("test_data_sst.pickle", "wb") as f:
        pickle.dump(test_dataloader, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("./models/sst/test_data_sst.pickle", "rb") as f:
    test_dataloader = pickle.load(f)

elmo = torch.load("./models/sst/pretrained_elmo_sst.pth", map_location="cpu")
sentiment_model = torch.load("./models/sst/finetuned_elmo_sst.pth", map_location="cpu")

from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from matplotlib import pyplot as plt

y_true = []
y_pred = []

sentiment_model.eval()

with torch.no_grad():
    for inputs, targets in tqdm(test_dataloader):
        model_input = inputs
        outputs = sentiment_model(model_input)

        targets = (targets > 0.5).long().tolist()

        y_true += targets
        y_pred += torch.argmax(outputs, dim=1).long().tolist()


print("Classification Report:")
print(classification_report(y_true, y_pred))

fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.show()
