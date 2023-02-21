#!/usr/bin/env python
# coding: utf-8

# Library Imports

import numpy as np
import progressbar
import random
import re
from nltk import sent_tokenize
import sys


# corpus = ["<sos> You are my friend <eos>", "<sos> They are my enemies <eos>", "<sos> I have friends and enemies <eos>"]
# Input format

modeldesc = {
    "LM1": {"filename": "Pride and Prejudice - Jane Austen.txt", "smtype": "k"},
    "LM2": {"filename": "Pride and Prejudice - Jane Austen.txt", "smtype": "w"},
    "LM3": {"filename": "Ulysses - James Joyce.txt", "smtype": "k"},
    "LM4": {"filename": "Ulysses - James Joyce.txt", "smtype": "w"},
}

# if len(sys.argv) != 3:
#     print("Command format is not defined")
#     sys.exit(1)
modelmode = (modeldesc.get(sys.argv[1]) and sys.argv[1][:-1] == "LM") == True

if modelmode:
    corpusfilepath = modeldesc[str(sys.argv[1])]["filename"]
    smtype = modeldesc[str(sys.argv[1])]["smtype"]
else:
    try:
        corpusfilepath = sys.argv[1]
        smtype = sys.argv[2]
    except IndexError:
        print("Command format is not defined")
        sys.exit(1)

    if smtype != "k" and smtype != "w":
        print("Smoothing type is not defined")
        sys.exit(1)


try:
    with open(corpusfilepath, "r") as f:
        text = f.read()
except FileNotFoundError:
    print("Corpus file not found!")
    sys.exit(1)


def clean_text(text: str) -> str:
    """Function to clean text, removes titles, urls, mentions, hashtags, etc.
    Args:
        text (str): Takes input a text sentence

    Returns:
        str: Clean text
    """

    text = re.sub(r"(Mr\.|Mrs\.|Ms\.)[a-zA-Z]*", "<TITLE>", text)
    text = re.sub(r"https?:\/\/\S+\b(?!\.)?", "<URL>", text)
    text = re.sub(r"@\w+", "<MENTION>", text)
    text = re.sub(r"#\w+", "<HASHTAG>", text)
    text = re.sub(r"\S*[\w\~\-]\@[\w\~\-]\S*", r"<EMAIL>", text)

    # Handling '[s, nt, m]
    text = re.sub(r"([a-zA-Z]+)n[\'’]t", r"\1 not", text)
    text = re.sub(r"([iI])[\'’]m", r"\1 am", text)
    text = re.sub(r"([a-zA-Z]+)[\'’]s", r"\1 is", text)

    # Handling footnotes
    text = re.sub(r"\*{2,}.*?\*{2,}", "", text, flags=re.DOTALL)

    # Handling markdown format
    text = re.sub(r"_(.*?)_", r"\1", text)

    # text = text.split()
    # text = " ".join(text)

    # Remove punctuation except for ., <, >
    text = re.sub(r"[^\w\s<>.]", " ", text)

    # Converting whole text to lowercase
    text = text.lower()

    text = "<s> " + text + " <e>\n"

    return text


text = sent_tokenize(text)

# text = clean_text(text)
# text = text.split(". ")
preprocessText = list()
for t in text:
    t = clean_text(t)
    t = t.split()

    if len(t) <= 4:
        continue

    t = " ".join(t)
    preprocessText.append(t)

corpus = preprocessText

# with open("clean-corpus.txt", "w+") as f:
#     f.writelines(corpus)
# print(corpus)


def train_test_split(text: list[str], test_size) -> (list[str], list[str]):
    """Method to split the list of sentence into train and test part absed on test size provided.

    Args:
        text (list[str]): List of corpus sentences
        test_size (float or int): Fraction or number of sentence to be places in train array

    Returns:
        tuple(list[str], list[str]): Pair of train array and test array
    """

    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = range(len(text))
    test_indices = random.sample(population=indices, k=test_size)

    train_data, test_data = [], []
    for i in range(len(text)):
        if i in test_indices:
            test_data.append(text[i])
        else:
            train_data.append(text[i])

    return train_data, test_data


train_data, test_data = train_test_split(preprocessText, 1000)

print("*" * 40)
print(f"Training Corpus Size: {len(train_data)}")
print(f"Testing Corpus Size: {len(test_data)}")
print("*" * 40)


# Global Dictionary for Caching Values
forward_n_gram = dict()
backward_n_gram = dict()
n_gram_containing_substring = dict()


def create_fb_dictionary(corpus: list[str]) -> None:
    for sentence in corpus:
        # Appending <sos> to cover all the context
        sent = f"<s> <s> {sentence}"
        sent = sent.split()

        # Storing forward, backward dictionary
        for k in range(2, 5):
            start = 0
            end = k
            while end <= len(sent):
                for_tup = tuple(sent[start : start + (k - 1)])
                back_tup = tuple(sent[start + 1 : start + (k)])
                if forward_n_gram.get(for_tup) is None:
                    forward_n_gram[for_tup] = dict()
                if backward_n_gram.get(back_tup) is None:
                    backward_n_gram[back_tup] = dict()
                if forward_n_gram[for_tup].get(sent[start + (k - 1)]) is None:
                    forward_n_gram[for_tup][sent[start + (k - 1)]] = 1
                else:
                    forward_n_gram[for_tup][sent[start + (k - 1)]] += 1
                if backward_n_gram[back_tup].get(sent[start]) is None:
                    backward_n_gram[back_tup][sent[start]] = 1
                else:
                    backward_n_gram[back_tup][sent[start]] += 1
                start += 1
                end += 1


def create_mid_dictionary(corpus: list[str]) -> None:
    for sentence in corpus:
        sent = f"<s> <s> {sentence}"
        sent = sent.split()

        start = 0
        end = 4

        while end <= len(sent):
            tup = tuple([sent[start + 1], sent[start + 2]])
            concatkey = f"{sent[start]}{sent[start+3]}"
            if n_gram_containing_substring.get(tup) is None:
                n_gram_containing_substring[tup] = dict()

            if n_gram_containing_substring[tup].get(concatkey) is None:
                n_gram_containing_substring[tup][concatkey] = 1
            else:
                n_gram_containing_substring[tup][concatkey] += 1

            start += 1
            end += 1

        start = 0
        end = 3

        while end <= len(sent):
            tup = tuple([sent[start + 1]])
            concatkey = f"{sent[start]}{sent[start+2]}"
            if n_gram_containing_substring.get(tup) is None:
                n_gram_containing_substring[tup] = dict()

            if n_gram_containing_substring[tup].get(concatkey) is None:
                n_gram_containing_substring[tup][concatkey] = 1
            else:
                n_gram_containing_substring[tup][concatkey] += 1

            start += 1
            end += 1


create_fb_dictionary(train_data)
create_mid_dictionary(train_data)


def get_unigram_frequency(corpus):
    unigram_frequency = dict()
    for sentence in corpus:
        sentence = f"<s> <s> {sentence}"
        tokens = sentence.split()
        for token in tokens:
            if unigram_frequency.get(token) is None:
                unigram_frequency[token] = 1
            else:
                unigram_frequency[token] += 1
    return unigram_frequency


unigram_dict = get_unigram_frequency(train_data)


def num_of_unique_bigrams():
    kys = list(forward_n_gram.keys())
    n = 0
    for k in kys:
        if len(k) == 1:
            n += len(forward_n_gram[k])

    return n


unique_bi = num_of_unique_bigrams()


def good_turing_estimation():
    n = 0
    total = 0

    vals = unigram_dict.values()

    for i in vals:
        if i == 1:
            n += 1
            total += 1
        else:
            total += i

    return n / total


unituringest = good_turing_estimation()


# def good_turing(n_gram):
#     n = 0
#     total = 0
#     for key in forward_n_gram:
#         if len(key) == n_gram - 1:
#             for ele in forward_n_gram[key]:
#                 if forward_n_gram[key][ele] == 1:
#                     n += 1
#                     total += 1
#                 else:
#                     total += forward_n_gram[key][ele]

#     return n / total


# turingest4 = {4: good_turing(4), 3: good_turing(3), 2: good_turing(2)}

# Utility functions for kneser ney smoothing algorithm


def basecase_numerator(n_gram: tuple):
    try:
        return len(backward_n_gram[n_gram])
    except KeyError:
        return -1


def highestorder_numerator1(n_gram: tuple):
    try:
        return max(forward_n_gram[tuple(n_gram[:-1])][n_gram[-1]] - 0.75, 0)
    except KeyError:
        return -1


def highestorder_denominator(n_gram: tuple):
    try:
        return forward_n_gram[tuple(n_gram[:-2])][n_gram[-2]]
    except KeyError:
        return 0


def highestorder_numerator2(n_gram: tuple):
    try:
        return len(forward_n_gram[tuple(n_gram[:-1])]) * 0.75
    except KeyError:
        return -1


def lowerorder_numerator1(n_gram: tuple):
    try:
        return max(len(backward_n_gram[n_gram]) - 0.75, 0)
    except KeyError:
        return -1


def lowerorder_denominator(n_gram: tuple):
    try:
        return len(n_gram_containing_substring[tuple(n_gram[:-1])])
    except KeyError:
        return 0


def lowerorder_numerator2(n_gram: tuple):
    try:
        return len(forward_n_gram[tuple(n_gram[:-1])]) * 0.75
    except KeyError:
        return -1


def kneser_ney_smoothing(n_gram: tuple, n: int) -> float:
    # Base Case
    if n == 1:
        numerator = basecase_numerator(n_gram)
        if numerator == -1:
            return 0.01 * unituringest
        deno = unique_bi
        return numerator / deno

    # Highest Order Case
    if n == 4:
        num1 = highestorder_numerator1(n_gram)
        deno1 = highestorder_denominator(n_gram)
        num2 = highestorder_numerator2(n_gram)
        deno2 = deno1

        # Handling when n_gram does not exist
        # if deno1 == 0 and num1 == -1 and num2 == -1:
        #     return turingest4[n]
        if deno1 == 0 or num1 == -1 or num2 == -1:
            return kneser_ney_smoothing(tuple(n_gram[1:]), n - 1)
            # return turingest
        return (num1 / deno1) + (num2 / deno2) * kneser_ney_smoothing(tuple(n_gram[1:]), n - 1)

    num1 = lowerorder_numerator1(n_gram)
    deno1 = lowerorder_denominator(n_gram)
    num2 = lowerorder_numerator2(n_gram)
    deno2 = deno1

    # Handling when n_gram does not exist
    # if deno1 == 0 and num1 == -1 and num2 == -1:
    #     return turingest4[n]
    if deno1 == 0 or num1 == -1 or num2 == -1:
        return kneser_ney_smoothing(tuple(n_gram[1:]), n - 1)
        # return turingest

    return (num1 / deno1) + (num2 / deno2) * kneser_ney_smoothing(tuple(n_gram[1:]), n - 1)


# Utility functions for witten bell smoothing
def no_of_n_grams(n_gram: tuple):
    # If n-gram is not found
    try:
        return forward_n_gram[tuple(n_gram[:-1])][n_gram[-1]]
    except KeyError:
        return 0


def all_unigrams():
    return np.sum(list(unigram_dict.values()))


all_unigram_counts = all_unigrams()


def witten_bell_smoothing(n_gram: tuple, n: int) -> float:
    # Base Case
    if n == 1:
        try:
            return unigram_dict[n_gram[0]] / all_unigram_counts
        except KeyError:
            return 0.01 * unituringest

    # count_n_gram = no_of_n_grams(n_gram)
    try:
        count_n_gram = forward_n_gram[tuple(n_gram[:-1])][n_gram[-1]]
    except KeyError:
        return witten_bell_smoothing(n_gram[1:], n - 1)
    # If n-1 gram is also not found then backoff
    try:
        unique_prefix_grams = len(forward_n_gram[tuple(n_gram[:-1])])
    except KeyError:
        return witten_bell_smoothing(n_gram[1:], n - 1)
        # return turingest

    count_all_n_gram = np.sum(list(forward_n_gram[tuple(n_gram[:-1])].values()))

    return (count_n_gram + unique_prefix_grams * witten_bell_smoothing(n_gram[1:], n - 1)) / (
        count_all_n_gram + unique_prefix_grams
    )


def sentence_perplexity(sentence: str, smoothing_type: str) -> float:
    if sentence.startswith("<s>"):
        sentence = f"<s> <s> {sentence}"
    else:
        sentence = f"<s> <s> <s> {sentence} <e>"
    sentence = sentence.split()
    # print(sentence)

    if len(sentence) >= 8000:
        return 0

    probabs = list()

    start = 0
    end = 4

    while end <= len(sentence):
        if smoothing_type == "k":
            c = kneser_ney_smoothing(tuple(sentence[start:end]), 4)
        else:
            c = witten_bell_smoothing(tuple(sentence[start:end]), 4)

        probabs.append((1 / c) ** (1 / (len(sentence))))
        start += 1
        end += 1
    return np.prod(probabs)


widgets = [
    " [",
    progressbar.Timer(),
    "] ",
    progressbar.Bar("="),
    progressbar.Percentage(),
    " (",
    progressbar.ETA(),
    ") ",
]


def save_perplexities(corpus, perplexities, mode):
    with open(f"2022201068_{sys.argv[1]}_{mode}.txt", "w+") as f:
        f.write(f"Average Perplexity: {np.mean(perplexities)}\n")
        for i, sentence in enumerate(corpus):
            f.write(f"{sentence}\t{perplexities[i]}\n")


def get_perplexity(corpus: list[str], smoothing_type: str, mode: str) -> float:
    perplexities = list()
    bar = progressbar.ProgressBar(max_value=len(corpus), widgets=widgets).start()
    for i, sentence in enumerate(corpus):
        perp = sentence_perplexity(sentence, smoothing_type)
        perplexities.append(perp)
        bar.update(i)

    if modelmode:
        save_perplexities(corpus, perplexities, mode)

    return np.mean(perplexities)


print("TRAINING CORPUS")
training_avg_perplexity = get_perplexity(train_data, smtype, "train")
print(f"\nAverage Training Perplexity: {training_avg_perplexity}")

print("*" * 40)
print("TESTING CORPUS")
testing_avg_perplexity = get_perplexity(test_data, smtype, "test")
print(f"\n Average Testing Perplexity: {testing_avg_perplexity}")
print("*" * 40)

if not modelmode:
    while True:
        inp = input("Enter a sentence\n")

        if inp == "0":
            break

        inp = clean_text(inp)

        print(sentence_perplexity(inp, smtype), "\n")
