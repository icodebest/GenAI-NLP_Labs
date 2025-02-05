import torch
from torch.utils.data import Dataset, DataLoader
import spacy
from collections import Counter
import pickle
import os
import tarfile
import requests

# Download and Load SpaCy Tokenizers
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

# URLs for dataset
TRAIN_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-AI0205EN-SkillsNetwork/training.tar.gz"
VALID_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-AI0205EN-SkillsNetwork/validation.tar.gz"

# Special tokens
PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']

# Tokenizer functions
def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

# Function to download and extract data
def download_and_extract(url, extract_path):
    filename = url.split('/')[-1]
    file_path = os.path.join(extract_path, filename)
    
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)
    
    if not os.path.exists(file_path):
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded {filename}")
    
    # Extract the tar.gz file
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=extract_path)
        print(f"Extracted {filename}")

# Function to create vocabulary
def build_vocab(sentences, tokenizer):
    counter = Counter()
    for sentence in sentences:
        counter.update(tokenizer(sentence))
    vocab = {word: i + 4 for i, (word, _) in enumerate(counter.most_common())}  # Offset for special tokens
    for i, token in enumerate(special_tokens):
        vocab[token] = i  # Assign special tokens
    return vocab

# Custom Dataset Class
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src = [self.src_vocab.get(token, UNK_IDX) for token in tokenize_de(self.src_sentences[idx])]
        tgt = [self.tgt_vocab.get(token, UNK_IDX) for token in tokenize_en(self.tgt_sentences[idx])]
        return torch.tensor([BOS_IDX] + src + [EOS_IDX]), torch.tensor([BOS_IDX] + tgt + [EOS_IDX])

# Function to load dataset
def load_data(data_dir):
    src_sentences, tgt_sentences = [], []
    
    train_src_path = os.path.join(data_dir, "train.de")
    train_tgt_path = os.path.join(data_dir, "train.en")
    
    with open(train_src_path, 'r', encoding='utf-8') as src_file, open(train_tgt_path, 'r', encoding='utf-8') as tgt_file:
        for src_line, tgt_line in zip(src_file, tgt_file):
            src_sentences.append(src_line.strip())
            tgt_sentences.append(tgt_line.strip())
    
    src_vocab = build_vocab(src_sentences, tokenize_de)
    tgt_vocab = build_vocab(tgt_sentences, tokenize_en)
    dataset = TranslationDataset(src_sentences, tgt_sentences, src_vocab, tgt_vocab)
    return dataset, src_vocab, tgt_vocab

# Collate function for padding
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, tgt_batch

# Function to get DataLoaders
def get_dataloaders(batch_size=16, data_dir="data"):
    download_and_extract(TRAIN_URL, data_dir)
    dataset, src_vocab, tgt_vocab = load_data(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    return dataloader, src_vocab, tgt_vocab
