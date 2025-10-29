import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

class Flickr8kDatasetComposer:
    """
    Unified Flickr8k dataset loader.
    Reads Flickr8k.token.txt once, generates vocabulary on train captions
    and provides .get_subset("train"/"val"/"test") subsets based on split files.
    """

    def __init__(self, root_dir="data", max_len=30, vocab_threshold=5):
        self.root_dir = root_dir
        self.max_len = max_len

        # Load all captions
        captions_path = os.path.join(root_dir, "Flickr8k.token.txt")
        self.full_df = self._load_captions(captions_path)

        # Load image splits
        self.splits = {
            "train": self._read_split_file(os.path.join(root_dir, "Flickr_8k.trainImages.txt")),
            "val": self._read_split_file(os.path.join(root_dir, "Flickr_8k.devImages.txt")),
            "test": self._read_split_file(os.path.join(root_dir, "Flickr_8k.testImages.txt")),
        }

        # Build vocabulary from train captions only
        train_captions = self.full_df[self.full_df["image"].isin(self.splits["train"])]["caption"].tolist()
        self.vocab = Vocabulary()
        self.vocab.build(train_captions, threshold=vocab_threshold)

    def _load_captions(self, path):
        """Reads Flickr8k.token.txt → returns DataFrame [image, caption_number, caption]."""
        records = []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                img_id, caption = parts
                img_name, cap_num = img_id.split("#")
                records.append((img_name, int(cap_num), caption.lower()))
        return pd.DataFrame(records, columns=["image", "caption_number", "caption"])

    def _read_split_file(self, path):
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def get_subset(self, split, transform=None):
        """
        Returns a self-sufficient Flickr8kSubset dataset instance for a given split ("train", "val", or "test").
        """
        if split not in self.splits:
            raise ValueError("split must be one of: 'train', 'val', 'test'")
        split_df = self.full_df[self.full_df["image"].isin(self.splits[split])].reset_index(drop=True)

        return Flickr8kSubset(split_df, self.vocab, self.root_dir, self.max_len, transform)


class Flickr8kSubset(Dataset):
    """
    A sub-dataset of the Flickr8k dataset (train/val/test) that returns (image, tokenized_caption_tensor).
    """

    def __init__(self, df, vocab, root_dir, max_len, transform):
        self.df = df
        self.vocab = vocab
        self.max_len = max_len
        self.transform = transform
        self.image_dir = os.path.join(root_dir, "images")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row["image"]

        ref_captions = self.df[self.df["image"] == image_name]["caption"].tolist()
        #ref_captions_tokenized = []
        #for c in ref_captions:
        #    c_tokens = c.split()
        #    ref_captions_tokenized.append(c_tokens)



        img_path = os.path.join(self.image_dir, image_name)
        image = F.to_tensor(Image.open(img_path).convert("RGB"))

        if self.transform is not None:
            image = self.transform(image)

        tokens = self.vocab.tokenize(row["caption"])
        caption_tensor = torch.tensor(self.vocab.pad(tokens, self.max_len))
        return image, caption_tensor, ref_captions

def flickr_collate_fn(batch):
    """
    Custom collate function for Flickr8k DataLoader.
    Args:
        batch: list of tuples (image_tensor, caption_tensor, ref_captions)
    Returns:
        images: Tensor [batch_size, 3, H, W]
        captions: Tensor [batch_size, max_seq_len]
        ref_captions: list of shape (batch_size, num_refs)
    """
    # Unzip batch
    image_tensors, caption_tensors, ref_captions = zip(*batch)

    #as default_collate
    images = torch.stack(image_tensors, dim=0)
    captions = torch.stack(caption_tensors, dim=0)

    #keep refs as list of lists
    ref_captions = list(ref_captions)

    return images, captions, ref_captions



class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>"}
        self.word_count = {}

    def build(self, captions, threshold=5):
        for cap in captions:
            for word in cap.lower().split():
                self.word_count[word] = self.word_count.get(word, 0) + 1
        for word, count in self.word_count.items():
            if count >= threshold and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def tokenize(self, sentence):
        tokens = [self.word2idx.get(word, 0) for word in sentence.split()]
        return [self.word2idx["<SOS>"]] + tokens + [self.word2idx["<EOS>"]]

    def pad(self, tokens, max_len):
        tokens = tokens[:max_len]
        return tokens + [self.word2idx["<PAD>"]] * (max_len - len(tokens))

    def detokenize(self, tokenized_pred_batch):
        pad_idx = self.word2idx["<PAD>"]
        eos_idx = self.word2idx["<EOS>"]
        sos_idx = self.word2idx["<SOS>"]

        pred_sentences_detokenized = []
        for pred in tokenized_pred_batch:
            # Cut off at EOS
            if eos_idx in pred:
                pred = pred[:torch.where(pred == eos_idx)[0][0]]

            # Detokenize
            pred_words_list = [
                self.idx2word[idx.item()]
                for idx in pred
                if idx.item() not in [pad_idx, eos_idx, sos_idx]
            ]

            if len(pred_words_list) > 0:
                pred_sentences_detokenized.append(" ".join(pred_words_list))

        return pred_sentences_detokenized
