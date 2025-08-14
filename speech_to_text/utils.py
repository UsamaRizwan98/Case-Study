# import torch
# from torch.utils.data import DataLoader
# import torchaudio
# from torchaudio.datasets import SPEECHCOMMANDS
# from torchaudio.transforms import MelSpectrogram
# import os

# class FilteredSpeechCommands(SPEECHCOMMANDS):
#     def __init__(self, root, subset=None, allowed_labels=None, n_mels=64):
#         self.allowed_labels = allowed_labels
#         self.mel_spec = MelSpectrogram(sample_rate=16000, n_mels=n_mels)

#         dataset_path = os.path.join(root, "speechcommands")
#         download_flag = not os.path.exists(dataset_path)
#         super().__init__(root=dataset_path, download=download_flag, subset=subset)

#         if self.allowed_labels:
#             self._walker = [
#                 w for w in self._walker
#                 if os.path.basename(os.path.dirname(w)) in self.allowed_labels
#             ]

#     def __getitem__(self, n):
#         waveform, sample_rate, label, *_ = super().__getitem__(n)
#         mel = self.mel_spec(waveform).squeeze(0).transpose(0, 1)  # [time, n_mels]
#         return mel, label

# def collate_fn(batch, label_to_idx):
#     tensors, targets = [], []
#     for mel, label in batch:
#         tensors.append(mel)  # [time, n_mels]
#         targets.append(label_to_idx[label])

#     tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)  # [B, max_time, n_mels]
#     tensors = tensors.permute(0, 2, 1).unsqueeze(1)  # [B, 1, n_mels, max_time]
#     return tensors, torch.tensor(targets)

# def get_speechcommands_loaders(data_dir="./data", batch_size=8, allowed_labels=None, n_mels=64):
#     if allowed_labels is None:
#         allowed_labels = ["yes", "no", "up", "down"]

#     label_to_idx = {label: idx for idx, label in enumerate(allowed_labels)}

#     train_set = FilteredSpeechCommands(
#         root=data_dir, subset="training",
#         allowed_labels=allowed_labels, n_mels=n_mels
#     )
#     test_set = FilteredSpeechCommands(
#         root=data_dir, subset="testing",
#         allowed_labels=allowed_labels, n_mels=n_mels
#     )

#     train_loader = DataLoader(
#         train_set, batch_size=batch_size, shuffle=True,
#         collate_fn=lambda b: collate_fn(b, label_to_idx)
#     )
#     test_loader = DataLoader(
#         test_set, batch_size=batch_size, shuffle=False,
#         collate_fn=lambda b: collate_fn(b, label_to_idx)
#     )

#     return train_loader, test_loader, allowed_labels

import torch
from torch.utils.data import DataLoader
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MelSpectrogram, FrequencyMasking, TimeMasking
import os

class FilteredSpeechCommands(SPEECHCOMMANDS):
    def __init__(self, root, subset=None, allowed_labels=None, n_mels=64, augment=False):
        self.allowed_labels = allowed_labels
        self.mel_spec = MelSpectrogram(sample_rate=16000, n_mels=n_mels)

        # Add augmentation transforms if enabled
        self.augment = augment
        if self.augment:
            self.freq_mask = FrequencyMasking(freq_mask_param=15)
            self.time_mask = TimeMasking(time_mask_param=35)

        dataset_path = os.path.join(root, "speechcommands")
        download_flag = not os.path.exists(dataset_path)
        super().__init__(root=dataset_path, download=download_flag, subset=subset)

        if self.allowed_labels:
            self._walker = [
                w for w in self._walker
                if os.path.basename(os.path.dirname(w)) in self.allowed_labels
            ]

    def __getitem__(self, n):
        waveform, sample_rate, label, *_ = super().__getitem__(n)
        mel = self.mel_spec(waveform).squeeze(0).transpose(0, 1)  # [time, n_mels]

        # Apply augmentation only on training set
        if self.augment:
            mel = mel.transpose(0, 1)  # [n_mels, time]
            mel = self.freq_mask(mel)
            mel = self.time_mask(mel)
            mel = mel.transpose(0, 1)  # [time, n_mels]

        return mel, label

def collate_fn(batch, label_to_idx):
    tensors, targets = [], []
    for mel, label in batch:
        tensors.append(mel)  # [time, n_mels]
        targets.append(label_to_idx[label])

    tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)  # [B, max_time, n_mels]
    tensors = tensors.permute(0, 2, 1).unsqueeze(1)  # [B, 1, n_mels, max_time]
    return tensors, torch.tensor(targets)

def get_speechcommands_loaders(data_dir="./data", batch_size=8, allowed_labels=None, n_mels=64):
    if allowed_labels is None:
        allowed_labels = ["yes", "no", "up", "down"]

    label_to_idx = {label: idx for idx, label in enumerate(allowed_labels)}

    # Training set with augmentation
    train_set = FilteredSpeechCommands(
        root=data_dir, subset="training",
        allowed_labels=allowed_labels, n_mels=n_mels, augment=True
    )
    # Test set without augmentation
    test_set = FilteredSpeechCommands(
        root=data_dir, subset="testing",
        allowed_labels=allowed_labels, n_mels=n_mels, augment=False
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, label_to_idx)
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, label_to_idx)
    )

    return train_loader, test_loader, allowed_labels

