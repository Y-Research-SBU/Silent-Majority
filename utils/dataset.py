import os, torch, sys
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer, AutoTokenizer, DistilBertTokenizer, GPT2Tokenizer


class ShortcutLearningDataset(Dataset):
    def __init__(self, basedir, split="train", reweight=False, transform=None, device='cuda'):
        try:
            split_i = ["train", "val", "test"].index(split)
        except ValueError:
            raise(f"Unknown split {split}")
        metadata_df = pd.read_csv(os.path.join(basedir, "metadata.csv"))
        print(f'Total samples #:{len(metadata_df)}')
        self.metadata_df = metadata_df[metadata_df["split"] == split_i]
        print(f'Split samples #:{len(self.metadata_df)}')
        self.basedir = basedir
        self.transform = transform
        self.y_array = self.metadata_df['y'].values
        self.p_array = self.metadata_df['place'].values
        self.n_classes = np.unique(self.y_array).size
        self.confounder_array = self.metadata_df['place'].values
        self.n_places = np.unique(self.confounder_array).size
        self.group_array = (self.y_array * self.n_places + self.confounder_array).astype('int')
        self.n_groups = self.n_classes * self.n_places
        self.group_counts = (
                torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array)).sum(1).float()
        self.y_counts = (
                torch.arange(self.n_classes).unsqueeze(1) == torch.from_numpy(self.y_array)).sum(1).float()
        self.p_counts = (
                torch.arange(self.n_places).unsqueeze(1) == torch.from_numpy(self.p_array)).sum(1).float()
        self.filename_array = self.metadata_df['img_filename'].values
        if reweight:
            self.weights = len(self.metadata_df)/self.group_counts[self.group_array]
        else:
            self.weights = torch.ones_like(self.group_counts[self.group_array])
        self.device = device

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        y = torch.tensor([self.y_array[idx], 1-self.y_array[idx]]).float()
        g = self.group_array[idx]
        p = self.confounder_array[idx]
        w = self.weights[idx]

        img_path = os.path.join(self.basedir, self.filename_array[idx])
        img = Image.open(img_path).convert('RGB')
        # img = read_image(img_path)
        # img = img.float() / 255.

        if self.transform:
            img = self.transform(img)
        return {'id': torch.tensor(idx).to(self.device),
                'input': img.to(self.device), 
                'output': y.to(self.device), 
                'group_id': torch.tensor(float(g)).to(self.device), 
                'spurious_factor_id': torch.tensor(float(p)).to(self.device), 
                'weight': torch.tensor(float(w)).to(self.device)}


class ShortcutLearningDatasetMinor(Dataset):
    def __init__(self, basedir, metadata_name, split="train", reweight=False, transform=None, device='cuda'):
        try:
            split_i = ["train", "val", "test"].index(split)
        except ValueError:
            raise(f"Unknown split {split}")
        metadata_df = pd.read_csv(os.path.join(basedir, metadata_name))
        print(f'Total samples #:{len(metadata_df)}')
        self.metadata_df = metadata_df[metadata_df["split"] == split_i]
        print(f'Split samples #:{len(self.metadata_df)}')
        self.basedir = basedir
        self.transform = transform
        self.y_array = self.metadata_df['y'].values
        self.p_array = self.metadata_df['place'].values
        self.n_classes = np.unique(self.y_array).size
        self.confounder_array = self.metadata_df['place'].values
        self.n_places = 2
        self.group_array = (self.y_array * self.n_places + self.confounder_array).astype('int')
        self.n_groups = self.n_classes * self.n_places
        self.group_counts = (
                torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array)).sum(1).float()
        self.y_counts = (
                torch.arange(self.n_classes).unsqueeze(1) == torch.from_numpy(self.y_array)).sum(1).float()
        self.p_counts = (
                torch.arange(self.n_places).unsqueeze(1) == torch.from_numpy(self.p_array)).sum(1).float()
        self.filename_array = self.metadata_df['img_filename'].values
        self.device = device

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        y = torch.tensor([self.y_array[idx], 1-self.y_array[idx]]).float()
        g = self.group_array[idx]
        p = self.confounder_array[idx]
        w = 1.0

        img_path = os.path.join(self.basedir, self.filename_array[idx])
        img = Image.open(img_path).convert('RGB')
        # img = read_image(img_path)
        # img = img.float() / 255.

        if self.transform:
            img = self.transform(img)
        return {'id': torch.tensor(idx).to(self.device),
                'input': img.to(self.device), 
                'output': y.to(self.device), 
                'group_id': torch.tensor(float(g)).to(self.device), 
                'spurious_factor_id': torch.tensor(float(p)).to(self.device),
                'weight': torch.tensor(float(w)).to(self.device)}


class SubpopDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    INPUT_SHAPE = None       # Subclasses should override
    SPLITS = {               # Default, subclasses may override
        'tr': 0,
        'va': 1,
        'te': 2
    }
    EVAL_SPLITS = ['te']     # Default, subclasses may override

    def __init__(self, root, split, metadata, transform, train_attr='yes', subsample_type=None, duplicates=None, device='cuda'):
        self.metadata_df = pd.read_csv(metadata)#[:5000]
        self.metadata_df = self.metadata_df[self.metadata_df["split"] == (self.SPLITS[split])]

        self.idx = list(range(len(self.metadata_df)))
        self.x = self.metadata_df["filename"].astype(str).map(lambda x: os.path.join(root, x)).tolist()
        self.y = self.metadata_df["y"].values
        self.p = self.metadata_df["a"].values
        self.transform_ = transform
        self.n_places = len(set(self.p))
        self.n_classes = len(set(self.y))
        self.n_groups = self.n_places * self.n_classes
        self.class_sizes = [0] * self.n_classes
        self.group_array = (self.y * self.n_places + self.p).astype('int')
        self._count_groups()
        self.device = device

        if subsample_type is not None:
            self.subsample(subsample_type)

        if duplicates is not None:
            self.duplicate(duplicates)

    def _count_groups(self):
        self.weights_g, self.weights_y = [], []
        self.group_sizes = [0] * self.n_places * self.n_classes
        self.class_sizes = [0] * self.n_classes

        for i in self.idx:
            self.group_sizes[self.n_places * self.y[i] + self.p[i]] += 1
            self.class_sizes[self.y[i]] += 1

        for i in self.idx:
            self.weights_g.append(len(self) / self.group_sizes[self.n_places * self.y[i] + self.p[i]])
            self.weights_y.append(len(self) / self.class_sizes[self.y[i]])

    def subsample(self, subsample_type):
        assert subsample_type in {"group", "class"}
        perm = torch.randperm(len(self)).tolist()
        min_size = min(list(self.group_sizes)) if subsample_type == "group" else min(list(self.class_sizes))

        counts_g = [0] * self.n_places * self.n_classes
        counts_y = [0] * self.n_classes
        new_idx = []
        for p in perm:
            y, a = self.y[self.idx[p]], self.p[self.idx[p]]
            if (subsample_type == "group" and counts_g[self.n_places * int(y) + int(a)] < min_size) or (
                    subsample_type == "class" and counts_y[int(y)] < min_size):
                counts_g[self.n_places * int(y) + int(a)] += 1
                counts_y[int(y)] += 1
                new_idx.append(self.idx[p])

        self.idx = new_idx
        self._count_groups()

    def duplicate(self, duplicates):
        new_idx = []
        for i, duplicate in zip(self.idx, duplicates):
            new_idx += [i] * duplicate
        self.idx = new_idx
        self._count_groups()

    def __getitem__(self, index):
        i = torch.tensor(self.idx[index]).to(self.device)
        x = self.transform(self.x[i]).to(self.device)
        y = torch.zeros(3, dtype=torch.float).to(self.device)
        y[self.y[i]] = 1
        # y = torch.tensor(self.y[i])
        g = torch.tensor(self.group_array[i], dtype=torch.long).to(self.device)
        a = torch.tensor(self.p[i], dtype=torch.long).to(self.device)
        return i, x, y, g, a

    def __len__(self):
        return len(self.idx)


class MultiNLI(SubpopDataset):
    N_STEPS = 30001
    CHECKPOINT_FREQ = 1000

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None):
        root = os.path.join(data_path, "multinli", "glue_data", "MNLI")
        metadata = os.path.join(data_path, "multinli", "metadata_multinli.csv")
        # https://github.com/izmailovpavel/spurious_feature_learning/blob/main/dataset_files/utils_glue.py under `root`` is needed to load data
        sys.path.append(root)

        self.features_array = []
        assert hparams['text_arch'] == 'bert-base-uncased'
        for feature_file in [
            "cached_train_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli-mm",
        ]:
            features = torch.load(os.path.join(root, feature_file))
            self.features_array += features

        self.all_input_ids = torch.tensor(
            [f.input_ids for f in self.features_array], dtype=torch.long)
        self.all_input_masks = torch.tensor(
            [f.input_mask for f in self.features_array], dtype=torch.long)
        self.all_segment_ids = torch.tensor(
            [f.segment_ids for f in self.features_array], dtype=torch.long)
        self.all_label_ids = torch.tensor(
            [f.label_id for f in self.features_array], dtype=torch.long)
        self.x_array = torch.stack(
            (self.all_input_ids, self.all_input_masks, self.all_segment_ids), dim=2)
        self.data_type = "text"
        super().__init__("", split, metadata, self.transform, train_attr, subsample_type, duplicates)

    def transform(self, i):
        return self.x_array[int(i)]


class CivilComments(SubpopDataset):
    N_STEPS = 30001
    CHECKPOINT_FREQ = 1000

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None,
                 granularity="coarse"):
        text = pd.read_csv(os.path.join(
            data_path, "civilcomments/civilcomments_{}.csv".format(granularity))
        )
        metadata = os.path.join(data_path, "civilcomments", "metadata_civilcomments_{}.csv".format(granularity))

        self.text_array = list(text["comment_text"])
        if hparams['text_arch'] == 'bert-base-uncased':
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif hparams['text_arch'] in ['xlm-roberta-base', 'allenai/scibert_scivocab_uncased']:
            self.tokenizer = AutoTokenizer.from_pretrained(hparams['text_arch'])
        elif hparams['text_arch'] == 'distilbert-base-uncased':
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        elif hparams['text_arch'] == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            raise NotImplementedError
        self.data_type = "text"
        super().__init__("", split, metadata, self.transform, train_attr, subsample_type, duplicates)

    def transform(self, i):
        text = self.text_array[int(i)]
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=220,
            return_tensors="pt",
        )

        if len(tokens) == 3:
            return torch.squeeze(
                torch.stack((
                    tokens["input_ids"],
                    tokens["attention_mask"],
                    tokens["token_type_ids"]
                ), dim=2
                ), dim=0
            )
        else:
            return torch.squeeze(
                torch.stack((
                    tokens["input_ids"],
                    tokens["attention_mask"]
                ), dim=2
                ), dim=0
            )


class CivilCommentsFine(CivilComments):

    def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None):
        super().__init__(data_path, split, hparams, train_attr, subsample_type, duplicates, "fine")


# class MetaShift(SubpopDataset):
#     CHECKPOINT_FREQ = 300 
#     INPUT_SHAPE = (3, 224, 224,)

#     def __init__(self, data_path, split, hparams, train_attr='yes', subsample_type=None, duplicates=None):
#         metadata = os.path.join(data_path, "metashift", "metadata_metashift.csv")

#         transform = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         ])
#         self.data_type = "images"
#         super().__init__('/', split, metadata, transform, train_attr, subsample_type, duplicates)

#     def transform(self, x):
#         return self.transform_(Image.open(x).convert("RGB"))
