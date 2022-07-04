from torch.utils.data import Dataset


class DecodeDataset(Dataset):
    def __init__(self, dico, train=True, transform=None, target_transform=None):
        x_diff_rnn = dico['x_diff_rnn']
        labels = dico['labels']

        if len(x_diff_rnn) != len(labels):
            raise ValueError('features and labels must have same length.')

        train_features = []
        train_labels = []
        test_features = []
        test_labels = []

        for i, (feature, label) in enumerate(zip(x_diff_rnn, labels)):
            
            if i % 10 == 0:
                test_features.append(feature)
                test_labels.append(label)
            else:
                train_features.append(feature)
                train_labels.append(label)

        self.features = train_features if train else test_features
        self.labels = train_labels if train else test_labels
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:
            label = self.target_transform(label)
        return feature, label
