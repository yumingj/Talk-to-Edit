import os.path

import numpy as np
from torch.utils.data import Dataset


class EncoderDataset(Dataset):

    def __init__(self, preprocessed_dir):

        # load text
        text_path = os.path.join(preprocessed_dir, 'text.npy')
        self.text = np.load(text_path)
        # load system_mode
        system_mode_path = os.path.join(preprocessed_dir, 'system_mode.npy')
        self.system_mode = np.load(system_mode_path)
        # load labels
        labels_path = os.path.join(preprocessed_dir, 'labels.npy')
        self.labels = np.load(labels_path)

    def __getitem__(self, index):
        # retrieve text
        text = self.text[index]
        # retrieve system_mode
        system_mode = self.system_mode[index]
        # retrieve labels
        labels = self.labels[index]

        return text, system_mode, labels

    def __len__(self):
        return len(self.text)


def main():
    """ Testing the Dataset"""

    encoderdataset = EncoderDataset(
        preprocessed_dir=  # noqa
        ''  # noqa
    )
    print('len(encoderdataset):', len(encoderdataset))
    print('encoderdataset[0]:', encoderdataset[0])


if __name__ == '__main__':
    main()
