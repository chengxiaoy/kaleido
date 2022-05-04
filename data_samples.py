import joblib
from torch.utils.data import dataloader, DataLoader, random_split
from torchvision.datasets.celeba import CelebA
from torchvision import transforms
import lmdb
from torch.utils.data import Dataset
from io import BytesIO
from PIL import Image

celebA_data_path = "/data/kaggle/shared/Data/"
ffhq_lmdb_data_path = "/data/kaggle/shared/Data/ffhq_lmdb"


def data_transforms(img_size):
    SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
    SetScale = transforms.Lambda(lambda X: X / X.sum(0).expand_as(X))

    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.CenterCrop(148),
                                    transforms.Resize(img_size),
                                    transforms.ToTensor(),
                                    SetRange])

    return transform


def get_celebA_dataloader(image_size: int, batch_size: int, split: str) -> dataloader.DataLoader:
    celebA_dataset = CelebA(root=celebA_data_path, split=split, transform=data_transforms(image_size), download=False)
    celebA_dataloader = dataloader.DataLoader(celebA_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return celebA_dataloader


ffhq_t = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda X: 2 * X - 1.)
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ]
)


def get_ffhq_dataloader(resolution: int, batch_size: int, transform=ffhq_t, split="train"):
    dataset = MultiResolutionDataset(ffhq_lmdb_data_path, transform)
    dataset.resolution = resolution
    length = len(dataset)
    train_size, validate_size = int(0.8 * length), int(0.2 * length)
    train_set, validate_set = random_split(dataset, [train_size + 1, validate_size])
    if split == "train":
        return DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=True)
    if split == "val":
        return DataLoader(validate_set, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=True)
    return None


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=8):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img, 1
