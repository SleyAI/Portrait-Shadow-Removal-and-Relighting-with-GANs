import re
import os
from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils


class InferenceDatasetSorted(Dataset):

	def __init__(self, root, opts, transform=None):
		self.paths = sorted(data_utils.make_dataset(root), key=lambda x: int(((os.path.splitext(x)[0]).split('/').pop()).split('_').pop()))
		self.transform = transform
		self.opts = opts

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		from_im = Image.open(from_path).convert('RGB')
		if self.transform:
			from_im = self.transform(from_im)
		return from_im
