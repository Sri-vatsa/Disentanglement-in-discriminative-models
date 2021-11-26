import os,glob,re,random,tarfile,pdb,json
from PIL import Image
from torch.utils.data import Dataset
class PetsDataset(Dataset):
    def __init__(self, root, transform):
        super().__init__()
        self.root = root
        self.files = [fname for fname in os.listdir(root) if fname.endswith('.jpg')]
#         self.classes = list(set([self._parse_breed(fname) for fname in self.files]))
        lis = [self._parse_breed(fname) for fname in self.files]
        self.classes = [] 
        for i in lis:
            if i not in self.classes:
                self.classes.append(i)
        self.transform = transform

    def _parse_breed(self,fname):
        parts = fname.split('_')
        return ' '.join(parts[:-1])
    def __len__(self):
        return len(self.files)
    def _pil_loader(self,path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
    def __getitem__(self, i):
        fname = self.files[i]
        fpath = os.path.join(self.root, fname)
        img = self.transform(self._pil_loader(fpath))
        class_idx = self.classes.index(self._parse_breed(fname))
        return img, class_idx