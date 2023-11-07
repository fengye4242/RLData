import hdf5storage
import numpy as np
from torch.utils.data import Dataset

class My_120_norm_Dataset(Dataset):
    def __init__(self , data_path):
        super(My_120_norm_Dataset, self).__init__()
        data = hdf5storage.loadmat(data_path)
        self.H_V=data['n_H_V']
        self.K_V=data['n_K_V']
        self.T_length=data['T_length']
        self.phase_point=data['phase_point'].astype(np.float)
        self.phase_label=data['phase_label']
        self.M_label=data['M_label']




    def __getitem__(self, item):
        H_V=self.H_V[item]
        K_V=self.K_V[item]
        phase_point=self.phase_point[item]
        phase_label=self.phase_label[item]
        M_label=self.M_label[item]
        T_length=self.T_length[item]
        return H_V,K_V,phase_point,phase_label,M_label,T_length

    def __len__(self):
        return len(self.T_length)

class My_norm_Dataset(Dataset):
    def __init__(self , data_path):
        super(My_norm_Dataset, self).__init__()
        data = hdf5storage.loadmat(data_path)
        self.H_V=data['H_V']
        self.K_V=data['K_V']
        self.T_length=data['T_length']
        self.phase_point=data['phase_point'].astype(np.float)
        self.phase_label=data['phase_label']
        self.M_label=data['M_label']




    def __getitem__(self, item):
        H_V=self.H_V[item]
        K_V=self.K_V[item]
        phase_point=self.phase_point[item]
        phase_label=self.phase_label[item]
        M_label=self.M_label[item]
        T_length=self.T_length[item]
        return H_V,K_V,phase_point,phase_label,M_label,T_length

    def __len__(self):
        return len(self.T_length)
