from .fatial_dataset import FatialDataset
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm as qtmd
import numpy as np
import os


class FatialDataEmbeddingsDict:
    """
    MainTain the dictionary of embeddings of data_set, whose key is label and value is a list of normalized embeddings.
    """
    max_labels_size = 3000

    def __init__(self, file_folder, data_set:FatialDataset, mem_size=2, file_raw_name="labels_set.pickle") -> None:
        self._file_folder = file_folder
        self._data_set = data_set
        self._file_raw_name = file_raw_name
        self.labels_dict_without_folder = None
        self.file_path_list = None
        self.file_labels_list = None
        self.labels_dict_with_folder = None
        self._mem_size = mem_size
        self._mode = "Normal"
        self._position = None

    def set_position(self, position):
        self._position = position

    @property
    def mode(self,):
        return self._mode
    
    @mode.setter
    def mode(self, mode):
        if mode not in ["Normal", "Single"]:
            raise ValueError("mode must be Normal or Single")
        self._mode = mode

    def __getitem__(self, key):
        return_value = None
        if self._file_folder is None:
            return_value = self.labels_dict_without_folder[key]
        elif self.labels_dict_with_folder is not None and key in self.labels_dict_with_folder:
            return_value = self.labels_dict_with_folder[key]
        if return_value is None:
            for file_path, file_labels in zip(self.file_path_list, self.file_labels_list):
                if key in file_labels:
                    with open(file_path, "rb") as f:
                        tmp_labels_dict = pickle.load(f)
                        if self.labels_dict_with_folder is None:
                            self.labels_dict_with_folder = tmp_labels_dict
                        else:
                            self.labels_dict_with_folder.update(tmp_labels_dict)
                        return_value = tmp_labels_dict[key]
                    break
        return self._process_getvalue(return_value)

    def _process_getvalue(self, value):
        return_list = None
        if self._mode == "Normal":
            return_list = value
        elif self._mode == "Single":
            if isinstance(value, list):
                if len(value) > 1:
                    # random choose one value
                    return_list = [value[1]]
                elif len(value) == 1:
                    return_list = [value[0]]
            else:
                raise TypeError("value type is not list")
        if self._position is not None:
            return_list = [value[self._position] for value in return_list]
        return return_list

    def keys(self,):
        if self._file_folder is None:
            return self.labels_dict_without_folder.keys()
        else:
            results = set()
            [results.update(labels) for labels in self.file_labels_list]
            return results

    def get_keys_without_duplicate(self, index=0):
        pass

    @torch.no_grad()
    def dump_norms(self, backbone, batch_size=10, **kwargs):
        loader = DataLoader(self.data_set, batch_size=batch_size, shuffle=False, pin_memory=kwargs.get("pin_memory", True),
                                num_workers=kwargs.get("num_workers", 8))
        backbone.eval()
        labels_dict = {}
        for i, (images, labels) in enumerate(qtmd(loader, "Enumerating", leave=False)):
            embeddings:np.ndarray = backbone(images)
            embeddings_norm = np.linalg.norm(embeddings, axis=-1, keepdims=True)
            for label, embedding_norm in zip(labels, embeddings_norm):
                if label not in labels_dict:
                    labels_dict[label] = []
                labels_dict[label].append(embedding_norm)
            
        return labels_dict

    def dump_embeddings(self, backbone, batch_size=10, normalize_func = None, sphere=False, **kwargs):
        if self.labels_dict_without_folder is not None or self.file_path_list is not None:
            return
        if sphere:
            normalize_func = None
        if self._file_folder is None:
            self._dump_embeddings_without_file_folder(backbone, batch_size=batch_size, normalize_func=normalize_func, **kwargs)
        else:
            self._dump_embeddings_with_file_folder(backbone, batch_size=batch_size, normalize_func=normalize_func, **kwargs)

    @torch.no_grad()
    def _dump_embeddings_with_file_folder(self, backbone, batch_size=10, normalize_func = None, **kwargs):
        if self._check_and_build_dir():
            with open(self._labels_set_file_path, "rb") as f:
                self.file_path_list, self.file_labels_list = pickle.load(f)
                # set _mem_size to the size of file_path_list
                self._mem_size = len(self.file_path_list)
            return
        else:
            raise NotImplemented("Not allow to dynamically dump embeddings")

    def _check_data_set(self,):
        if self._data_set is None:
            raise ValueError("data_set is None, please set it by set_data_set function")
        if not isinstance(self.data_set, FatialDataset):
            raise TypeError("data_set is not instance of FatialDataset")
        return True
    
    @property
    def data_set(self,):
        return self._data_set
    
    @data_set.setter
    def data_set(self, data_set:FatialDataset):
        self._data_set = data_set

    @property
    def embedding_dir(self,):
        self._check_data_set()
        return os.path.join(self._file_folder, self._data_set.name)

    @property 
    def _labels_set_file_path(self,):
        return os.path.join(self.embedding_dir, self._file_raw_name)

    def _check_and_build_dir(self,):
        if not os.path.exists(self.embedding_dir):
            os.makedirs(self.embedding_dir)
            return False
        elif not os.path.exists(self._labels_set_file_path):
            return False
        return True