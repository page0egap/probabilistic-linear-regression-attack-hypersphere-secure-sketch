from .fatial_dataset import FatialDataset
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm as qtmd
import numpy as np
import os
import gc
import datetime


def _normalize(embeddings):
    if isinstance(embeddings, np.ndarray):
        return embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
    elif isinstance(embeddings, torch.Tensor):
        return embeddings / torch.norm(embeddings, dim=-1, keepdim=True)
    else:
        raise TypeError("embeddings type is not numpy array or torch.tensor, type is {}".format(type(embeddings)))

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

    def dump_embeddings(self, backbone, batch_size=10, **kwargs):
        if self.labels_dict_without_folder is not None or self.file_path_list is not None:
            return
        if self._file_folder is None:
            raise ValueError("File Folder is None")
        else:
            self._dump_embeddings_with_file_folder(backbone, batch_size=batch_size, **kwargs)

    @torch.no_grad()
    def _dump_embeddings_with_file_folder(self, backbone, batch_size=10, normalize_func = _normalize, **kwargs):
        if self._check_and_build_dir():
            with open(self._labels_set_file_path, "rb") as f:
                self.file_path_list, self.file_labels_list = pickle.load(f)
                # set _mem_size to the size of file_path_list
                self._mem_size = len(self.file_path_list)
            return
        
        loader = DataLoader(self.data_set, batch_size=batch_size, shuffle=False, pin_memory=kwargs.get("pin_memory", False),
                                num_workers=kwargs.get("num_workers", 8))
        backbone.eval()
        # dump the embeddings with labels_dict, whose key is label and value is a list of embeddings
        # since the embeddings of each label may be too large, use tmp_file_list to store the temporary file path, each file contains a labels_dict
        # but different labels_dict may have same label, thus finally we need to merge the same label embeddings in the same file
        labels_dict = {}
        tmp_file_list = []
        tmp_file_labels_list = []
        for i, (images, labels) in enumerate(qtmd(loader, "Enumerating", leave=False)):
            embeddings = backbone(images)
            embeddings = normalize_func(embeddings)
            for label, embedding in zip(labels, embeddings):
                if label not in labels_dict:
                    labels_dict[label] = []
                labels_dict[label].append(embedding)
            # if labels_dict key size is larger than max_labels_size, dump labels_dict to temporary file and clear labels_dict, remember the labels_dict keys 
            if len(labels_dict) > self.max_labels_size:
                tmp_file_list.append(self._dump_labels_dict_to_tmp_file(labels_dict))
                tmp_file_labels_list.append(set(labels_dict.keys()))
                labels_dict = {}
                gc.collect()
        
        # if labels_dict is not empty, dump it to temporary file and clear labels_dict, remember the labels_dict keys
        if len(labels_dict) > 0:
            tmp_file_list.append(self._dump_labels_dict_to_tmp_file(labels_dict))
            tmp_file_labels_list.append(set(labels_dict.keys()))
            labels_dict = {}

        self.file_path_list, self.file_labels_list = self._merge_embeddings(tmp_file_list, tmp_file_labels_list)
        # remove tmp_file
        [os.remove(tmp_file_path) for tmp_file_path in tmp_file_list]
        # dump final_file_path_list and final_file_labels_list to file_dir/labels_set.pickle
        with open(self._labels_set_file_path, "wb") as f:
            pickle.dump((self.file_path_list, self.file_labels_list), f)

    def _merge_embeddings(self, file_list, file_labels_list):
        final_file_path_list = []
        final_file_labels_list = []
        # get ordered labels list whose type is list
        ordered_labels_list = []
        for tmp_file_labels in file_labels_list:
            ordered_labels_list.extend(list(tmp_file_labels))
        ordered_labels_list = sorted(ordered_labels_list)
        # merge the same label embeddings in the same file
        # seperate ordered_labels_list into different parts, whose size is max_labels_size
        for i in qtmd(range(0, len(ordered_labels_list), self.max_labels_size), desc="Merging", leave=False):
            tmp_ordered_labels_set = set(ordered_labels_list[i:i+self.max_labels_size])
            # get where the tmp_ordered_labels_list is in tmp_file_list
            tmp_need_read_file_list = []
            for i, tmp_file_labels_set in enumerate(file_labels_list):
                if not tmp_ordered_labels_set.isdisjoint(tmp_file_labels_set):
                    tmp_need_read_file_list.append(file_list[i])
            # read the tmp_need_read_file_list and merge the same label embeddings in the same file
            tmp_labels_dict = {}
            for tmp_file_path in tmp_need_read_file_list:
                with open(tmp_file_path, "rb") as f:
                    tmp_labels_dict2 = pickle.load(f)
                    need_update_keys = tmp_ordered_labels_set.intersection(set(tmp_labels_dict2.keys()))
                    for k in need_update_keys:
                        if k not in tmp_labels_dict:
                            tmp_labels_dict[k] = []
                        tmp_labels_dict[k].extend(tmp_labels_dict2[k])
                    gc.collect()
            # dump the tmp_labels_dict to tmp file
            tmp_file_path = self._dump_labels_dict_to_tmp_file(tmp_labels_dict)
            final_file_path_list.append(tmp_file_path)
            final_file_labels_list.append(tmp_ordered_labels_set)
            gc.collect()
        return final_file_path_list, final_file_labels_list

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
    
    def _dump_labels_dict_to_tmp_file(self, labels_dict):
        tmp_file_path = os.path.join(self.embedding_dir, f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}.pickle")
        with open(tmp_file_path, "wb") as f:
            print("dumping labels_dict to tmp file: {}".format(tmp_file_path))
            pickle.dump(labels_dict, f)
        return tmp_file_path