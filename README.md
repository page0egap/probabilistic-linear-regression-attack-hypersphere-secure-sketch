# Probabilistic Linear Regression Attack on reusability of IronMask 

Implementation of Probabilistic Linear Regression Attack to attack the reusability of IronMask, which is fatial template protection scheme.


important files:

svd.ipynb -- test success rate for svd-based solver given "correct matrix"
local_search.ipynb -- test success rate for lsa-based solver given "correct matrix"
example.ipynb -- examples for probabilistic linear regression solver to attack on reusability of IronMask 

attacker.py -- implementation of probabilistic linear regression solver
ironmask.py -- secure sketch part of ironmask scheme
sample.py -- sampler for simulating getting multiple sketches from biometrics
utils.py -- useful utils for sample.py and attacker.py

## Prequsites

1. `conda` enviroments
2. `git clone insightface` in the same directory of README file
3. Linux or WSL(WINDOWS)

## Implement attack by simulating data
1. (if not)create conda env(intel-attack-m) from file `envs/environment_intel_attack_m.yml` by command `conda env create -f environment_intel_attack_m.yml` in directory `envs`;
2. Follow the `example.ipynb` with jupyter core(intel-attack-m).

## Experiments by simulating data
1. Follow the `local_search.ipynb` with jupyter core(intel-attack-m) for local search solver;
2. Follow the `svd.ipynb` with jupyter core(intel-attack-m) for svd solver;

## Experiments by real dataset(e.g. FEI dataset)

### Prepare real dataset

1. download FEI dataset from website: https://fei.edu.br/~cet/facedatabase.html(original images); 
2. extract them and put all images in the same directory;
3. create conda env(mtcnn) from file `envs/environment_mtcnn.yml` by command `conda env create -f environment_mtcnn.yml` in directory `envs`;
4. activate the enviroment by command `conda activate mtcnn`;
5. build `rcnn` by command `make` in directory `insightface/detection/retinaface` with `mtcnn` env;
6. download retinaface-r50 model(pretrained ImageNet ResNet50) from `insightface` following the website commands: https://github.com/deepinsight/insightface/tree/master/detection/retinaface
7. In dataset directory, preprocess the face images following the script `face_retinaface_align.py` by command `python3 face_retinaface_align.py --data_dir your_data_dir --output_dir your_output_dir --det_prefix your_retinaface_r50_dir` with replacement of proper directory paths;
8. use script `fei_dataset_build.py` to build the dataset `labels.csv` by command `python3 fei_dataset_build --fei_dir --preprocess_fei_dir --output_file output_dir/labels.csv` with replacement of proper directory. We recommend you to put `preprocess_fei_dir` under a new directory and put the `labels.csv` in this directory, the structure of the files are like:
    -- fei_upper_directory(dir)
       -- labels.csv
       -- face_files(dir)
          -- 1-01.jpg
          -- 1-02.jpb
          -- ...
9. (options) To prepare other face images dataset, you should follow the similar operations like building fei dataset: preprocess/align the images, build the labels.csv file. The labels.csv file format is like:
    accumulate_indicator, image_file_path_relative_to_labels_csv_file_directory_path, label
    0, face_files/1-01.jpg,1
    1, face_files/1-02.jpg,2
    ...
    x, face_files/8-10.jpg,8

### Test in Real Dataset
1. Download proper model from `insightface` "model zoo": https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch. Here we take ms1mv3_arcface_r100_fp16 as our backend;
2. follow the `real-world-dataset.ipynb` notebook.