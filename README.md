# Probabilistic Linear Regression Attack on reusability of IronMask

Implementation of Probabilistic Linear Regression Attack to attack the reusability of IronMask, which is a facial template protection scheme. The project is mainly covering the experiments -- Section 5 of the paper "Probabilistic Linear Regression Attack on reusability of IronMask".


Important notebooks:
- `svd.ipynb`: test success rate for svd-based solver given "correct matrix"
- `local_search.ipynb`: test success rate for lsa-based solver given "correct matrix"
- `example.ipynb`: examples for probabilistic linear regression solver to attack on reusability of IronMask
- `noise-trade-off.ipynb`: defense strategies of IronMask, including adding extra noise and salting
- `real-word-dataset.ipynb`: use real world dataset to validate the attack and test the recognition performance when applying the "adding extra noise" defense

Important python files:
- `attacker.py`: implementation of probabilistic linear regression solver
- `ironmask.py`: secure sketch part of IronMask scheme
- `sample.py`: sampler for simulating getting multiple sketches from biometrics
- `utils.py`: useful utils for `sample.py` and `attacker.py`

## Prerequisites

### Hardware:
1. Recommended: 16GB or more RAM

### Environments:

Two environments with dependencies are provided: one to pre-process the datasets and another to run the experiments using the provided Jupyter notebooks.

#### conda environment for dataset processing

1. Linux or WSL (WINDOWS) or WINDOWS
2. Use the `conda` environment (`mtcnn`) by executing `conda env create -f envs/environment_mtcnn.yml` (only to pre-process the datasets, irrelevant to experiments)

#### Environments for running experiments

1. Build the docker image by executing `docker build -t hypersphere-secure-sketch-probabilistic-linear-regression-attack:v-pets .` or
   obtain the already built image from
   [DockerHub](https://hub.docker.com/r/zhupengxu/hypersphere-secure-sketch-probabilistic-linear-regression-attack)
by running `docker pull zhupengxu/hypersphere-secure-sketch-probabilistic-linear-regression-attack:v-pets`
2. Run the Docker container by binding it to port `-p 8888:8888` with the command `docker run -p 8888:8888 --name plra hypersphere-secure-sketch-probabilistic-linear-regression-attack:v-pets`
3. Open in the browser the Jupyter session link provided in the terminal output:
   `http://127.0.0.1:8888/tree?token=......`

or

1. Use `uv` and build the environment by executing `uv sync` in the root directory of this project
2. Then open the Jupyter notebooks


## Example: Implement attack by simulating data
1. Follow and execute the Jupyter notebook `example.ipynb`.

## Experiments by simulating data
1. Follow and execute the Jupyter notebook `local_search.ipynb` for local search solver
2. Follow and execute the Jupyter notebook `svd.ipynb` for svd solver
3. You can adjust the parameters `k, noise_angle, threshold` to see the effects of these parameters or verify the paper and get results.

## Experiments on real dataset (e.g. FEI dataset)

Follow the instructions below to download and prepare the FEI dataset.

### Real Dataset
1. You should follow the [insightface](https://github.com/deepinsight/insightface/tree/master/detection/retinaface) instructions to align and crop the face, here we give a full example of how to do it:
   1. Download FEI dataset from the [website](https://fei.edu.br/~cet/facedatabase.html) (original images)
   2. Extract them and put all images in the same directory
   3. Create the conda environment (`mtcnn`): `conda env create -f envs/environment_mtcnn.yml`
   4. Activate the environment: `conda activate mtcnn`
   5. Build `rcnn` by running `make` in the directory `insightface/detection/retinaface` with the `mtcnn` env
   6. Download `retinaface-r50` model (pretrained ImageNet ResNet50) from `insightface` following the [website commands](https://github.com/deepinsight/insightface/tree/master/detection/retinaface)
   7. In the dataset directory, preprocess the face by executing `python3 face_retinaface_align.py --data_dir your_data_dir --output_dir your_output_dir --det_prefix your_retinaface_r50_dir` with the proper paths
      - Note: `your_retinaface_r50_dir` should be like `your_path/R50` where although `R50` is not a directory, it is the prefix of the pretrained model.

### Prepare real dataset

1. Use the script `dataset/fei_dataset_build.py` to build the dataset `labels.csv` by running `python3 fei_dataset_build --fei_dir --preprocess_fei_dir --output_file output_dir/labels.csv` and replacing with proper directory paths. We recommend you to put `preprocess_fei_dir` under a new directory and put the `labels.csv` in this directory, the structure of the files would look like:
```
-- fei_upper_directory(dir)
   -- labels.csv
   -- face_files(dir)
      -- 1-01.jpg
      -- 1-02.jpb
      -- ...
```
1. (options) To prepare other face images datasets, you should follow similar operations than the ones for the FEI dataset: preprocess/align the images and build the `labels.csv` file with format:
```csv
accumulate_indicator, image_file_path_relative_to_labels_csv_file_directory_path, label
0, face_files/1-01.jpg,1
1, face_files/1-02.jpg,2
...
x, face_files/8-10.jpg,8
```

### Testing on the Real Dataset
1. Download the proper model from `insightface` ["model zoo"](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch). Here we take `ms1mv3_arcface_r100_fp16` as our backend (store in `models` directory)
2. Follow and execute the Jupyter notebook `real-world-dataset.ipynb`

Warning: The model and dataset need to be added as a volume to the docker container if you are using docker. To do so, after having finished the section "Prepare real dataset", run at the root of this git repository `docker run -p 8888:8888 --name plra -v ${PWD}:/app hypersphere-secure-sketch-probabilistic-linear-regression-attack:v-pets` to launch a docker container with the current directory mounted as a volume on `/app` inside the container.