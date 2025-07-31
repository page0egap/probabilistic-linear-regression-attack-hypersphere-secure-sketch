# Probabilistic Linear Regression Attack on reusability of IronMask 

Implementation of Probabilistic Linear Regression Attack to attack the reusability of IronMask, which is fatial template protection scheme. The project is mainly covering the experiments -- Section 5 of the paper "Probabilistic Linear Regression Attack on reusability of IronMask".


important notebooks:
- svd.ipynb:  test success rate for svd-based solver given "correct matrix"
- local_search.ipynb: test success rate for lsa-based solver given "correct matrix"
- example.ipynb: examples for probabilistic linear regression solver to attack on reusability of IronMask 
- noise-trade-off.ipynb: defence strategies of ironmask, including adding extra noise and salting
- real-word-dataset.ipynb: use real world dataset to validate the attack and test the recognition performance when applying the "adding extra noise" defence

important python files:
- attacker.py: implementation of probabilistic linear regression solver
- ironmask.py: secure sketch part of ironmask scheme
- sample.py: sampler for simulating getting multiple sketches from biometrics
- utils.py: useful utils for sample.py and attacker.py

## Prequsites

### Hardware:
1. Recommended: 16GB or more RAM

### Envs:

#### enviroments for running experiments

1. Run docker to build the image by `docker build -t probabilistic-linear-regression-attack .` or already built image from dockerhub `https://hub.docker.com/r/zhupengxu/hypersphere-secure-sketch-probabilistic-linear-regression-attack`.
2. Run it by with binding port `-p 8888:8888` with command `docker run -p 8888:8888 --name plra probabilistic-linear-regression-attack`
3. Open the browser and go to `localhost:8888`, you probabily need the jupyter session key which will be displayed in the terminal when you run the docker.

or 

1. `uv` environment and build the env by `uv sync` in the root directory of this project;

#### conda environment for dataset processing

1. Linux or WSL(WINDOWS) or WINDOWS
2. `conda` environments(only for datasets, irrelavent to experiments) and build the envs by `conda env create -f {envfile}` by replacing {envfile} with the file in envs/

### Datasets:
1. (Only for Experiments by real dataset) You should follow the [insightface](https://github.com/deepinsight/insightface/tree/master/detection/retinaface) instructions to align and crop the face, here we give a full example of how to do it:
   1. Download FEI dataset from website: https://fei.edu.br/~cet/facedatabase.html(original images); 
   2. Extract them and put all images in the same directory;
   3. Create conda env(mtcnn) from file `envs/environment_mtcnn.yml` by command `conda env create -f environment_mtcnn.yml` in directory `envs`;
   4. Activate the enviroment by command `conda activate mtcnn`;
   5. Build `rcnn` by command `make` in directory `insightface/detection/retinaface` with `mtcnn` env;
   6. Download retinaface-r50 model(pretrained ImageNet ResNet50) from `insightface` following the website commands: https://github.com/deepinsight/insightface/tree/master/detection/retinaface
   7. In dataset directory, preprocess the face images following the script `face_retinaface_align.py` by command `python3 face_retinaface_align.py --data_dir your_data_dir --output_dir your_output_dir --det_prefix your_retinaface_r50_dir` with replacement of proper directory paths;
      1. `your_retinaface_r50_dir` should be like `your_path/R50` where although `R50` is not a directory, it is the prefix of the pretrained model.

## Example: Implement attack by simulating data
1. Follow the `example.ipynb` with jupyter core( .venv).

## Experiments by simulating data
1. Follow the `local_search.ipynb` with jupyter core( .venv) for local search solver;
2. Follow the `svd.ipynb` with jupyter core( .venv) for svd solver;
3. You could adjust the parameters `k, noise_angle, threshold` to see the
effects of these parameters or verify the paper and get results.

## Experiments by real dataset(e.g. FEI dataset)

You might find the already prepared dataset `fei_face_dataset`(In our already built docker image). If you just want to verify, you could just use `fei_face_dataset` and skip the step "Prepare real dataset".

### Prepare real dataset


1. use script `dataset/fei_dataset_build.py` to build the dataset `labels.csv` by command `python3 fei_dataset_build --fei_dir --preprocess_fei_dir --output_file output_dir/labels.csv` with replacement of proper directory. We recommend you to put `preprocess_fei_dir` under a new directory and put the `labels.csv` in this directory, the structure of the files are like:
    -- fei_upper_directory(dir)
       -- labels.csv
       -- face_files(dir)
          -- 1-01.jpg
          -- 1-02.jpb
          -- ...
2. (options) To prepare other face images dataset, you should follow the similar operations like building fei dataset: preprocess/align the images, build the labels.csv file. The labels.csv file format is like:
    accumulate_indicator, image_file_path_relative_to_labels_csv_file_directory_path, label
    0, face_files/1-01.jpg,1
    1, face_files/1-02.jpg,2
    ...
    x, face_files/8-10.jpg,8

### Testing with the Real Dataset
1. Download proper model from `insightface` "model zoo": https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch. Here we take ms1mv3_arcface_r100_fp16 as our backend(store in `models` directory);
2. follow the `real-world-dataset.ipynb` notebook(.venv)
   
Warning: The model and dataset should be attached to or exist in the docker image if you are using docker. Because the docker container should be able to access the model and dataset through file system.

Hint: If you are using docker environment and have finished the section "Prepare real dataset", you could use command `docker run -p 8888:8888 --name plra -v /path/to/your_dataset_dir:/app/your_dataset_dir probabilistic-linear-regression-attack` to mount the dataset into the docker container. Then you could adjust the `data_fei_retinaface_path` parameter to finish the experiments. The operations for model are similar.