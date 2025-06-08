# Artifact Appendix

Paper title: **Hypersphere Secure Sketch Revisited: Probabilistic Linear
Regression Attack on IronMask in Multiple Usage**

Artifacts HotCRP Id: **#16** (not your paper Id, but the artifacts id)

Requested Badge: Either **Available**, **Functional**

## Description
Implementation of Probabilistic Linear Regression Attack to attack the reusability of IronMask, which is fatial template protection scheme. The project is mainly covering the experiments -- Section 5 of the paper "Probabilistic Linear Regression Attack on reusability of IronMask".

### Security/Privacy Issues and Ethical Concerns (All badges)
The face dataset is third-party dataset, and we do not include the dataset in our repository due to  the license of the dataset, but we did provide the processed dataset in our dockerfile for convenience to evaluate, actually it violate the distribution policy of the dataset, so we might eventually not include them in any public repository. So as the model.

## Basic Requirements (Only for Functional and Reproduced badges)


### Hardware Requirements
Recommended: 16GB or more RAM

### Software Requirements
Docker

### Estimated Time and Storage Consumption
The parameters of the experiments should be **handy adjusted** according to the paper. And for each parameter, the estimated time is from few minutes to hours. We use tqdm to roughly show the progress and the estimated required time. I think taking some parameters to test might be time-available, or it could consume a few days.

## Environment 
In the following, describe how to access our artifact and all related and necessary data and software components.
Afterward, describe how to set up everything and how to verify that everything is set up correctly.

### Accessibility (All badges)
Source code: https://github.com/page0egap/probabilistic-linear-regression-attack-hypersphere-secure-sketch

### Set up the environment (Only for Functional and Reproduced badges)
We have built a docker image with a processed dataset and downloaded model. Yo could ignore the steps before "Run it by with binding port `-p 8888:8888`".

```bash
git clone https://github.com/page0egap/probabilistic-linear-regression-attack-hypersphere-secure-sketch
cd probabilistic-linear-regression-attack-hypersphere-secure-sketch
docker build . -t probabilistic-linear-regression-attack-hypersphere-secure-sketch
docker run -p  8888:8888 probabilistic-linear-regression-attack-hypersphere-secure-sketch
# open the browser and go to localhost:8888
```

### Testing the Environment (Only for Functional and Reproduced badges)
open the browser and go to localhost:8888

## Artifact Evaluation (Only for Functional and Reproduced badges)

### Main Results and Claims
List all your paper's results and claims that are supported by your submitted artifacts.

#### Main Result 1: The expected attack time in experiments in Section 5.2 and 5.3(Table 1 and Table 3) is right

By adjusting the parameters of (k, noise_angle(\theta'), threshold(\theta_t)), the exstimated attack time(Time(t_all)) should be at same order  as the paper.(notebook `svd.ipynb` for SVD algorithm and `local_search.ipynb` for LSA algorithm)

#### Main Result 2: The expected attack time in experiments in Section 5.4(Table 3) is right
By adjusting the parameters of (k, threshold(\theta_t)), the exstimated attack time(Time(t_all)) should be at same order as the paper.(notebook `real-world-dataset.ipynb`)(Due to the dataset privacy, we do not provide the Color-FERET dataset, it now could only be evaluated on FEI dataset.)

### Experiments 
List each experiment the reviewer has to execute. Describe:
 - How to execute it in detailed steps.
 - What the expected result is.
 - How long it takes and how much space it consumes on disk. (approximately)
 - Which claim and results does it support, and how.

#### Experiment 1: Noiseless and Noisy scenarios of SVD-based attacker
Open the notebook `svd.ipynb` and execute the cell. 
`## Test for svd getting 2 sketches if sampled matrix is "correct"` Corrsponding to the scenario where "# sketches" = 2. Change the settings $k, noise_angle, threshold$ based on the paper and run the Test Code and get the result.

`## Test for svd getting multiple matrices if sampled matrix is "correct"` Corrsponding to the scenario where "# sketches" not equal to 2. Change the settings $k, noise_angle, threshold$ based on the paper and run the Test Code and get the result.

#### Experiment 2: Noiseless and Noisy scenarios of LSA-based attacker
Open the notebook `local_search.ipynb` and execute the cell. 
`## Test for getting 3 sketches` Corrsponding to the scenario where "# sketches" = 3. Change the settings $k, noise_angle, threshold$ based on the paper and run the Test Code and get the result. Since the evaluation is time-consuming, we store the result of most-time consuming part and you could turn off by setting `use_already_store_result` to `False`.

`##  Test for getting (tmpk + 1) sketches` Corrsponding to the scenario where "# sketches" not equal to 3. Change the settings $k, noise_angle, threshold$ based on the paper and run the Test Code and get the result. Since the evaluation is time-consuming, we store the result of most-time consuming part and you could turn off by setting `use_already_store_result` to `False`.

#### Experiment 3: Real Dataset for LSA-based attacker
Open the notebook `local_search.ipynb` and execute the cell.

Change `## Parameters` > `### Deep Recognition Model` > `model_path` to the path of your model(here we already provided a model in `models/ms1mv3_arcface_r100_fp16` directory).

Change `## Parameters` > `### Dataset` > `data_fei_retinaface_path` to the path of your face dataset(here we already provided a model in `fei_face_dataset` directory), change selected_subset_index for different subsets of the dataset(p04, p05, p06) - [1, 2, 3] (p03, p05, p08) - [0, 2 ,5].

Change `## Parameters` > `### IronMask and Attack's Parameters` > `(k, threshold)` based on the paper. Turn off by setting `use_precompute=False`.


## Limitations (Only for Functional and Reproduced badges)
The expected time should not equalt to the paper, but with similar performance. As the computer processor's power is different,
and the estimation is probabistic, the time and also "p_k * p_f" might vary even in the same computer.

For svd-based solver, the time might be 10 times longer than the paper. It's because the underlying based algorithm is varying 
whether using intel-mkl to compute svd or not. The environment we provide is using openblas, which is not using intel-mkl, which is 
10 times slower. Intel-mkl envs are hard to provide and specific to intel-core processors. For simplicity, we choose not to provide it.

## Notes on Reusability (Only for Functional and Reproduced badges)
The attack algorithm might be reusable for other protected algorithms that has the same vulnerabilities of IronMask.

