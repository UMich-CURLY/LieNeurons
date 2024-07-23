# Lie Neurons: Adjoint Equivariant Neural Networks for Semi-simple Lie Algebras

Paper: [Link](https://arxiv.org/abs/2310.04521)

Poster: [Link](figures/Lie_Nuerons_ICML24.pdf)

An MLP framework that takes Lie algebraic data as inputs and is equivariant to the adjoint representation of the group by construction.

![front_figure](figures/lie_neurons_icon.jpg?raw=true "Title")

## Modules
![modules](figures/lie_neurons_modules.jpg?raw=true "Modules")

## Docker
* We provide [docker](https://docs.docker.com/get-started/) files in [`docker/`](https://github.com/UMich-CURLY/LieNeurons/tree/main/docker).
* Detailed tutorial on how to build the docker container can be found in the README in each docker folder.

## Training the Network
* All the training codes for experiments are in [`experiment/`](https://github.com/UMich-CURLY/LieNeurons/tree/main/experiment).
* Before training, you'll have to generate the data using Python scripts in [`data_gen`](https://github.com/UMich-CURLY/LieNeurons/tree/main/data_gen).

## Citation
If you find the work useful, please kindly cite our paper:
```
@inproceedings{takakura2023approximation,
  title={Lie Neurons: Adjoint-Equivariant Neural Networks for Semisimple Lie Algebras},
  author={Lin, Tzu-Yuan and Zhu, Minghan and Ghaffari, Maani},
  booktitle={International Conference on Machine Learning},
  pages={},
  year={2024},
  organization={PMLR}
}
```
