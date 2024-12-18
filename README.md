<h3 align="center"><strong>Lie Neurons: Adjoint Equivariant Neural Networks for Semi-simple Lie Algebras</strong></h3>

  <p align="center">
      <a href="https://tzuyuan.github.io/" target='_blank'>Tzu-Yuan Lin</a><sup>*</sup>&nbsp;&nbsp;&nbsp;
      <a href="https://minghanz.github.io/" target='_blank'>Minghan Zhu</a><sup>*</sup>&nbsp;&nbsp;&nbsp;
      <a href="https://name.engin.umich.edu/people/ghaffari-maani/" target='_blank'>Maani Ghaffari</a><sup></sup>&nbsp;&nbsp;&nbsp;
  <br />
  <sup>*</sup>Eqaul Contributions&nbsp;&nbsp;&nbsp;
  </p>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2310.04521" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-%F0%9F%93%83-slategray">
  </a>

  <a href="https://github.com/UMich-CURLY/LieNeurons/blob/main/figures/Lie_Nuerons_ICML24.pdf" target='_blank'>
    <img src="https://img.shields.io/badge/poster-%F0%9F%93%88-green">
  </a>
</p>

## About
An MLP framework that takes Lie algebraic data as inputs and is equivariant to the adjoint representation of the group by construction.

![front_figure](figures/lie_neurons_icon.jpg?raw=true "Title")

## Modules
![modules](figures/lie_neurons_modules.jpg?raw=true "Modules")

## Updates
* [07/2024] The initial code is open-sourced. We are still re-organizing the code. We plan to release a cleaner version of the code soon. Feel free to reach out if you have any questions! :)
* [07/2024] We presented our paper at ICML 24!

## Docker
* We provide [docker](https://docs.docker.com/get-started/) files in [`docker/`](https://github.com/UMich-CURLY/LieNeurons/tree/main/docker).
* Detailed tutorial on how to build the docker container can be found in the README in each docker folder.

## Training the Network
* All the training codes for experiments are in [`experiment/`](https://github.com/UMich-CURLY/LieNeurons/tree/main/experiment).
* Before training, you'll have to generate the data using Python scripts in [`data_gen`](https://github.com/UMich-CURLY/LieNeurons/tree/main/data_gen).
* Empirically, we found out that using a lower learning rate (around `3e-5`) helps the convergence during training. This is likely due to the lack of normalization layers.
* When working with $\mathfrak{so}(3)$, Lie Neurons specialize to [Vector Neurons](https://github.com/FlyingGiraffe/vnn) with an additional bracket nonlinearity and a channel mixing layer. Since the inner product is well-defined on $\mathfrak{so}(3)$, one can plug in the batch normalization layers proposed in Vector Neurons to improve stability during training. 

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
