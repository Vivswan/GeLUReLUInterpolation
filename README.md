### GeLUReLUInterpolation

This is the official repository for the paper: <br>
[Leveraging Continuously Differentiable Activation for Learning in Analog and Quantized Noisy Environments](https://arxiv.org/abs/2402.02593)

## Requirements

The following packages are required to run the simulation:

- [Python 3.6+](https://www.python.org/downloads/)
- [PyTorch 2.0.0+](https://pytorch.org/get-started/locally/)
- [Tensorboard](https://www.tensorflow.org/tensorboard)
- Other required python packages are listed in [requirements.txt](requirements.txt) file.

## Run the simulation

### Datasets

- CIFAR-10 and CIFAR-100 datasets are used in the experiments. The datasets are automatically downloaded by the PyTorch
  library.

### Models

- ConvNet: Model with 6 convolutional layers and 3 fully connected layers. Run this model using `src/run_conv.py`
  script.
- ResNet: Run this model using `src/run_resnet.py` script.
- VGG: Run this model using `src/run_vgg.py` script.
- ViT: Run this model using `src/run_vit.py` script.

## Cite

We would appreciate if you cite the following paper in your publications if you find this code useful:

```bibtex
@article{pintus_integrated_2024,
	title = {Integrated non-reciprocal magneto-optics with ultra-high endurance for photonic in-memory computing},
	issn = {1749-4885, 1749-4893},
	url = {https://www.nature.com/articles/s41566-024-01549-1},
	doi = {10.1038/s41566-024-01549-1},
	abstract = {Abstract
            Processing information in the optical domain promises advantages in both speed and energy efficiency over existing digital hardware for a variety of emerging applications in artificial intelligence and machine learning. A typical approach to photonic processing is to multiply a rapidly changing optical input vector with a matrix of fixed optical weights. However, encoding these weights on-chip using an array of photonic memory cells is currently limited by a wide range of material- and device-level issues, such as the programming speed, extinction ratio and endurance, among others. Here we propose a new approach to encoding optical weights for in-memory photonic computing using magneto-optic memory cells comprising heterogeneously integrated cerium-substituted yttrium iron garnet (Ce:YIG) on silicon micro-ring resonators. We show that leveraging the non-reciprocal phase shift in such magneto-optic materials offers several key advantages over existing architectures, providing a fast (1 ns), efficient (143 fJ per bit) and robust (2.4 billion programming cycles) platform for on-chip optical processing.},
	language = {en},
	urldate = {2024-12-04},
	journal = {Nature Photonics},
	author = {Pintus, Paolo and Dumont, Mario and Shah, Vivswan and Murai, Toshiya and Shoji, Yuya and Huang, Duanni and Moody, Galan and Bowers, John E. and Youngblood, Nathan},
	month = oct,
	year = {2024},
}
```

Or in textual form:

```text
Vivswan Shah, and Nathan Youngblood. "AnalogVNN: A fully modular framework for modeling 
and optimizing photonic neural networks." APL Machine Learning 1.2 (2023).
DOI: 10.1063/5.0134156
```