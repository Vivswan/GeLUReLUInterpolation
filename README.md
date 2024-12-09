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
@article{shah2024leveraging,
  title={Leveraging Continuously Differentiable Activation Functions for Learning in Quantized Noisy Environments},
  author={Shah, Vivswan and Youngblood, Nathan},
  journal={arXiv preprint arXiv:2402.02593},
  url = {http://arxiv.org/abs/2402.02593},
  doi = {10.48550/arXiv.2402.02593},
  year={2024}
}
```

Or in textual form:

```text
Shah, Vivswan, and Nathan Youngblood. "Leveraging Continuously Differentiable Activation
Functions for Learning in Quantized Noisy Environments." arXiv preprint arXiv:2402.02593 (2024).
```