from typing import Type, Union

from analogvnn.nn.module.FullSequential import FullSequential
from analogvnn.nn.noise.GaussianNoise import GaussianNoise
from analogvnn.nn.normalize.Normalize import Normalize
from analogvnn.nn.precision.ReducePrecision import ReducePrecision
from analogvnn.nn.precision.StochasticReducePrecision import StochasticReducePrecision


class WeightModel(FullSequential):
    def __init__(
            self,
            norm_class: Type[Normalize] = None,
            precision_class: Type[Union[ReducePrecision, StochasticReducePrecision]] = None,
            precision: Union[int, None] = None,
            noise_class: Type[Union[GaussianNoise]] = None,
            leakage: Union[float, None] = None,
    ):
        super(WeightModel, self).__init__()
        self.norm_class = norm_class
        self.precision_class = precision_class
        self.precision = precision
        self.noise_class = noise_class
        self.leakage = leakage

        self.all_layers = []

        if norm_class is not None:
            self.all_layers.append(norm_class())
        if precision_class is not None:
            self.all_layers.append(precision_class(precision=precision))
        if noise_class is not None:
            self.all_layers.append(noise_class(leakage=leakage, precision=precision))

        self.eval()
        if len(self.all_layers) > 0:
            self.add_sequence(*self.all_layers)

    def hyperparameters(self):
        return {
            'weight_model_class': self.__class__.__name__,

            'norm_class_w': self.norm_class.__name__ if self.norm_class is not None else str(None),
            'precision_class_w': self.precision_class.__name__ if self.precision_class is not None else str(None),
            'precision_w': self.precision,
            'noise_class_w': self.noise_class.__name__ if self.noise_class is not None else str(None),
            'leakage_w': self.leakage,
        }
