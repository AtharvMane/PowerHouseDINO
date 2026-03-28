# Review
This document contains the review of the trainer written to critique and suggest improvements for productionization and better experimenation experience.

# Hard Coding
It is quite evident that the biggest issue plaguing this implementation is the amount of hard-coded variables. When running large scale experiments, it is important to have important hyperparameters stored in a single place. Typically a config file. The code, including that for the trainer as well as the architecture could be improved a lot simply by making architecture hyperparameters and trainer parameters as arguments to the model/the trainer in the `__init__` and then passing these arguments as `CONSTANTS` taken from the config. 

# Modularity
The code lacks modularity, especially with respect to data handling. A single file contains both the dataset class definition as well as the definition for the model. Ideally thse should be placed in completely differnt files and be instantialted in either the `def main()` function or `if __name__=='__main__'` block, then passed to the trainer as training arguments.

Dataloading, especially preprocessing and model instantiation should not be a part of the trainer. They should be created outside the trainer and then passed in as an argument. This is specifically true for using different architectures and datasets, the current implementation allows only for a specific assumption of dataset format (.csv) and only a specific architecture (even with the hyper parameter specifications). This implementation does not give experimentor the freedom to choose their own datasets/model. Ideally, the dataset class should be defined in an entirely different file.

Moreover, the dataset is processed as self implemented numpy code. It is much better to process the dataset using pytorch modules, since they're generally device optimized as well as have numerically stable properties. They also result in very low `code-reproduction` issues.

The dataloaders also don't seem to use the `pin_memory` argument or the `num_workers` argument that better helps with optimizing for execution speed and experimentation speed.

# Training Loop
The training loop assumes that the validation run only happens per epoch. This is a dangerous assumption for larger datasets. In case of very large datasets, it is better to set evaluation frequency based on steps/percentage of the total run instead of epochs. Data logging should also be made `step-wise` instead of `epoch-wise`. This change is required to discard unstable, non-convergent or divergent models from the very beginning, thus saving precious compute and time.

The logging of the run could be handled better. It is necessary to log training config along with the loss/performance data to be able to distinguish between experiments.


Instead of using matplotlib and simple array to log training loss, it is much better to use backages like `TensorBoard` or `WeightsAndBiases (wandb)` for this purpose.

It is also better in log `grad_norm` along with training loss to check for numerical instability or `gradient_explosions`

# Comments/Docstring
The code severely lacks either comments or docstring. at least one of these is necessary for collaborative effort.

# Good stuff
While the code does require more modularity by separating data-pre-processing, loading and model definitions, the structure for classes for the dataset as well model is well written. The code does provide most of the functionality needed for very quick prototyping of models.

# Slight correction
`self.model.to(self.device)` needs to be changed to `self.model = self.model.to(self.device)`. This line does not put the `self.model` instance onto the device. 