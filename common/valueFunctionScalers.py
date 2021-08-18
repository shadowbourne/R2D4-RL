import torch

class valueFunctionScaler():
    def __init__(self, vfEpsilon=0.001) -> None:
        self.vfEpsilon = vfEpsilon

    # Edited from Google's SEED RL [8] Repo.
    def value_function_rescaling(self, x):
        """Value function rescaling used in R2D2 paper [1], see table 2, or Proposition A.2 in paper "Observe and Look Further" [7]."""
        return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.) - 1.) + self.vfEpsilon * x

    # Edited from Google's SEED RL [8] Repo.
    def inverse_value_function_rescaling(self, x):
        """Inverse of the above function. See Proposition A.2 in paper "Observe and Look Further" [7]."""
        return torch.sign(x) * (
            torch.square(((torch.sqrt(
                1. + 4. * self.vfEpsilon * (torch.abs(x) + 1. + self.vfEpsilon))) - 1.) / (2. * self.vfEpsilon)) -
            1.)