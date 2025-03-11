from .base import BaseNet

class Stepper(BaseNet):
    """
    A `Stepper` takes the latent representation at the current time step, along
    with the mean wind and source ray volume information, and predicts the
    latent representation at the next time step.
    """

    @property
    def _inputs(self) -> list[str]:
        """
        The `Stepper` makes predictions from the latent space, but also includes
        information from the mean state and the wave source.
        """

        return ['u', 'S', 'Z']
    
    @property
    def _outputs(self) -> str:
        """The `Stepper` predicts in the latent space."""
        return 'Z'