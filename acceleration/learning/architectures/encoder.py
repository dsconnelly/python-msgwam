from .base import BaseNet

class Encoder(BaseNet):

    @property
    def _inputs(self) -> list[str]:
        """
        
        """

        return ['R']

    @property
    def _output(self) -> str:
        """
        
        """

        return 'Z'