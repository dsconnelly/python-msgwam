from .base import BaseNet

class Stepper(BaseNet):
    
    @property
    def _inputs(self) -> list[str]:
        """
        
        """

        return ['u', 'S', 'Z']
    
    @property
    def _output(self) -> str:
        """
        
        """

        return 'Z'