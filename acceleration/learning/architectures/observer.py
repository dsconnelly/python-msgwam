from .base import BaseNet

class Observer(BaseNet):

    @property
    def _inputs(self) -> list[str]:
        """
        
        """

        return ['Z']
    
    @property
    def _output(self) -> str:
        """
        
        """

        return 'F'