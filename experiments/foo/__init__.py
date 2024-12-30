from . import bar, baz

__all__ = ['bar', 'baz', 'change']

def change(a: str) -> None:
    bar.name = a

