from typing import Callable

def by_kind(func: Callable) -> Callable:
    """
    Wrap a plotting function that takes a `kind` specifier so that if `'all'` is
    passed, all three `kind` options are called and plotted.
    """

    def wrapped(kind: str, *args, **kwargs):
        """Wrapped function to parse `kind` specifiers."""

        kinds = ['wind', 'flux', 'acceleration'] if kind == 'all' else [kind]
        for k in kinds: func(k, *args, **kwargs)

    return wrapped
