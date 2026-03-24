from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class OrderingState:
    """
    Immutable MDP state: a partial ordering (prefix tuple of variable indices).

    Attributes:
        order: Tuple of variable indices chosen so far.
        d:     Total number of variables.
    """
    order: Tuple[int, ...]
    d: int

    def is_terminal(self) -> bool:
        return len(self.order) == self.d

    def available_actions(self) -> Tuple[int, ...]:
        """Actions = variables not yet placed in the ordering."""
        used = set(self.order)
        return tuple(v for v in range(self.d) if v not in used)
