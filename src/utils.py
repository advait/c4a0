from typing import List, Tuple, TypeVarTuple

Ts = TypeVarTuple("Ts")
Ls = TypeVarTuple("Ls")


def unzip(tuples: List[Tuple[*Ts]]) -> Tuple[*Ls]:
    """Unzips a list of tuples into a tuple of lists."""
    if len(tuples) == 0:
        return ((),)

    ret = tuple([] for _ in range(len(tuples[0])))
    for t in tuples:
        for i, item in enumerate(t):
            ret[i].append(item)

    return ret
