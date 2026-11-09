from pathlib import Path

import numpy as np
import pytest

from c4a0.labels import (
    ILLEGAL,
    Labels,
    Moves,
    Recipe,
    Split,
    format_labels,
    parse_labels,
    read_labels,
    write_labels,
)

X = int(ILLEGAL)
RECIPE = Recipe(version=1, split=Split.DEV, seed=1, per_ply=64, plies=(0, 41))

# Solver rows, moves 0-indexed (recorded from Pons's `c4solver -a`, which the native solver matches).
SOLVER_ROWS = {
    "": [-2, -1, 0, 1, 0, -1, -2],
    "3344": [-4, -3, 18, 2, 2, 18, -3],
    "000000": [X, -2, 0, 1, -2, 0, -2],
    "01010": [1, -18, -18, -18, -18, -18, -18],
}


def make_labels(rows: dict[str, list[int]], recipe: Recipe = RECIPE) -> Labels:
    return Labels(
        recipe=recipe,
        moves=tuple(Moves(m) for m in rows),
        scores=np.array(list(rows.values()), dtype=np.int8).reshape(-1, 7),
    )


def test_derived_facts_from_solver_rows() -> None:
    labels = make_labels(SOLVER_ROWS)
    assert labels.best.tolist() == [
        [False, False, False, True, False, False, False],
        [False, False, True, False, False, True, False],
        [False, False, False, True, False, False, False],
        [True, False, False, False, False, False, False],
    ]
    assert labels.value.tolist() == [1, 1, 1, 1]
    assert labels.legal[2].tolist() == [False] + [True] * 6
    assert labels.ply.tolist() == [0, 4, 6, 5]
    assert labels.trivial.tolist() == [False] * 4


def test_trivial_rows_and_values() -> None:
    labels = make_labels(
        {
            "0": [-2] * 7,  # every move loses equally
            "1": [0, 0, 0, -1, 0, 0, 0],
            "2": [X, X, X, X, X, X, -1],  # one legal move
        }
    )
    assert labels.trivial.tolist() == [True, False, True]
    assert labels.value.tolist() == [-1, 0, -1]


def test_buckets_are_six_plies_wide() -> None:
    labels = make_labels(
        {"0" * 5: [0] * 7, "012345": [0] * 7, "0" * 3 + "1" * 6 + "2" * 3: [0] * 7}
    )
    assert labels.bucket.tolist() == [0, 1, 2]


def test_round_trip_is_byte_identical(tmp_path: Path) -> None:
    labels = make_labels(SOLVER_ROWS)
    path = tmp_path / "dev.tsv"
    write_labels(path, labels)
    text = path.read_text()
    assert text.splitlines()[:3] == [
        '# c4a0-labels {"version":1,"split":"dev","seed":1,"per_ply":64,"plies":[0,41]}',
        "moves\tscores",
        "\t-2,-1,0,1,0,-1,-2",
    ]
    assert "000000\tx,-2,0,1,-2,0,-2" in text.splitlines()
    again = read_labels(path)
    assert again.moves == labels.moves and again.recipe == labels.recipe
    np.testing.assert_array_equal(again.scores, labels.scores)
    assert format_labels(again) == text


@pytest.mark.parametrize(
    "body",
    [
        "3\t1,2,3\n",
        "3\t1,2,3,4,5,6,a\n",
        "9\t1,2,3,4,5,6,7\n",
        "3\tx,x,x,x,x,x,x\n",
        "3\t1,2,3,4,5,6,99\n",
    ],
)
def test_parse_rejects_malformed_rows(body: str) -> None:
    header = format_labels(make_labels({}))
    with pytest.raises(ValueError):
        parse_labels(header + body)
