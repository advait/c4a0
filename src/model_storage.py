import os
import pickle
from typing import List, NewType, Optional, Tuple


from nn import ConnectFourNet
from self_play import Sample


ModelGen = NewType("ModelGen", int)


def load_gen(gen: ModelGen) -> ConnectFourNet:
    """Loads a specific generation of model from the checkpoints directory."""
    gen_dir = os.path.join("checkpoints", str(gen))
    latest_checkpoint = max(
        os.listdir(gen_dir),
        key=lambda cpkt: os.path.getctime(os.path.join(gen_dir, cpkt)),
    )
    model = ConnectFourNet.load_from_checkpoint(
        os.path.join(gen_dir, latest_checkpoint)
    )
    return model


def list_gens() -> List[ModelGen]:
    """List available model gens in ascending order."""
    gens = sorted(os.listdir("checkpoints"))
    return [ModelGen(int(gen)) for gen in gens]


def load_latest_model() -> Tuple[ModelGen, ConnectFourNet]:
    """Loads the latest generation of model from the checkpoints directory."""
    gens = list_gens()
    if len(gens) == 0:
        return ModelGen(0), ConnectFourNet()
    return gens[-1], load_gen(gens[-1])


def load_all_models() -> List[Tuple[ModelGen, ConnectFourNet]]:
    gens = list_gens()
    return [(gen, load_gen(gen)) for gen in gens]


def store_samples(samples: List[Sample], gen: ModelGen):
    """Cache samples for re-use."""
    gen_path = os.path.join("samples", str(gen))
    if not os.path.exists(gen_path):
        os.mkdir(gen_path)
    pkl_path = os.path.join(gen_path, "samples.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(samples, f)


def load_cached_samples(gen: ModelGen) -> Optional[List[Sample]]:
    """Attempts to load previously generated samples to skip self play."""
    pkl_path = os.path.join("samples", str(gen), "samples.pkl")
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)  # type: ignore
