# c4a0: Connect Four Alpha-Zero

An alpha-zero-style Connect Four neural network trained via self play.
- c4.py: Connect four game logic
- nn.py: Neural net architecture
- mcts.py: Monte carlo tree search implementation
- self_play.py: Multiprocessing self play training sample generation
- training_py: Generational training implementation
- tournament.py: Round-robin multi-model tournament implementation

## Usage

`poetry install --no-root` to install deps.

`poetry run src/main.py train` to train:
- Stores training state in training/ folder

`poetry run src/main.py tournament --gen-id 22 --gen-id 21 --gen-id 18`:
- Runs a tournament with the given generations

## Results
After training for ~12 hours with an RTX 3090, we achieved the following results:
- Total training positions: 2,361,492
- Number of generations trained: 22

### Training gen 1 from 0. Tournament results:
```
gen1: 46.0
uniform: 41.0
random: 33.0
```

### Training gen 2 from 1. Tournament results:
```
gen2: 107.5
gen1: 45.5
uniform: 45.0
random: 42.0
```

### Training gen 3 from 2. Tournament results:
```
gen2: 129.0
gen3: 113.0
gen1: 70.5
random: 52.5
uniform: 35.0
```

### Training gen 4 from 2. Tournament results:
```
gen4: 135.0
gen2: 94.0
gen3: 92.5
gen1: 48.0
random: 30.5
```

### Training gen 5 from 4. Tournament results:
```
gen2: 100.0
gen4: 100.0
gen3: 100.0
gen5: 70.0
gen1: 30.0
```

### Training gen 6 from 2. Tournament results:
```
gen3: 100.0
gen6: 100.0
gen4: 80.0
gen2: 80.0
gen5: 40.0
```

### Training gen 7 from 3. Tournament results:
```
gen4: 120.0
gen6: 100.0
gen7: 80.0
gen3: 60.0
gen2: 40.0
```

### Training gen 8 from 4. Tournament results:
```
gen4: 120.0
gen8: 90.0
gen6: 80.0
gen7: 70.0
gen3: 40.0
```

### Training gen 9 from 4. Tournament results:
```
gen4: 120.0
gen8: 90.0
gen9: 80.0
gen6: 60.0
gen7: 50.0
```

### Training gen 10 from 4. Tournament results:
```
gen8: 100.0
gen9: 100.0
gen10: 80.0
gen4: 80.0
gen6: 40.0
```

### Training gen 11 from 8. Tournament results:
```
gen11: 130.0
gen8: 80.0
gen9: 70.0
gen10: 60.0
gen4: 60.0
```

### Training gen 12 from 11. Tournament results:
```
gen12: 120.0
gen11: 110.0
gen8: 80.0
gen9: 50.0
gen10: 40.0
```

### Training gen 13 from 12. Tournament results:
```
gen12: 130.0
gen13: 120.0
gen11: 70.0
gen8: 70.0
gen9: 10.0
```

### Training gen 14 from 12. Tournament results:
```
gen12: 110.0
gen13: 100.0
gen14: 80.0
gen11: 80.0
gen8: 30.0
```

### Training gen 15 from 12. Tournament results:
```
gen15: 110.0
gen11: 80.0
gen12: 80.0
gen13: 80.0
gen14: 50.0
```

### Training gen 16 from 15. Tournament results:
```
gen15: 100.0
gen12: 80.0
gen13: 80.0
gen16: 80.0
gen11: 60.0
```

### Training gen 17 from 15. Tournament results:
```
gen15: 110.0
gen12: 100.0
gen16: 70.0
gen17: 70.0
gen13: 50.0
```

### Training gen 18 from 15. Tournament results:
```
gen12: 100.0
gen15: 100.0
gen16: 70.0
gen17: 70.0
gen18: 60.0
```

### Training gen 19 from 12. Tournament results:
```
gen12: 90.0
gen15: 90.0
gen17: 80.0
gen19: 80.0
gen16: 60.0
```

### Training gen 20 from 12. Tournament results:
```
gen20: 120.0
gen12: 80.0
gen17: 70.0
gen15: 70.0
gen19: 60.0
```

### Training gen 21 from 20. Tournament results:
```
gen20: 130.0
gen15: 90.0
gen12: 80.0
gen21: 50.0
gen17: 50.0
```

### Training gen 22 from 20. Tournament results:
```
gen20: 150.0
gen22: 80.0
gen15: 80.0
gen12: 60.0
gen21: 30.0
```
