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
- Number of generations trained: 22
- Total training positions: 2,361,492

### Final ELO table

|    | Player  |   ELO    |   Rank |
|---:|:--------|---------:|-------:|
|  0 | gen20   |  1811.67 |      1 |
|  1 | gen4    |  1594.95 |      2 |
|  2 | gen15   |  1563.46 |      3 |
|  3 | gen11   |  1557.35 |      4 |
|  4 | gen22   |  1546.2  |      5 |
|  5 | gen12   |  1543.55 |      6 |
|  6 | gen14   |  1543.14 |      7 |
|  7 | gen13   |  1541.27 |      8 |
|  8 | gen16   |  1532.4  |      9 |
|  9 | gen17   |  1516.62 |     10 |
| 10 | gen10   |  1513.8  |     11 |
| 11 | gen18   |  1510.89 |     12 |
| 12 | gen2    |  1503.38 |     13 |
| 13 | gen19   |  1500.64 |     14 |
| 14 | gen8    |  1496.41 |     15 |
| 15 | gen7    |  1487.36 |     16 |
| 16 | gen6    |  1464.17 |     17 |
| 17 | gen5    |  1443.68 |     18 |
| 18 | gen3    |  1417.44 |     19 |
| 19 | gen21   |  1393.73 |     20 |
| 20 | gen1    |  1385.84 |     21 |
| 21 | uniform |  1385.49 |     22 |
| 22 | gen9    |  1378.7  |     23 |
| 23 | random  |  1367.88 |     24 |

### Training gen 1 from 0. Tournament results:
| Player   |   Score |
|----------|---------|
| gen1     |      46 |
| uniform  |      41 |
| random   |      33 |

### Training gen 2 from 1. Tournament results:
| Player   |   Score |
|----------|---------|
| gen2     |   107.5 |
| gen1     |    45.5 |
| uniform  |    45   |
| random   |    42   |

### Training gen 3 from 2. Tournament results:
| Player   |   Score |
|----------|---------|
| gen2     |   129   |
| gen3     |   113   |
| gen1     |    70.5 |
| random   |    52.5 |
| uniform  |    35   |

### Training gen 4 from 2. Tournament results:
| Player   |   Score |
|----------|---------|
| gen4     |   135   |
| gen2     |    94   |
| gen3     |    92.5 |
| gen1     |    48   |
| random   |    30.5 |

### Training gen 5 from 4. Tournament results:
| Player   |   Score |
|----------|---------|
| gen2     |     100 |
| gen4     |     100 |
| gen3     |     100 |
| gen5     |      70 |
| gen1     |      30 |

### Training gen 6 from 2. Tournament results:
| Player   |   Score |
|----------|---------|
| gen3     |     100 |
| gen6     |     100 |
| gen4     |      80 |
| gen2     |      80 |
| gen5     |      40 |

### Training gen 7 from 3. Tournament results:
| Player   |   Score |
|----------|---------|
| gen4     |     120 |
| gen6     |     100 |
| gen7     |      80 |
| gen3     |      60 |
| gen2     |      40 |

### Training gen 8 from 4. Tournament results:
| Player   |   Score |
|----------|---------|
| gen4     |     120 |
| gen8     |      90 |
| gen6     |      80 |
| gen7     |      70 |
| gen3     |      40 |

### Training gen 9 from 4. Tournament results:
| Player   |   Score |
|----------|---------|
| gen4     |     120 |
| gen8     |      90 |
| gen9     |      80 |
| gen6     |      60 |
| gen7     |      50 |

### Training gen 10 from 4. Tournament results:
| Player   |   Score |
|----------|---------|
| gen8     |     100 |
| gen9     |     100 |
| gen10    |      80 |
| gen4     |      80 |
| gen6     |      40 |

### Training gen 11 from 8. Tournament results:
| Player   |   Score |
|----------|---------|
| gen11    |     130 |
| gen8     |      80 |
| gen9     |      70 |
| gen10    |      60 |
| gen4     |      60 |

### Training gen 12 from 11. Tournament results:
| Player   |   Score |
|----------|---------|
| gen12    |     120 |
| gen11    |     110 |
| gen8     |      80 |
| gen9     |      50 |
| gen10    |      40 |

### Training gen 13 from 12. Tournament results:
| Player   |   Score |
|----------|---------|
| gen12    |     130 |
| gen13    |     120 |
| gen11    |      70 |
| gen8     |      70 |
| gen9     |      10 |

### Training gen 14 from 12. Tournament results:
| Player   |   Score |
|----------|---------|
| gen12    |     110 |
| gen13    |     100 |
| gen14    |      80 |
| gen11    |      80 |
| gen8     |      30 |

### Training gen 15 from 12. Tournament results:
| Player   |   Score |
|----------|---------|
| gen15    |     110 |
| gen11    |      80 |
| gen12    |      80 |
| gen13    |      80 |
| gen14    |      50 |

### Training gen 16 from 15. Tournament results:
| Player   |   Score |
|----------|---------|
| gen15    |     100 |
| gen12    |      80 |
| gen13    |      80 |
| gen16    |      80 |
| gen11    |      60 |

### Training gen 17 from 15. Tournament results:
| Player   |   Score |
|----------|---------|
| gen15    |     110 |
| gen12    |     100 |
| gen16    |      70 |
| gen17    |      70 |
| gen13    |      50 |

### Training gen 18 from 15. Tournament results:
| Player   |   Score |
|----------|---------|
| gen12    |     100 |
| gen15    |     100 |
| gen16    |      70 |
| gen17    |      70 |
| gen18    |      60 |

### Training gen 19 from 12. Tournament results:
| Player   |   Score |
|----------|---------|
| gen12    |      90 |
| gen15    |      90 |
| gen17    |      80 |
| gen19    |      80 |
| gen16    |      60 |

### Training gen 20 from 12. Tournament results:
| Player   |   Score |
|----------|---------|
| gen20    |     120 |
| gen12    |      80 |
| gen17    |      70 |
| gen15    |      70 |
| gen19    |      60 |

### Training gen 21 from 20. Tournament results:
| Player   |   Score |
|----------|---------|
| gen20    |     130 |
| gen15    |      90 |
| gen12    |      80 |
| gen21    |      50 |
| gen17    |      50 |

### Training gen 22 from 20. Tournament results:
| Player   |   Score |
|----------|---------|
| gen20    |     150 |
| gen22    |      80 |
| gen15    |      80 |
| gen12    |      60 |
| gen21    |      30 |
