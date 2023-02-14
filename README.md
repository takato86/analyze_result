# Analysis of learning results
This repository includes the program aggregating the results of reinforcement learning agents and visualizing them for my research. 
## Requirements

```
pip install -r requirements.txt
```

## Preparing data

### Navigation in a four-room domain

Please write file patterns with the rules used by the Unix shell in `fourrooms/config.json`.
The files is output to `steps` directory by the learning program of four-room domain. The path pattern example is `out/dynamic_human_20221024_1650/steps/*`.

### Navigation in a pinball domain

Please write file patterns with the rules used by the Unix shell in `pinball/config.json`.
The files is output to `steps` directory by the learning program of pinball domain. The path pattern example is `out/dynamic_human_v3/steps/*.csv`.

### Fetch robot picks and places

Befor running the analyzing program, you must export a csv file from tensorboard output files.
Please modify file paths in `transform.py`.

```
cd pick_and_place
python transform.py
```

## How to run

### Navigation in a four-room domain

```
cd fourrooms
python analyze.py 
```


### Navigation in a pinball domain

```
cd pinball
python analyze.py
```

### Fetch robot picks and places

```
cd pick_and_place
python analyze.py
```