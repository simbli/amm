# PhD Thesis - Code & Publications

[![python version](https://img.shields.io/badge/python-3.11-blue)]()
[![license](https://img.shields.io/badge/license-MIT-green)]()
[![platform](https://img.shields.io/badge/platform-windows-lightgrey)]()

### Risk-aware decision-making & control of autonomous ships
###### Simon Vinding Blindheim 

\
This repository contains code for figures presented in the following article:

- [Autonomous Machinery Management for Supervisory Risk Control Using 
Particle Swarm Optimization (2023)](https://doi.org/10.3390/jmse11020327)

\
See also the 
[articles](https://github.com/simbli/thesis/tree/main/articles)
directory to find each conference and journal article as a PDF.

The complete PhD thesis may be found 
[here](https://hdl.handle.net/11250/3071297).


### Usage
Clone the repository from [GitHub](https://github.com/simbli/thesis):
```
git clone https://github.com/simbli/amm.git
```

Create and activate a Conda environment from the provided YAML file:
```
conda update -c conda-forge conda
conda env create --f env.yml --name amm 
conda activate amm
```

Import the `thesis` package and run the `create` function to generate a figure,
as in the below example:

```python
import thesis

thesis.create('2023amm', 8)
thesis.create('2023amm', 9)
thesis.create('2023amm', 10)
thesis.create('2023amm', 11)
thesis.create('2023amm', 12)
thesis.create('2023amm', 13)
thesis.create('2023amm', 14)
thesis.create('2023amm', 15)
thesis.create('2023amm', 16)
thesis.create('2023amm', 17)
thesis.create('2023amm', 18)
```

See [a2023amm.py](./thesis/figures/a2023amm.py) for the implementations and 
image descriptions of all available figures.

### License

This work uses the [MIT](https://choosealicense.com/licenses/mit/) license.
