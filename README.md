# FLAC - numcodecs implementation

[Numcodecs](https://numcodecs.readthedocs.io/en/latest/index.html) wrapper to the 
[FLAC](https://xiph.org/flac/index.html) audio codec using [pyFLAC](https://github.com/sonos/pyFLAC).

This implementation enables one to use WavPack as a compressor in 
[Zarr](https://zarr.readthedocs.io/en/stable/index.html).

## Installation

Install via `pip`:

```
pip install flac-numcodecs
```

Or from sources:

```
git clone https://github.com/AllenNeuralDynamics/flac-numcodecs.git
cd flac-numcodecs
pip install .
```

## Usage

This is a simple example on how to use the `Flac` codec with `zarr`:

```
from flac_numcodecs import Flac

data = ... # any numpy array

# instantiate Flac compressor
flac_compressor = Flac(level=5)

z = zarr.array(data, compressor=flac_compressor)

data_read = z[:]
```
Available `**kwargs` can be browsed with: `Flac?`

**NOTE:** 
In order to reload in zarr an array saved with the `Flac`, you just need to have the `flac_numcodecs` package
installed.