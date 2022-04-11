# seismo-ml-models-integration
seisan/earthworm integration scripts to process seismology data using machine learning models

## Quick Example
```
git clone https://github.com/Syler1984/seismo-ml-models-integration
cd seismo-ml-models-integration
pip install --user -r requirements.txt
python archive_scan.py --start YYYYMMDD --end YYYYMMDD
```

Note: required `python == 3.8`

## Quick setup guide

### Install the programm
```
git clone https://github.com/Syler1984/seismo-ml-models-integration
cd seismo-ml-models-integration
pip install --user -r requirements.txt
```

### Specify archives to scan
List of archive stations to scan is read from `MULPLT.DEF` file. Default search path is `data/MULPLT.DEF` but if
file is not found there, program will look up `$SEISAN_TOP/DAT/MULPLT.DEF`.

To set a custom list of archive channels for a scan, you can create a `data/MULPLT.DEF`. Example can be found
in `data/EXAMPLE_MULPLT.DEF`.

### Run `archive_scan.py`

This will run a scan using Seismo-Performer model for the October 1st 2021:
```
python archive_scan.py --start 20211001 --end 20211002
```

Date format is `YYYYMMDD` or `YYYYMMDDTHHmmss`, e.g. running a scan for a two hours window:
```
python archive_scan.py --start 20211001T10:00:00 --end 20211002T12:00:00
```

### Using different NN models

Three models are supported by default and can be specified via command-line options:

`--cnn` - Seismo CNN model. <br>
`--gpd` - GPD (Generalized Phase Detection) or ConvNet. <br>
*no model flag* or `--favor` - Seismo-Performer.

It is also possible to provide a custom model, see more at [Custom models](#custom-models) section. Or see at 
[Model configuration](#model-configuration) at how to set models through config file, including usage of different
models for different stations.


## Full setup guide

### Configuring the stations 
#### MULPLT.DEF
Default search path is `/data/MULPLT.DEF`. You can supply different path through `--mulplt-def <path>` option, or
by using configuration file option:
```
mulplt-def = <path>
```
If no `MULPLT.DEF` file found, program will look for `$SEISAN_TOP/DAT/MULPLT.DEF` file instead.

Note, that neither name `MULPLT.DEF` or extension `.DEF` are specifically required, you can provide any
file as long as it has station channels definitions like following:

```
DEFAULT CHANNEL NYSH EN Z
```

`DEFAULT CHANNEL` serves as an indicator for a channel to scan, followed by a station name and 
a component information. Unlike standard `MULPLT.DEf`, precise character positions does not matter as long as words
are separated by any amount of whitespaces.

#### Channel order

Default Seismo-Performer, Seismo-CNN and GPD models require three-channel input. Channel order can be configured
by using either `--channel-order <channels>` command line option, or through
```
channel-order = <channels>
```
`<channels>` is a string (without whitespaces) of separate channel order arrangements, each  arrangement consists
of components separated by a comma, e.g. `N,E,Z`. Arrangements are separated by a semi-column, e.g. 
`N,E,Z;1,2,Z`.

Later, station archives are passed to a NN model in the first channel order, which can describe 
specified station archives.

For example, with `MULPLT.DEF` being:

```
DEFAULT CHANNEL STAT1 SH Z
DEFAULT CHANNEL STAT1 SH N
DEFAULT CHANNEL STAT1 SH E

DEFAULT CHANNEL STAT2 EH 1
DEFAULT CHANNEL STAT2 EH 2
DEFAULT CHANNEL STAT2 EH Z

DEFAULT CHANNEL STAT3 EH Z
```

And `channel-order = N,E,Z;1,2,Z;Z,Z,Z`

STAT1 would be passed to NN model in order N, E and Z. STAT2 would be passed in order 1, 2 and Z.
And a single STAT3 channel would be tripled and passed to a model (order Z, Z, Z).

Note, that if you flip order of arrangements like that: `channel-order = Z,Z,Z;N,E,Z;1,2,Z`, than 
STAT1, STAT2 and STAT3 all be passed as Z, Z and Z, because `Z,Z,Z` arrangement would have the highest priority and
all three stations fit that arrangement.

Default channel-order is `N,E,Z;1,2,Z;Z,Z,Z`, so there is no need to specify it, unless extra arrangements 
are required.

Note, that all default models trained on data with Z channal being the last one, so it is recommended to keep that
order in custom arrangements.

## Detailed configuration

In *archive_scan.py* .ini config file, provided via `--config PATH` option,
station-specific options (eg filtering, output, channel order, ...) might be written.

In order to configure individual station, simply write a new section with station name.
Example of station section can be found in `data/config.ini`:

```
[ARGI]
no-filter = true
out = predictions_argi.txt
```

### Model configuration
WIP

## Custom models
WIP

## Advanced search
With flag `--advanced-search` (or option `advanced-search = true` in config file) enabled, 
for every _detected event_ (_detected event_ is a combination of closely packed detection with number of events greater or equal 
`detections-for-event` parameter value) additional scan will be performed.
This scan could have different threshold and window shift to extract more detailed information about
event.

Note: scan will be performed only on stations on which detections were found (you can change that behaviour).

List of related command line options / parameters:
- `--advanced-search` - enable advanced search
- `--advanced-search-range <number>` - range of search (in seconds) around detected events, default: 30
- `--advanced-search-threshold <threshold>` - threshold for advanced search, could be set just like regular `--threshold`:
with number from 0 to 1.0 or for _p_ and _s_ labels individually, default: 0.9
- `--advanced-search-shift <number>` - window shift for advanced search (in samples), default: 2
- `--advanced-search-combine` - if specified (or set in config: `advanced-search-combine = true`) 
will combine all detections in advanced search as single event, otherwise will use normal event combination
method. Without this option enabled, if `advanced-search-range` is larger than `combine-events-range`, advanced search
for a single event could potentially yeild multiple events.
- `--advanced-search-all-stations` - with this option enabled the scan will be performed on full list
of stations (same as for the original scan), otherwise, only stations with detected wave arrivals will be used for 
the advanced search.