# seismo-ml-models-integration
seisan/earthworm integration scripts to process seismology data using machine learning models

## Quick Example
```
git clone https://github.com/Syler1984/seismo-ml-models-integration
cd seismo-ml-models-integration
python archive_scan.py
```

## Configuring stations

### MULPLT.DEF
List of stations to scan is read from *MULPLT.DEF* file. One such file is located in 
 `data/MULPLT.DEF` and used as default option by the *archive_scan.py*.
To specify path to specific use `--mulplt PATH` option.

Only lines read from the *MULPLT.DEF* file are entries like this:

```
#DEFAULT CHANNEL NYSH EN Z
```

One line represents one channel (or one daily archive file) for a seismic station.

`#DEFAULT CHANNEL ` - is necessary to find the station entry, station name and channel are used to locate an archive.

Note, that station also must be described in *SEISAN.DEF*.

```
#DEFAULT CHANNEL NYSH   EN Z
#DEFAULT CHANNEL NYSH   EN N
#DEFAULT CHANNEL NYSH   EN E
#DEFAULT CHANNEL LNSK   EN Z
#DEFAULT CHANNEL LNSK   EN N
#DEFAULT CHANNEL LNSK   EN E
```

### Config file

In *archive_scan.py* .ini config file, provided via `--config PATH` option,
station-specific options (eg filtering, output, channel order, ...) might be written.

In order to configure individual station, simply write a new section with station name.
Example of station section can be found in `data/config.ini`:

```
[ARGI]
no-filter = true
out = predictions_argi.txt
```