# seismo-ml-models-integration
seisan/earthworm integration scripts to process seismology data using machine learning models

## Usage:
```archive_scan.py [-h] [--config CONFIG] [options]```
<br><br><br>
**Optional arguments:**
<br><br>
```-h, --help``` show this help message and exit
<br>
```--config PATH``` path to the config file, default: "config.ini"
<br>
```--verbose VERBOSE``` 0 - non verbose, 1 - verbose
<br>
```--frequency FREQUENCY``` base stream frequency, default: 100 (Hz)
<br>
```--output_file OUTPUT_FILE_NAME``` output text file name, default: "out.txt"
<br>
```--multplt_path MULTPLT_PATH``` path to MULTPLT.DEF file
<br>
```--seisan_path SEISAN_PATH``` path to SEISAN.DEF
<br>
```--model_path MODEL_PATH```path to model file, might be empty with in-code initialized models
<br>
```--weights_path WEIGHTS_PATH``` path to model weights file
<br>
```--start_date START_DATE``` start date in ISO 8601 format:
<br>
&nbsp;&nbsp;&nbsp;&nbsp;```{year}-{month}-{day}T{hour}:{minute}:{second}.{microsecond}```
<br>&nbsp;&nbsp;&nbsp;&nbsp;or<br>
&nbsp;&nbsp;&nbsp;&nbsp;```{year}-{month}-{day}T{hour}:{minute}:{second}```
<br>&nbsp;&nbsp;&nbsp;&nbsp;or<br>
&nbsp;&nbsp;&nbsp;&nbsp;```{year}-{month}-{day}```
<br>
&nbsp;&nbsp;&nbsp;&nbsp;default: yesterday midnight
<br>
```--end_date END_DATE```   end date in ISO 8601 format
<br>
```--threshold THRESHOLD``` model prediction threshold