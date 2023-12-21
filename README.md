# I2OT_energy
I2OT Hackathon

The goal of this project is to use energy data to display the energy flow in a house.
Furthermore, we want to use the data to predict the energy consumption of the house.

## Requirements
In order to run the project, you need to install the required packages by running the following command:
```
pip install -r requirements.txt
```


## Frameworks and libraries
The project is based on the following frameworks and libraries:
- influxdb_client
- pandas
- paho.mqtt
- logging
- TODO: Prognose


## Getting started
In order to run the project, you need to install the required libraries, e.g. set up an environment using
```python -m venv /path/to/new/virtual/environment```.
Afterwards, start the influxdb server and the mqtt broker.
The mqtt data is stored in the influxdb database and thus, a docker container has to be started to run the mqtt_service.py file.
The docker image has to be created by running the following command in the terminal when being in this folder:
```sudo docker build -t mqtt-image .```
Define the name of the image by replacing __mqtt-image__.
The docker container can be started by running the following command:
``` run sudo docker run --network host -d -e INFLUXDB_TOKEN=TOKEN -it --rm mqtt-image```
Replace the token of the influxdb database instead of __TOKEN__.
Omit __-d__ if you want to see the output of the mqtt_service.py file.
Since the logging library is used, the output is also stored in the log file.
The log file is stored in the __logs__ folder (the files may be invisible on IOS systems- use __ls -la__ in the terminal to display them).


## Additional information
The data, i.e. the influxdb server, is stored on a raspberry pi 4.
The repository is pulled to the rasberry pi and the project is run, i. e. the docker image is started etc., on the raspberry pi.
The raspberry pi is connected to the local network and thus, can be accessed by other devices in the network via __ssh pi@IPADDR__ 
(replace __IPADDR__ by the IP address of the raspberry pi in the local network).


### Database
The data of the mqtt broker is stored in the influxdb database.
The database is used for time series data.

### Visualization
The data is visualized using Grafana.

### Prognosis
Both a CNN and an LSTM were used to create a prognosis on future energy data.
Some of the results are displayed below:
![alt text](https://github.com/klaraGtknst/I2OT/blob/main/15min_prediction_lstm.png?raw=true)
![alt text](https://github.com/klaraGtknst/I2OT/blob/main/CNN_15min.png?raw=true)
![alt text](https://github.com/klaraGtknst/I2OT/blob/main/CNN_entire_day.png?raw=true)

## License
MIT License

Copyright (c) [2023] [IO2T Energy]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.