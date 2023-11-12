import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd
from constants import *

URL="http://10.23.101.74:8086"
TOKEN="6AS2vsZ0AZoW-HbG-kXrjR10Uk_x0nZ08ozDlsr-Jm4m-szn1alDVkgzU3B62i_rSnVz8Eck57hXr8G98CEf5Q=="

class influxConnector:
    """
    Influx Connector for the forecast service.
    
    Gives read access to the influx database to get the last data point of a device.
    """
    def __init__(self):
        self.client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
        self.query_api = self.client.query_api()
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        
    def get_data_between(self, start, end, device_id=None, _messurement=None):
        """
        Get data between two timestamps for a device.
        
        :param start: start timestamp
        :param end: end timestamp
        :param device_id: device id
        :param _messurement: messurement name
        :return: list of data points
        """
        prev_filter = False
        query = 'from(bucket: "energydata") |> range(start: ' + str(start) + ', stop: ' + str(end) + ')'
        if any([device_id, _messurement]):
            query += ' |> filter(fn: (r) => '
        if device_id is not None:
            query += 'r.device_id == "' + device_id + '"'
            prev_filter = True
        if _messurement is not None:
            if prev_filter:
                query += ' and '
            query += 'r._measurement == "' + _messurement + '"'
            prev_filter = True
        if any([device_id, _messurement]):
            query += ')'
        query += '|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
        result = self.query_api.query_data_frame(query)
        return result
    
    def get_latest_data(self, device_id=None, _messurement=None, interval=None):
        """
        Get latest data point for a device.
        
        :param device_id: device id
        :param _messurement: messurement name
        :param interval: interval to go back in time
        :return: latest data point
        """
        prev_filter = False
        query = 'from(bucket: "energydata") |> range(start: -'
        query += str(interval) if interval is not None else '5m'
        query += ')'
        if any([device_id, _messurement]):
            query += ' |> filter(fn: (r) => '
        if device_id is not None:
            query += 'r.device_id == "' + device_id + '"'
            prev_filter = True
        if _messurement is not None:
            if prev_filter:
                query += ' and '
            query += 'r._measurement == "' + _messurement + '"'
            prev_filter = True
        if any([device_id, _messurement]):
            query += ')'
        query += '|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
        result = self.query_api.query_data_frame(query)
        return result

if __name__ == '__main__':
    influx = influxConnector()
    #print(influx.get_data_between(1614556800000000000, 1614643200000000000, 'inverter_001'))
    #print(influx.get_data_between(1614556800000000000, 1614643200000000000))