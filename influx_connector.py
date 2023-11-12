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
        query = 'from(bucket: "energydata") |> range(start: ' + str(start) + ', stop: ' + str(end) + ')'

        query = self.add_filter(device_id, _messurement, query)
        
        query += '|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
        result = self.query_api.query_data_frame(query)
        return result

    def add_filter(self, device_id: str, _messurement: str, query: str):
        '''
        Adds filter to the query. Auxiliary function for get_data_between and get_latest_data.

        :param device_id: device id
        :param _messurement: messurement name
        :param query: query string

        :return: query string with filter
        '''
        filter2attr = ['r.device_id', 'r._measurement']
        filter = [f'{filter2attr[i]} == "' + filter_name + '"' for i,filter_name in enumerate([device_id, _messurement]) if filter_name is not None]
        filter_query = ' and '.join(filter)
        if any([device_id, _messurement]):
            query += ' |> filter(fn: (r) => {}'.format(filter_query) + ')'

        return query
    
    def get_latest_data(self, device_id=None, _messurement=None, interval=None):
        """
        Get latest data point for a device.
        
        :param device_id: device id
        :param _messurement: messurement name
        :param interval: interval to go back in time
        :return: latest data point
        """
        query = 'from(bucket: "energydata") |> range(start: -'
        query += str(interval) if interval is not None else '5m'
        query += ')'

        query = self.add_filter(device_id, _messurement, query)

        query += '|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
        result = self.query_api.query_data_frame(query)
        return result

if __name__ == '__main__':
    influx = influxConnector()
    # works:
    # print(influx.get_latest_data(device_id='inverter_001', _messurement='Battery'))
    # print(influx.get_latest_data(_messurement='Battery'))
    # print(influx.get_latest_data(device_id='inverter_001'))
    # print(influx.get_latest_data())

    # works:
    print(influx.get_data_between(1614556800000000000, 1614643200000000000, device_id='inverter_001', _messurement='Battery'))

    # does not works:
    # print(influx.get_data_between(1614556800000000000, 1614643200000000000, device_id='inverter_001'))
    # print(influx.get_data_between(1614556800000000000, 1614643200000000000, _messurement='Battery'))
    # print(influx.get_data_between(1614556800000000000, 1614643200000000000, 'inverter_001', 'Battery'))
    #print(influx.get_data_between(1614556800000000000, 1614643200000000000))