import datetime
import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd
from constants import *


class influxConnector:
    """
    Influx Connector for the forecast service.
    
    Gives read access to the influx database to get the last data point of a device.
    """
    def __init__(self):
        self.client = InfluxDBClient(url=URL, token=TOKEN, org=ORG)
        self.query_api = self.client.query_api()
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

    def __add_filter(self, device_id: str, _messurement: str, _field:str, query: str):
        '''
        Adds filter to the query. Auxiliary function for get_data_between and get_latest_data.

        :param device_id: device id
        :param _messurement: messurement name
        :param _field: _field name
        :param query: query string

        :return: query string with filter
        '''
        filter2attr = ['r.device_id', 'r._measurement', 'r._field']
        filter = [f'{filter2attr[i]} == "' + filter_name + '"' for i, filter_name in enumerate([device_id, _messurement, _field]) if filter_name is not None]
        filter_query = ' and '.join(filter)
        if any([device_id, _messurement, _field]):
            query += ' |> filter(fn: (r) => {}'.format(filter_query) + ')'

        return query
        
    def get_data_between(self, start, end, device_id=None, _messurement=None, _field=None):
        """
        Get data in a time range, for instance, "-24h", "0h",  for a device.
        
        :param start: start time of type string defines the number of hours before the current time, e.g. "-24h"
        :param end: end time of type string defines the number of hours before the current time, e.g. "0h"
        :param device_id: device id
        :param _messurement: messurement name
        :param _field: _field name
        :return: list of data points
        """
        query = 'from(bucket: "energydata") |> range(start: ' + str(start) + ', stop: ' + str(end) + ')'

        query = self.__add_filter(device_id=device_id, _messurement=_messurement, _field=_field, query=query)
        
        query += '|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
        result = self.query_api.query_data_frame(query)
        return result
    
    def get_latest_data(self, device_id=None, _messurement=None, interval=None, _field=None):
        """
        Get latest data point for a device.
        
        :param device_id: device id
        :param _messurement: messurement name
        :param interval: interval to go back in time
        :param _field: _field name
        :return: latest data point
        """
        query = 'from(bucket: "energydata") |> range(start: -'
        query += str(interval) if interval is not None else '5m'
        query += ')'

        query = self.__add_filter(device_id, _messurement, query=query, _field=_field)

        query += '|> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'
        result = self.query_api.query_data_frame(query)
        return result
    
    def get_last_x_hours_diff(self, x:int=6, device_id:str='meter_004', _messurement:str='EnergyMeter', tag:str='EnergyOut'):
        '''
        Queries for last x hours and returns (and saves) the difference between two sequential values.
        
        :param x: number of hours to query
        :return: dataframe with difference between two sequential values WITHOUT timestamps
        '''
        df = influx.get_data_between(f"-{x}h", "0h", device_id=device_id, _field=tag).loc[:, [tag]].diff()
        df.to_csv(f'csv_exports/{datetime.date.today()}{datetime.datetime.now().strftime("%H-%M-%S")}.csv')
        return df

if __name__ == '__main__':
    influx = influxConnector()

    # print(influx.get_latest_data(device_id='inverter_001', _messurement='Battery'))
    # print(influx.get_latest_data(_messurement='Battery'))
    # print(influx.get_latest_data(device_id='inverter_001'))
    # print(influx.get_latest_data())

    print(influx.get_last_x_hours_diff(x=6))
    # print(influx.get_data_between("-24h", "0h", device_id='inverter_001', _messurement='EnergyMeter'))
    # print(influx.get_data_between("-24h", "0h", device_id='inverter_001'))
    # print(influx.get_data_between("-24h", "0h", _messurement='Battery'))
    # print(influx.get_data_between("-24h", "0h"))