import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from paho.mqtt import client as mqtt_client
from constants import *
import json


def connect_mqtt() -> mqtt_client:
    '''
    Connect to MQTT Broker.
    Broker address and port are defined in constants.py.
    :return: mqtt_client
    '''
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)
    print(f"Connecting to MQTT Broker {BROKER}:{PORT} with Client ID: {CLIENT_ID}")
    client = mqtt_client.Client(CLIENT_ID)
    client.on_connect = on_connect
    client.connect(BROKER, PORT)
    return client

def write_data(write_api, device_topic:str, payload:dict):
    '''
    :param write_api: influxdb_client.write_api
    :param device_topic: device name from MQTT message, will be converted to device_id
    :param payload: dict
    Write data to InfluxDB.
    The parameters bucket and organization are defined in constants.py.
    '''
    device_id = TOPIC_LOOKUP[device_topic]
    device_type = DEVICE_TYPE[device_id]
    device_name = DEVICE_LOOKUP[device_id]

    print(f"Device: {device_id}-{device_name}-{device_type} send data: {payload}")
    
    point = influxdb_client.Point(payload["FeatureType"]).tag("device_id", device_id).tag("device_type", device_type).tag("device_name", device_name).time(time.time_ns(), WritePrecision.NS)

    del payload["FeatureType"]

    for key in payload.keys():
        point.field(key, payload[key])
    write_api.write(BUCKET, ORG, point)

def subscribe(client: mqtt_client, write_api):
    '''
    :param client: mqtt_client
    :param write_api: influxdb_client.write_api
    Subscribe to all MQTT topics (inverter and meter).
    '''
    def on_message(client, userdata, msg, write_api=write_api):
        try:
            write_data(write_api=write_api, device_topic=msg.topic.split('/')[1], payload=json.loads(msg.payload.decode()))
        except Exception as e:
            print(e)
            print(f"Message: {msg.topic} {msg.payload.decode()}")
            pass
        
    def subscribe_to_specfic_topic(client, device_type:str):
        try:
            for device_topic in list(MQTT_TOPICS[device_type].values()):
                for subtopic in list(MQTT_SUBTOPICS[device_type].values()):
                        topic = f"{BASETOPIC}/{device_topic}/{subtopic}"
                        try:
                            client.subscribe(topic)
                        except Exception as e:
                            print(e)
                            print(f"Error subscribing device: {device_topic} with topic: {topic}")
                            pass
        except Exception as e:
            print(e)
            print(f"Error subscribing to {device_type}")
            pass

    subscribe_to_specfic_topic(client=client, device_type="meter")
    subscribe_to_specfic_topic(client=client, device_type="inverter")
    
    client.on_message = on_message



def connect_influxdb():
    '''connect to influxdb'''
    write_client = influxdb_client.InfluxDBClient(url=URL, token=TOKEN, org=ORG)
    write_api = write_client.write_api(write_options=SYNCHRONOUS)
    return write_api


def run():
    '''
    Connect to MQTT Broker, connects, subscribes and writes to InfluxDB.
    Is called once at the start of the script.
    '''
    mqtt_client = connect_mqtt()
    mqtt_client.loop_start()
    write_api = connect_influxdb()
    subscribe(mqtt_client, write_api)

if __name__ == '__main__':
    run()
    exit_str = ""
    while exit_str not in ['q', 'quit']:
        exit_str = input("Enter 'q' or 'quit' to exit.\n")