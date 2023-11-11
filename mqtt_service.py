import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from paho.mqtt import client as mqtt_client
from constants import *

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)
    # Set Connecting Client ID
    client = mqtt_client.Client(CLIENT_ID)
    client.on_connect = on_connect
    client.connect(BROKER, PORT)
    return client

def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")

    client.subscribe(TOPIC)
    client.on_message = on_message

def connect_influxdb():
    write_client = influxdb_client.InfluxDBClient(url=URL, token=TOKEN, org=ORG)
    write_api = write_client.write_api(write_options=SYNCHRONOUS)
    return write_api


def run():
    mqtt_client = connect_mqtt()
    mqtt_client.loop_start()
    connect_influxdb()
    subscribe(mqtt_client)

if __name__ == '__main__':
    run()
    while True:
        next