import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from paho.mqtt import client as mqtt_client
from constants import *
import json


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

def wirte_data(device_topic, payload):
    device_id = TOPIC_LOOKUP[device_topic]
    device_type = DEVICE_TYPE[device_id]
    device_name = DEVICE_LOOKUP[device_id]

    print(f"Device: {device_id} - {device_type} - {device_name}")
    print(f"Payload: {payload} - {type(payload)}")


def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
        wirte_data( msg.topic.split('/')[1], json.loads(msg.payload.decode()))
    
    def subscribe_to_topic(client:mqtt_client, main_topic, sub_topic):
        topic = f"{BASETOPIC}/{main_topic}/{sub_topic}"
        client.subscribe(topic)
        client.on_message = on_message

    for meter in MQTT_TOPICS["meter"]:
        meter_topic = MQTT_TOPICS["meter"][meter]
        for subtopic_name in MQTT_SUBTOPICS["meter"]:
            subtopic = MQTT_SUBTOPICS["meter"][subtopic_name]
            subscribe_to_topic(client, meter_topic, subtopic)
    
    for inverter in MQTT_TOPICS["inverter"]:
        inverter_topic = MQTT_TOPICS["inverter"][inverter]
        for subtopic_name in MQTT_SUBTOPICS["inverter"]:
            subtopic = MQTT_SUBTOPICS["inverter"][subtopic_name]
            subscribe_to_topic(client, inverter_topic, subtopic)

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