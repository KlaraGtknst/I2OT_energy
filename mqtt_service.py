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

def wirte_data(write_api, device_topic, payload):
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
    def on_message(client, userdata, msg, write_api=write_api):
        try:
            wirte_data(write_api, msg.topic.split('/')[1], json.loads(msg.payload.decode()))
        except Exception as e:
            print(e)
            print(f"Message: {msg.topic} {msg.payload.decode()}")
            pass
    
    def subscribe_to_topic(client:mqtt_client, main_topic, sub_topic):
        try:
            topic = f"{BASETOPIC}/{main_topic}/{sub_topic}"
            client.subscribe(topic)
        except Exception as e:
            print(e)
            print(f"Topic: {topic}")
            pass

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
    
    client.on_message = on_message

def connect_influxdb():
    write_client = influxdb_client.InfluxDBClient(url=URL, token=TOKEN, org=ORG)
    write_api = write_client.write_api(write_options=SYNCHRONOUS)
    return write_api


def run():
    mqtt_client = connect_mqtt()
    mqtt_client.loop_start()
    write_api = connect_influxdb()
    subscribe(mqtt_client, write_api)

if __name__ == '__main__':
    run()
    exit_str = ""
    while exit_str not in ['q', 'quit']:
        exit_str = input("Enter 'q' or 'quit' to exit.\n")