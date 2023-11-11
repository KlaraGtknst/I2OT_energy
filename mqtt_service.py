import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from paho.mqtt import client as mqtt_client

# MQTT Client
broker = 'ts.ewbk.de'
port = 1883
topic = "hack/SMA 01B8.3015653785/TotalActivePower"
client_id = f'raspi4-energymanagement'

# InfluxDB Client
token = os.environ.get("INFLUXDB_TOKEN")
org = "I2OT_energy"
url = "http://127.0.0.1:8086"
bucket="energydata"

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)
    # Set Connecting Client ID
    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):
        print(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")

    client.subscribe(topic)
    client.on_message = on_message

def connect_influxdb():
    write_client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)
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