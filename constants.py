import os

# MQTT Client
BROKER = 'ts.ewbk.de'
PORT = 1883
TOPIC = "hack/SMA 01B8.3015653785/TotalActivePower"
CLIENT_ID = f'raspi4-energymanagement'

# InfluxDB Client
TOKEN = os.environ.get("INFLUXDB_TOKEN")
ORG = "I2OT_energy"
URL = "http://127.0.0.1:8086"
BUCKET="energydata"