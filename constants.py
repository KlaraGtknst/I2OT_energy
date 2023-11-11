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

# MQTT Topics
MQTT_TOPICS = {
        "meters": {
            "meter001": "shellyem3-C8C9A3705CFC",
            "meter002": "shellyem3-E89F6D84968A",
            "meter003": "SMA 015D.1901407994",
            "meter004": "SMA 0174.3017198275",
        "inverters": {
            "inverter001": "SMA 017A.3017445303",
            "inverter002": "SMA 01B8.3015653785"
        }
    }
}