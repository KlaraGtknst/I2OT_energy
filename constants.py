import os

# MQTT Client
BROKER = 'ts.ewbk.de'
PORT = 1883
BASETOPIC = "hack"
CLIENT_ID = f'raspi4-energymanagement'

# InfluxDB Client
TOKEN = os.environ.get("INFLUXDB_TOKEN")
ORG = "I2OT_energy"
URL = "http://127.0.0.1:8086"
BUCKET="energydata"

# MQTT Topics
MQTT_TOPICS = {
    # Zaehler
    "meter": {
        "meter_001": "shellyem3-C8C9A3705CFC",
        "meter_002": "shellyem3-E89F6D84968A",
        "meter_003": "SMA_015D.1901407994",
        "meter_004": "SMA_0174.3017198275",
    },
    # Weschelrichter
    "inverter": {
        "inverter_001": "SMA_017A.3017445303",
        "inverter_002": "SMA_01B8.3015653785"
    }
}

MQTT_SUBTOPICS = {
    "meter": {
        "power": "TotalActivePower"
    },
    "inverter": {
        "power": "TotalActivePower",
        "string_a": "String A",
        "string_b": "String B",
        "battery": "Battery",
    }
}

# InfluxDB Tags
d_swap_list = [{value: key for key, value in MQTT_TOPICS[device].items()} for device in MQTT_TOPICS.keys()]
TOPIC_LOOKUP = dict(pair for d in d_swap_list for pair in d.items())
del d_swap_list

DEVICE_LOOKUP = {
    "meter_001": "buero",
    "meter_002": "wohnung01",
    "meter_003": "keller",
    "meter_004": "hausanschluss",
    "inverter_001": "sunnytripower4",
    "inverter_002": "sunnytripower10se",
}

DEVICE_TYPE = {
    "meter_001": "meter",
    "meter_002": "meter",
    "meter_003": "meter",
    "meter_004": "meter",
    "inverter_001": "inverter",
    "inverter_002": "inverter",
}

# InfluxDB Fields
FIELDS_LOOKUP = {
    
}