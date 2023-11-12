from datetime import datetime
import time
import numpy as np
import pandas as pd
from pathlib import Path 
import matplotlib.pyplot as plt

while True:
    rows,cols = 1000,1   # (number of samples, number of columns)
    data = np.random.rand(rows,cols)
    data = data * 2500
    tidx = pd.date_range(datetime.now(), periods=rows, freq='15T')
    data_frame = pd.DataFrame(data, columns=['Power'], index=tidx)
    print(data_frame)

    filepath = Path('/opt/pi-grafana/csv_exports/out.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    data_frame.to_csv(filepath)  

    # plt.plot(data_frame)
    # plt.show()
    time.sleep(1)

