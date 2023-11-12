import numpy as np
import pandas as pd
from pathlib import Path 
np.random.seed(1)

rows,cols = 100,1
data = np.random.rand(rows,cols) # You can use other random functions to generate values with constraints
tidx = pd.date_range('2019-01-01', periods=rows, freq='T') # freq='MS'set the frequency of date in months and start from day 1. You can use 'T' for minutes and so on
data_frame = pd.DataFrame(data, columns=['Power'], index=tidx)
print(data_frame)

filepath = Path('csv_exports/out.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
data_frame.to_csv(filepath)  