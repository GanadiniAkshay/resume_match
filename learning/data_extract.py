## This is the script to pull the data from TSV and store it in text files

import pandas as pd
xl = pd.ExcelFile('../raw_data/data.xls')
xl.sheet_names

df = xl.parse('Sheet1')
items = df.items()

for item in items:
    head = item[0]
    if head == 'Responsibilities/Job description':
        print(len(item[1]))
    
