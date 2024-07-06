import pandas as pd

file = pd.read_json(r'utils/sample.json')

print(file.loc['Database'])