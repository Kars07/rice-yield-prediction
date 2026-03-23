import pandas as pd
df = pd.read_stata('Anonymized data/1_Identification_ano.dta')
print(list(df.columns))
print(df.head())
