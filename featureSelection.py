import pandas as pd

data = pd.read_csv("CS170_Small_Data__114.txt", sep="  ", header=None)
classification = data.iloc[:,0]
inst1feature2 = data.iloc[0].iat[2]

print(data)
print()

print("Classification:")
print(classification)
print()

print("2nd Feature of 1st Instance:", end='')
print(inst1feature2)

