import numpy as np

data = open('opacity.txt','r')             #lest fil
data_split = [line.split() for line in data]    #split

logT = []
logkpa = []
logR = []

logR_values = True
for values in data_split:
    if len(values) > 0:
        if logR_values == False:    #Testing if we are on the top line (logR values)
            logT.append(values[0])
            del values[0]           #Deleting the logT value, and appending rest
            logkpa.append(values)
        elif logR_values == True:
            del values[0]
            logR.append(values)
            logR_values = False

#Making arrays consisting of floats
logT_arr = np.array(logT).astype(np.float)
logkpa_arr = np.array(logkpa).astype(np.float)
logR_arr = np.array(logR).astype(np.float)
