#import csv  
import pandas as pd
import numpy  as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

################################## VARIABLES ##################################

categories = np.array([
    'mc0', 'mc1', 'mc2', 'mc3', 'mc4', 'mc5', 'mc6', 'mc7', 'mc8', 'mc9'
    ])

classes = np.array([
    'Carbon',  'Ceramic', 'FerrousMetal',  
    'NonferrousMetal',  'OtherEngineeringMaterials', 
    'RapidPrototypingPolymer', 'Thermoplastic', 'Thermoset', 
    'PureElement', 'Wood'
    ])

pdSeriescat = pd.Series(data=categories, index=None, copy=False)

pdSeriesclass = pd.Series(data=classes, index=None, copy=False)

removedValues = pd.DataFrame(columns=['Name','Density (g/cc)','E-Modul (GPa)','Class'])

################################## FUNCTIONS ##################################

# function for receiving the occurances of a specific value in a specific column
def usecount(s,x):
    return (df[s] == x).sum()

################################## FILE READ ##################################

print("\n-------------- CSV PROCESSING TEST BEGIN --------------\n")

# Alternative but not as well integrated as pandas... i'll keep this just in case
'''
# open file for reading
with open('input/Materials_ModulusOfElasticity_Density.csv') as csvDataFile:
    read file as csv file 
    csvFile = list(csv.reader(csvDataFile, delimiter=';'))
'''

# load csv file with ';' delimiter
df = pd.read_csv('input/Materials_ModulusOfElasticity_Density.csv', ';')

#rc = df.shape[0]
rc = len(df.index)

print("Original rowcount:", rc, '\n')

################################## REMOVE DUPLICATES ##################################

# extract duplicates (This count does not include multiple duplicates within my filter )
dup=(df[df.duplicated(subset=['Name','Density (g/cc)'], keep='first')]) #
#print(dup) # show duplicates

removedValues = pd.concat([removedValues,dup])

# Remove duplicates from main df
df.drop_duplicates(subset=['Name','Density (g/cc)','E-Modul (GPa)' ], keep='first', inplace= True)
print("-- Removing identical duplicates...")

#Convert unique values back to csv
#df.to_csv('out/extractedIdentical.csv', ';', columns=['Name','Density (g/cc)','E-Modul (GPa)','Class'], index_label=False , index=False)

# save that value for later...
duprc= len(df.index)
print("--", rc - len(df.index), "found and removed!")

print("\n New rowcount:", len(df.index), "\n") 

################################## REMOVE SPECIFIC WORDS ##################################

# List of Words to filter
searchfor = ['human', 'tissue', 'Tendon', 'cloth', 'Overview', 
            'liquid', 'adhesive', 'wound', 
            'injection', 'cable', 'pipe', 'gel', '#']

print("-- Removing unsuitable materials and summaries ...")

removedValues = pd.concat([removedValues, df.Name.str.contains('|'.join(searchfor), na=False, case=False)])

df = df[~df.Name.str.contains('|'.join(searchfor), na=False, case=False)]

print("--", duprc - len(df.index), "not found and removed!")

print("\n New rowcount:", len(df.index), "\n") 

# Delete unwanted characters
df = df.replace('"','', regex=True)

#df.to_csv('out/extractIdenticalsAndWords.csv', ';', columns=['Name','Density (g/cc)','E-Modul (GPa)','Class'], index_label=False , index=False)

# Replace all delimiter which would cause the table to uncorrectly displayed
'''
for i in range(0, rc):
	
    if((df.Name[i].count(';'))):
    print(df.Name[i].count(';'))
print(df.Name[161])
        df.Name[i].replace(";","p")
        
df.Name[6277].replace('"', ' ')

df.to_csv('out/uniqueValues.csv', ';', columns=['Name','Density (g/cc)','E-Modul (GPa)','Class'], index_label=False , index=False)

print(usecount('Class','0-0'))
'''

################################## REMOVE CLASS-SPECIFIC DUPLICATES ##################################

#df to collect all true identical mats (with small varientaions)
uniqueDF = pd.DataFrame(columns=['Name','Density (g/cc)','E-Modul (GPa)','Class'])

offset = 0
it = 0
i = 0

print("-- Removing all materials with identical properties within the same class... ")

# for every materialclass
for i in pdSeriescat:

    i = usecount("Class", classes[it])

    tmpdf = df.iloc[offset:offset+i,0:4]

    tmpdf.drop_duplicates(subset=['Density (g/cc)','E-Modul (GPa)' ], keep='first', inplace= True)

    #uniqueDF.append(tmpdf, ignore_index=True)
    uniqueDF = pd.concat([uniqueDF,tmpdf])
    removedValues = pd.concat([removedValues,tmpdf.duplicated(subset=['Density (g/cc)'], keep='first')])

    it += 1
    offset += i

#removedValues.to_csv('out/twoProps/removed.csv', ';', columns=['Name','Density (g/cc)','E-Modul (GPa)','Class'], index_label=False , index=False)

print ("--",len(df.index)-len(uniqueDF.index), "found and removed")

print("\n Final rowcount:", len(uniqueDF.index), "\n")

#cdfData = {'Class': ['1.2 - 1.4', '2.4 - 3.54','2.5 - 7.1', '3.2'] , 'Count': ['2.21 - 3.21', '5.4 - 5.5', '4.5', '0.9 - 2.2']}

#cdf = pd.DataFrame(data=cdfData)

udfL = len(uniqueDF.index)

#print (cdf)

for i in range (udfL):

    tmpCell1 = uniqueDF.iloc[i,1]
    tmpCell2 = uniqueDF.iloc[i,2]

    if '-' in tmpCell1:

        splitted1 = tmpCell1.split(' - ')
        splitted1[0] = float(splitted1[0])
        splitted1[1] = float(splitted1[1])
        newVal = round(((splitted1[0] + splitted1[1]) / 2) , 2 )
        uniqueDF.iloc[i,1] = newVal

    else:
        pass

    if '-' in tmpCell2:
	
        splitted2 = tmpCell2.split(' - ')
        splitted2[0] = float(splitted2[0])
        splitted2[1] = float(splitted2[1])
        newVal = round(((splitted2[0] + splitted2[1]) / 2) , 2 )
        uniqueDF.iloc[i,2] = newVal

    else:
        pass

uniqueDF.to_csv('out/twoProps/uniqueValues.csv', ';', columns=['Name','Density (g/cc)','E-Modul (GPa)','Class'], index_label=False , index=False)

print("--------------- CSV PROCESSING TEST END ---------------\n")