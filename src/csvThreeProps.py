import pandas as pd
import numpy  as np
import warnings
import sys
import time

warnings.simplefilter(action='ignore', category=FutureWarning)

################################## VARIABLES ##################################

classes = np.array([
    'Carbon',  'Ceramic', 'FerrousMetal',  
    'NonferrousMetal',  'OtherEngineeringMaterials', 
    'RapidPrototypingPolymer', 'Thermoplastic', 'Thermoset', 
    'PureElement', 'Wood'
    ])

pdSeriesclass = pd.Series(data=classes, index=None, copy=False)

removedValues = pd.DataFrame(columns=['Name','Density (g/cc)','E-Modul (GPa)','Class'])

################################## FUNCTIONS ##################################

# function for receiving the occurances of a specific value in a specific column
def usecount(s,x):
    return (df[s] == x).sum()

def loadingStatus(x,lines):
    a = (x/lines)*100
    return(round(a,1))

################################## FILE READ ##################################

print("\n-------------- CSV PROCESSING TEST BEGIN --------------\n")

# load csv file with ';' delimiter
df = pd.read_csv('input/Materials_PoissonRatio_ModulusOfElasticity_Density.csv', ';')

# class not yet determined -> fill up with 0
df['Class'] = 0

compdf = pd.read_csv('out/twoProps/uniqueValues.csv', ';')

#rc = df.shape[0]
rc = len(df.index)

print("Original rowcount:", rc, '\n')

################################## REMOVE DUPLICATES ##################################

# extract duplicates (This count does not include multiple duplicates within my filter )
duplicates=(df[df.duplicated(subset=['Name','Density (g/cc)','E-Modul (GPa)','Poissons Ratio'], keep='first')]) #
#print(dup) # show duplicates

removedValues = pd.concat([removedValues,duplicates])

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
            'injection', 'cable', 'pipe', 'gel']

print("-- Removing unsuitable materials and summaries ...")

removedValues = pd.concat([removedValues, df.Name.str.contains('|'.join(searchfor), na=False, case=False)])

df = df[~df.Name.str.contains('|'.join(searchfor), na=False, case=False)]

print("--", duprc - len(df.index), "found and removed!")

print("\n New rowcount:", len(df.index), "\n") 

# Replace all delimiter which would cause the table to uncorrectly displayed
df = df.replace('"','', regex=True)

'''
tdf =df

tdfL = len(tdf.index)

for i in range (tdfL):
	
    tmpCell1 = tdf.iloc[i,1]
    tmpCell2 = tdf.iloc[i,2]
    tmpCell3 = tdf.iloc[i,3]

    if '-' in tmpCell1:

        splitted1 = tmpCell1.split(' - ')
        splitted1[0] = float(splitted1[0])
        splitted1[1] = float(splitted1[1])
        newVal = round(((splitted1[0] + splitted1[1]) / 2) , 2 )
        tdf.iloc[i,1] = newVal

    else:
        pass

    if '-' in tmpCell2:
	
        splitted2 = tmpCell2.split(' - ')
        splitted2[0] = float(splitted2[0])
        splitted2[1] = float(splitted2[1])
        newVal = round(((splitted2[0] + splitted2[1]) / 2) , 2 )
        tdf.iloc[i,2] = newVal

    else:
        pass

    if '-' in tmpCell3:
		
        splitted3 = tmpCell3.split(' - ')
        splitted3[0] = float(splitted3[0])
        splitted3[1] = float(splitted3[1])
        newVal = round(((splitted3[0] + splitted3[1]) / 2) , 2 )
        tdf.iloc[i,3] = newVal

    else:
        pass

tdf.to_csv('out/threeProps/noIdentity.csv', ';', columns=['Name','Density (g/cc)','E-Modul (GPa)','Poissons Ratio','Class'], index_label=False , index=False)


'''


################################## COMPARE EACH FILE TO DETERMINE CLASS ##################################

print("-- Mapping rows and classes...\n")

lines = len(df.index)

for x in range (len(df.index)):
    status= "STATUS: "
    num = loadingStatus(x,lines)
    restout = "%" + " done!"
    print(status, num, restout, end="\r")
    compRow = df.iloc[x,0]
    for y in range (len(compdf.index)):
        #print("Comparing row", x , "from main with row", y, "from comp")
        if compRow == compdf.iloc[y,0]:
            copyVal = compdf.iloc[y,3]
            #df.loc[str(x),'Class'] = copyVal
            df.iloc[x,4] = copyVal
            break
        else:
            #df.iloc[x,4] = 0
            pass

######################################## SORT BY CLASS #########################################

df = df[(df['Class'] != 0 )]

df = df.sort_values(by=['Class'], axis=0, ascending= True, ignore_index=True)

print ("--",lines - len(df.index), "found and removed")

print("\n Final rowcount:", len(df.index), "\n")

dfL = len(df.index)

# remove value ranges and replace
for i in range (dfL):
	
    tmpCell1 = df.iloc[i,1]
    tmpCell2 = df.iloc[i,2]
    tmpCell3 = df.iloc[i,3]

    if '-' in tmpCell1:

        splitted1 = tmpCell1.split(' - ')
        splitted1[0] = float(splitted1[0])
        splitted1[1] = float(splitted1[1])
        newVal = round(((splitted1[0] + splitted1[1]) / 2) , 2 )
        df.iloc[i,1] = newVal

    else:
        pass

    if '-' in tmpCell2:
	
        splitted2 = tmpCell2.split(' - ')
        splitted2[0] = float(splitted2[0])
        splitted2[1] = float(splitted2[1])
        newVal = round(((splitted2[0] + splitted2[1]) / 2) , 2 )
        df.iloc[i,2] = newVal

    else:
        pass

    if '-' in tmpCell3:
	
        splitted3 = tmpCell3.split(' - ')
        splitted3[0] = float(splitted3[0])
        splitted3[1] = float(splitted3[1])
        newVal = round(((splitted3[0] + splitted3[1]) / 2) , 2 )
        df.iloc[i,3] = newVal

    else:
        pass

df.to_csv('out/threeProps/uniqueValues.csv', ';', columns=['Name','Density (g/cc)','E-Modul (GPa)','Poissons Ratio','Class'], index_label=False , index=False)



print("--------------- CSV PROCESSING TEST END ---------------\n")

