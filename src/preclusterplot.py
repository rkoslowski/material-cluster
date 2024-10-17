import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import random as rnd
import warnings
import seaborn as sns

# Ignore certain warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)

########################### FUNCTIONS AND VARIABLES ###########################
def usecount(s,x):
    return (df2[s] == x).sum()

def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')	

classes = np.array([
    'Carbon',  'Ceramic', 'FerrousMetal',  
    'NonferrousMetal',  'OtherEngineeringMaterials', 
    'RapidPrototypingPolymer', 'Thermoplastic', 'Thermoset', 
    'PureElement', 'Wood'
    ])

########################### READ CSV FILES ###########################

df2 	= pd.read_csv('out/twoProps/uniqueValues.csv', sep=';')
df3 	= pd.read_csv('out/threeProps/uniqueValues.csv', sep=';')
dfnI 	= pd.read_csv('out/threeProps/noIdentity.csv', sep=';')

########################### EDIT COLUMNS ###########################

df2.rename(columns={'Density (g/cc)': 'Density'}, inplace=True)
df2.rename(columns={'E-Modul (GPa)': 'ModulusOfElasticity'}, inplace=True)

df3.rename(columns={'Density (g/cc)': 'Density'}, inplace=True)
df3.rename(columns={'E-Modul (GPa)': 'ModulusOfElasticity'}, inplace=True)
df3.rename(columns={'Poissons Ratio': 'PoissonsRatio'}, inplace=True)

dfnI.rename(columns={'Density (g/cc)': 'Density'}, inplace=True)
dfnI.rename(columns={'E-Modul (GPa)': 'ModulusOfElasticity'}, inplace=True)
dfnI.rename(columns={'Poissons Ratio': 'PoissonsRatio'}, inplace=True)


# PD Series to map each Class to one number for making a colorseries used in cmap
g = rnd.choices(range(1,10), k=len(df2.index))

colorSeries = pd.Series(data=g)

t = len(classes)
offset = 0
it = 0
r = 0

for i in range(0,t):
	
	r = usecount("Class", classes[it])
	for j in range(0,r):
		colorSeries[j]=i

	it += 1
	offset += r


'''
tmp=len(classes)

l = rnd.choices(range(1,10), k=tmp)

cdfData = {'Class': l, 'Count': l}

cdf = pd.DataFrame(data=cdfData)	

it = 0
n = 0

for i in classes:

	cc = usecount("Class", classes[it])

	cdf.iloc[n,0] = classes[it]
	cdf.iloc[n,1] = cc

	it += 1
	n += 1
'''

# Data normalization between 0 and 1

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


#### data from df2
name 		= pd.Series(df2['Name'])
density 	= pd.Series(df2['Density'])
ndensity 	= NormalizeData(density)
eModul 		= pd.Series(df2['ModulusOfElasticity'])
neModul 	= NormalizeData(eModul)

### data from df3
name3d 		= pd.Series(df3['Name'])
density3d  	= pd.Series(df3['Density'])
modulus3d	= pd.Series(df3['ModulusOfElasticity'])
poisson3d  	= pd.Series(df3['PoissonsRatio'])

### data from dfnI
name3dnI 	= pd.Series(dfnI['Name'])
density3dnI = pd.Series(dfnI['Density'])
modulus3dnI	= pd.Series(dfnI['ModulusOfElasticity'])
poisson3dnI = pd.Series(dfnI['PoissonsRatio'])

# create figures
fig2d 	= plt.figure(num= "2D Plot" ,figsize=[13,8])

fig3d 	= plt.figure(num= "3D Plot - Classified" ,figsize=[13,8])

fig3nI 	= plt.figure(num= "3D Plot - no Identity" ,figsize=[13,8])


############################# 2D Plot #############################


'''
fig2d.tight_layout(pad=3.5)
#fig2d.set_dpi(300)

# Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = GridSpec(4,4)

ax = fig2d.add_subplot(gs[1, 0])

ax_joint = fig2d.add_subplot(gs[1:4,0:3])
ax_marg_x = fig2d.add_subplot(gs[0,0:3])
ax_marg_y = fig2d.add_subplot(gs[1:4,3])


groups1 = df2.groupby('Class')
for name, group in groups1:
    #subplot1.scatter(group.Density, group.EModul, c=colorSeries, cmap='magma', s=40, label=name, edgecolors='none', alpha=0.5 )
	# use the previously defined function
	ax_joint.scatter(group.Density, group.ModulusOfElasticity, s=40, label=name, edgecolors='none', alpha=0.65 )
	


#logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

hist, bins = np.histogram(density, bins=125)

logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

ax_marg_x.hist(density, bins=logbins)

#logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

ax_marg_y.hist(eModul,orientation="horizontal", bins=logbins)

# Turn off tick labels on marginals
plt.setp(ax_marg_x.get_xticklabels(), visible=False)
#plt.setp(ax_marg_x.get_yticklabels(), visible=False)
#plt.setp(ax_marg_y.get_xticklabels(), visible=False)
plt.setp(ax_marg_y.get_yticklabels(), visible=False)


# Set labels on marginals
ax_marg_y.set_xlabel('Count')
ax_marg_x.set_ylabel('Count')

ax_joint.set_yscale('log')
ax_joint.set_xscale('log')

ax_marg_y.set_yscale('log')
#ax_marg_y.set_xscale('log')

#ax_marg_x.set_yscale('log')
ax_marg_x.set_xscale('log')

#scatter = subplot1.scatter(density, eModul,  alpha=0.6, edgecolors='none', cmap='magma', s=40)

ax_joint.grid(True,which="both", linestyle=':', linewidth=1)
ax_joint.legend(loc='lower right')

ax_joint.set_title("2D-Plot LOG")
ax_joint.set_xlabel('Density [g/cc]')
ax_joint.set_ylabel('E-Modul [GPa]')
'''

############################# 2D Plot #############################

sub2d = fig2d.add_subplot(1, 1, 1)
#sub2d2= fig2d.add_subplot(2, 1, 2)
#sub3 = fig2d.add_subplot(2, 1, 3)
#sub4 = fig2d.add_subplot(3, 1, 3)

sub2d.set_title("2D-Plot LOG")
sub2d.set_xlabel('Density [g/cc]')
sub2d.set_ylabel('E-Modul [GPa]')

fig2d.tight_layout(pad=3.5)
#fig2d.set_dpi(300)

groups1 = df2.groupby('Class')
for name, group in groups1:
    #subplot1.scatter(group.Density, group.EModul, c=colorSeries, cmap='magma', s=40, label=name, edgecolors='none', alpha=0.5 )
	# use the previously defined function
	sub2d.scatter(group.Density, group.ModulusOfElasticity, s=40, label=name, edgecolors='none', alpha=0.65 )

sub2d.set_yscale('log')
sub2d.set_xscale('log')

#scatter = subplot1.scatter(density, eModul,  alpha=0.6, edgecolors='none', cmap='magma', s=40)

sub2d.grid(True,which="both", linestyle=':', linewidth=1)
sub2d.legend(loc='lower right')

#plt.grid(b=True, which= 'major', axis='both')

############################# 3D Plot #############################

fig3dsub = fig3d.add_subplot(projection='3d')

ax = fig3d.gca(projection='3d')

ax.set_title("3D-Plot")
ax.set_xlabel('Density [g/cc]')
ax.set_ylabel('E-Modul [GPa]')
ax.set_zlabel('Poissons Ratio')

cmap1 = ListedColormap(sns.color_palette("husl", 256).as_hex())

groups3d = df3.groupby('Class')
for name, group in groups3d:
    #subplot1.scatter(group.Density, group.EModul, c=colorSeries, cmap='magma', s=40, label=name, edgecolors='none', alpha=0.5 )
	sc=ax.scatter(group.Density, group.ModulusOfElasticity, group.PoissonsRatio, s=40, label=name, edgecolors='none', alpha=0.65, cmap=cmap1 )



fig3dnIsub = fig3nI.add_subplot(projection= '3d')

axx = fig3nI.gca(projection='3d')

fig3dnIsub.set_title("3D-Plot")
fig3dnIsub.set_xlabel('Density [g/cc]')
fig3dnIsub.set_ylabel('E-Modul [GPa]')
fig3dnIsub.set_zlabel('Poissons Ratio')
'''
groups3dnI = dfnI.groupby('Density')
for name, group in groups3dnI:
    #subplot1.scatter(group.Density, group.EModul, c=colorSeries, cmap='magma', s=40, label=name, edgecolors='none', alpha=0.5 )
	sc=fig3dnIsub.scatter(group.Density, group.ModulusOfElasticity, group.PoissonsRatio, s=40, edgecolors='none', alpha=0.65, cmap=cmap1 )

'''

scatterPlotnI = fig3dnIsub.scatter(density3dnI ,modulus3dnI, poisson3dnI, alpha=0.6, edgecolors='none', cmap=cmap1, s=40)


#rc = fig3dsub.scatter(density3dnI, modulus3dnI, poisson3dnI, s=40, edgecolors='none', alpha=0.65 )

#plt.legend(*rc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)



#plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)




'''


################### ... ###################


#subplot1.scatter(density, eModul, c='tab:blue', alpha=1, edgecolors='none', s=40)
subplot1.scatter(density, eModul, c=colorSeries, alpha=0.5, edgecolors='none', s=40, cmap='gist_earth', label=df2.Class)



################### AGGLOMARATION ###################

colors = ['silver','indianred','wheat','mediumseagreen','lightsteelblue','mediumpurple','lightpink','sandybrown','teal','orange']

t = len(classes)
offset = 0
it = 0
r = 0

for i in range(0,t):
	
	r = usecount("Class", classes[it])
	tmpdf = df2.iloc[offset:offset+r,0:4]
	den=pd.Series(tmpdf['Density'])
	emod=pd.Series(tmpdf['EModul'])

	#subplot1.scatter(density, eModul, c='blue', alpha=0.1, edgecolors='none', s=40)
	#subplot2.scatter(den, emod, c=colors[i], alpha=0.5, edgecolors='none', cmap='viridis', s=40)
	subplot2.scatter(density, eModul, c='tab:blue', alpha=0.05, edgecolors='none', s=40) # agglomeration

	it += 1
	offset += r

'''


plt.show()