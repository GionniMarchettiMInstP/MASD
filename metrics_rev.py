import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cityblock
import matplotlib.pyplot as plt
#from matplotlib import rc

# Set the default font for the entire plot
plt.rcParams['font.family'] = 'Arial'  # Replace 'Arial' with the desired font
#plt.rcParams['font.size'] = 12  # Replace 12 with the desired font size

#path = '/Users/myfavouritefolder/dataset_fibrinogen/'

path = '/Users/gionni/Desktop/revision_alberto/dataset_fibrinogen/'


# Check current working directory.
retval = os.getcwd()

print("Current working directory %s" % retval)

# Now change the directory
os.chdir( path )

# Check current working directory.
retval = os.getcwd()

print("Directory changed successfully %s" % retval)




#inputs


# number of UVRR spectra Fib, Fib + CNP, Fib + SIO2NP

nos = 17  

# number of UV spectra  Fib + CNP

nou = 34

# number of CD spectra Fib + CNP

noc = 34


# number of UV spectra  Fib + SIO2NP

noua = 33

# number of CD spectra Fib + SIO2NP

noca = 33



# index for reference spectrum

irs = 0 


#interval limits for spectral truncations of UVRR (unit: cm^{-1})

l1 = 1310.
l2 = 1391.

# upper limit for Fib + nanoparticles
l2n = 1390.

l3 = 1530.
l4 = 1580.

l5 = 1660.
l6 = 1670.

l7 = 1730.
l8 = 1800.

l9 = 2800.
l10 = 3800.

#interval limits for spectral truncations of UV (unit: nm)


m1 = 210.
m2 = 260.


#interval limits for spectral truncations of CD (unit: nm)


n1 = 210.
n2 = 260.


# list of temperature in Celsius

tc =[22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74, 78, 82, 86]

tc = np.array(tc)



#function

def Read_Two_Column_File(file_name):
    with open(file_name, 'r') as data:
        x = []
        y = []
        for line in data:
            p = line.split()
            x.append(float(p[0]))
            y.append(float(p[1]))

    return x, y


# Construction of composite spectra bulk Fibrinogen



intensity = []


for i in range(0,nos):
    strin =  str(i)
    string = 'inputUVRR' + '_' + strin + '.txt'
    x0, y0 = Read_Two_Column_File(string)
    for itemX in x0:
        if itemX >= l1 and itemX <= l2:
            index = x0.index(itemX)
            intensity.append(y0[index])


length = len(intensity)
lenr = float(length)

split = lenr/float(nos)   
splitin = int(split)

ina = [intensity[x:x+splitin] for x in range(0, length, splitin)]


intensity = []


for i in range(0,nos):
    strin =  str(i)
    string = 'inputUVRR' + '_' + strin + '.txt'
    x0, y0 = Read_Two_Column_File(string)
    for itemX in x0:
        if itemX >= l3 and itemX <= l4:
            index = x0.index(itemX)
            intensity.append(y0[index])


length = len(intensity)
lenr = float(length)

split = lenr/float(nos)   
splitin = int(split)

inb = [intensity[x:x+splitin] for x in range(0, length, splitin)]





intensity = []


for i in range(0,nos):
    strin =  str(i)
    string = 'inputUVRR' + '_' + strin + '.txt'
    x0, y0 = Read_Two_Column_File(string)
    for itemX in x0:
        if itemX >= l5 and itemX <= l6:
            index = x0.index(itemX)
            intensity.append(y0[index])


length = len(intensity)
lenr = float(length)

split = lenr/float(nos)   
splitin = int(split)

inc = [intensity[x:x+splitin] for x in range(0, length, splitin)]




intensity = []


for i in range(0,nos):
    strin =  str(i)
    string = 'inputUVRR' + '_' + strin + '.txt'
    x0, y0 = Read_Two_Column_File(string)
    for itemX in x0:
        if itemX >= l7 and itemX <= l8:
            index = x0.index(itemX)
            intensity.append(y0[index])


length = len(intensity)
lenr = float(length)

split = lenr/float(nos)   
splitin = int(split)

ind = [intensity[x:x+splitin] for x in range(0, length, splitin)]


intensity = []


for i in range(0,nos):
    strin =  str(i)
    string = 'inputUVRR' + '_' + strin + '.txt'
    x0, y0 = Read_Two_Column_File(string)
    for itemX in x0:
        if itemX >= l9 and itemX <= l10:
            index = x0.index(itemX)
            intensity.append(y0[index])


length = len(intensity)
lenr = float(length)

split = lenr/float(nos)   
splitin = int(split)

ine = [intensity[x:x+splitin] for x in range(0, length, splitin)]




intensity = []


for i in range(0,nou):
    strin =  str(i)
    string = 'inputUV' + '_' + strin + '.txt'
    x0, y0 = Read_Two_Column_File(string)
    for itemX in x0:
        if itemX >= m1 and itemX <= m2:
            index = x0.index(itemX)
            intensity.append(y0[index])
           

length = len(intensity)
lenr = float(length)

split = lenr/float(nou)   
splitin = int(split)

inf = [intensity[x:x+splitin] for x in range(0, length, splitin)]


intensity = []


for i in range(0,noc):
    strin =  str(i)
    string = 'inputCD' + '_' + strin + '.txt'
    x0, y0 = Read_Two_Column_File(string)
    for itemX in x0:
        if itemX >= n1 and itemX <= n2:
            index = x0.index(itemX)
            intensity.append(y0[index])
            

length = len(intensity)
lenr = float(length)

split = lenr/float(noc)   
splitin = int(split)

ing = [intensity[x:x+splitin] for x in range(0, length, splitin)]




# Construction of data matrix whose rows are composite bulk spectra 

joinList = []

for i in range(0,nos):
    j = 2 * i
    intot = ina[i] + inb[i] + inc[i] + ind[i] + ine[i] + inf[j] + ing[j]
    joinList.append(intot)


X = np.array(joinList)





# Construction of composite spectra in presence of CNP


intensity = []


for i in range(0,nos):
    strin =  str(i)
    string = 'inputUVRR_CNP' + '_' + strin + '.txt'
    x0, y0 = Read_Two_Column_File(string)
    for itemX in x0:
        if itemX >= l1 and itemX <= l2n:
            index = x0.index(itemX)
            intensity.append(y0[index])


length = len(intensity)
lenr = float(length)

split = lenr/float(nos)   
splitin = int(split)

ina = [intensity[x:x+splitin] for x in range(0, length, splitin)]


intensity = []


for i in range(0,nos):
    strin =  str(i)
    string = 'inputUVRR_CNP' + '_' + strin + '.txt'
    x0, y0 = Read_Two_Column_File(string)
    for itemX in x0:
        if itemX >= l3 and itemX <= l4:
            index = x0.index(itemX)
            intensity.append(y0[index])


length = len(intensity)
lenr = float(length)

split = lenr/float(nos)   
splitin = int(split)

inb = [intensity[x:x+splitin] for x in range(0, length, splitin)]





intensity = []


for i in range(0,nos):
    strin =  str(i)
    string = 'inputUVRR_CNP' + '_' + strin + '.txt'
    x0, y0 = Read_Two_Column_File(string)
    for itemX in x0:
        if itemX >= l5 and itemX <= l6:
            index = x0.index(itemX)
            intensity.append(y0[index])


length = len(intensity)
lenr = float(length)

split = lenr/float(nos)   
splitin = int(split)

inc = [intensity[x:x+splitin] for x in range(0, length, splitin)]




intensity = []


for i in range(0,nos):
    strin =  str(i)
    string = 'inputUVRR_CNP' + '_' + strin + '.txt'
    x0, y0 = Read_Two_Column_File(string)
    for itemX in x0:
        if itemX >= l7 and itemX <= l8:
            index = x0.index(itemX)
            intensity.append(y0[index])


length = len(intensity)
lenr = float(length)

split = lenr/float(nos)   
splitin = int(split)

ind = [intensity[x:x+splitin] for x in range(0, length, splitin)]


intensity = []


for i in range(0,nos):
    strin =  str(i)
    string = 'inputUVRR_CNP' + '_' + strin + '.txt'
    x0, y0 = Read_Two_Column_File(string)
    for itemX in x0:
        if itemX >= l9 and itemX <= l10:
            index = x0.index(itemX)
            intensity.append(y0[index])


length = len(intensity)
lenr = float(length)

split = lenr/float(nos)   
splitin = int(split)

ine = [intensity[x:x+splitin] for x in range(0, length, splitin)]




intensity = []


for i in range(0,nou):
    strin =  str(i)
    string = 'inputUV_CNP' + '_' + strin + '.txt'
    x0, y0 = Read_Two_Column_File(string)
    for itemX in x0:
        if itemX >= m1 and itemX <= m2:
            index = x0.index(itemX)
            intensity.append(y0[index])
           

length = len(intensity)
lenr = float(length)

split = lenr/float(nou)   
splitin = int(split)

inf = [intensity[x:x+splitin] for x in range(0, length, splitin)]


intensity = []


for i in range(0,noc):
    strin =  str(i)
    string = 'inputCD_CNP' + '_' + strin + '.txt'
    x0, y0 = Read_Two_Column_File(string)
    for itemX in x0:
        if itemX >= n1 and itemX <= n2:
            index = x0.index(itemX)
            intensity.append(y0[index])
            #print(x0[index],y0[index])


length = len(intensity)
lenr = float(length)

split = lenr/float(noc)   
splitin = int(split)

ing = [intensity[x:x+splitin] for x in range(0, length, splitin)]




# Construction of data matrix whose rows are multi-component spectra 
# in presence of CNP

joinList = []

for i in range(0,nos):
    j = 2 * i
    intot = ina[i] + inb[i] + inc[i] + ind[i] + ine[i] + inf[j] + ing[j]
    joinList.append(intot)


X_c = np.array(joinList)




# Construction of composite spectra in presence of SIO2NP


intensity = []


for i in range(0,nos):
    strin =  str(i)
    string = 'inputUVRR_SIO2NP' + '_' + strin + '.txt'
    x0, y0 = Read_Two_Column_File(string)
    for itemX in x0:
        if itemX >= l1 and itemX <= l2n:
            index = x0.index(itemX)
            intensity.append(y0[index])


length = len(intensity)
lenr = float(length)

split = lenr/float(nos)   
splitin = int(split)

ina = [intensity[x:x+splitin] for x in range(0, length, splitin)]


intensity = []


for i in range(0,nos):
    strin =  str(i)
    string = 'inputUVRR_SIO2NP' + '_' + strin + '.txt'
    x0, y0 = Read_Two_Column_File(string)
    for itemX in x0:
        if itemX >= l3 and itemX <= l4:
            index = x0.index(itemX)
            intensity.append(y0[index])


length = len(intensity)
lenr = float(length)

split = lenr/float(nos)   
splitin = int(split)

inb = [intensity[x:x+splitin] for x in range(0, length, splitin)]





intensity = []


for i in range(0,nos):
    strin =  str(i)
    string = 'inputUVRR_SIO2NP' + '_' + strin + '.txt'
    x0, y0 = Read_Two_Column_File(string)
    for itemX in x0:
        if itemX >= l5 and itemX <= l6:
            index = x0.index(itemX)
            intensity.append(y0[index])


length = len(intensity)
lenr = float(length)

split = lenr/float(nos)   
splitin = int(split)

inc = [intensity[x:x+splitin] for x in range(0, length, splitin)]




intensity = []


for i in range(0,nos):
    strin =  str(i)
    string = 'inputUVRR_SIO2NP' + '_' + strin + '.txt'
    x0, y0 = Read_Two_Column_File(string)
    for itemX in x0:
        if itemX >= l7 and itemX <= l8:
            index = x0.index(itemX)
            intensity.append(y0[index])


length = len(intensity)
lenr = float(length)

split = lenr/float(nos)   
splitin = int(split)

ind = [intensity[x:x+splitin] for x in range(0, length, splitin)]


intensity = []


for i in range(0,nos):
    strin =  str(i)
    string = 'inputUVRR_SIO2NP' + '_' + strin + '.txt'
    x0, y0 = Read_Two_Column_File(string)
    for itemX in x0:
        if itemX >= l9 and itemX <= l10:
            index = x0.index(itemX)
            intensity.append(y0[index])


length = len(intensity)
lenr = float(length)

split = lenr/float(nos)   
splitin = int(split)

ine = [intensity[x:x+splitin] for x in range(0, length, splitin)]




intensity = []


for i in range(0,noua):
    strin =  str(i)
    string = 'inputUV_SIO2NP' + '_' + strin + '.txt'
    x0, y0 = Read_Two_Column_File(string)
    for itemX in x0:
        if itemX >= m1 and itemX <= m2:
            index = x0.index(itemX)
            intensity.append(y0[index])
           

length = len(intensity)
lenr = float(length)

split = lenr/float(noua)   
splitin = int(split)

inf = [intensity[x:x+splitin] for x in range(0, length, splitin)]


intensity = []


for i in range(0,noca):
    strin =  str(i)
    string = 'inputCD_SIO2NP' + '_' + strin + '.txt'
    x0, y0 = Read_Two_Column_File(string)
    for itemX in x0:
        if itemX >= n1 and itemX <= n2:
            index = x0.index(itemX)
            intensity.append(y0[index])
           

length = len(intensity)
lenr = float(length)

split = lenr/float(noca)   
splitin = int(split)

ing = [intensity[x:x+splitin] for x in range(0, length, splitin)]




# Construction of data matrix whose rows are multi-component spectra 
# in presence of SIO2NP

joinList = []

for i in range(0,nos):
    j = 2 * i
    intot = ina[i] + inb[i] + inc[i] + ind[i] + ine[i] + inf[j] + ing[j]
    joinList.append(intot)


X_s = np.array(joinList)

print(X_s.shape)



# Calculate the variance along the rows
row_variances = np.var(X, axis=1)

# Find the maximum variance
max_variance = np.max(row_variances)

print("Row Variances:", row_variances)
print("Maximum Variance Fib:", max_variance)



# Calculate the variance along the rows
row_variances = np.var(X_c, axis=1)

# Find the maximum variance
max_variance = np.max(row_variances)

print("Row Variances:", row_variances)
print("Maximum Variance CNP:", max_variance)


# Calculate the variance along the rows
row_variances = np.var(X_s, axis=1)

# Find the maximum variance
max_variance = np.max(row_variances)

print("Row Variances:", row_variances)
print("Maximum Variance SiO2:", max_variance)





sys.exit()



# Standardise the data


scaler = StandardScaler()

X_st = scaler.fit_transform(X)
X_c_st = scaler.fit_transform(X_c)
X_s_st = scaler.fit_transform(X_s)


#print(X_st.shape)

#print(X.mean(axis=0))
#print('beheld')
#print(X_st.mean(axis=0))

#print(np.var(X, axis=0))

#variance = np.var(X_st, axis=0)
#print(variance.shape)




euclidean = []
wasserstein = []
manhattan = []




ref = X_st[irs]

for j in range(0, nos):  
    test = X_st[j]
    det =  distance.euclidean(test, ref)
    #dwt = wasserstein_distance(test, ref)
    dmt = cityblock(test, ref)
    euclidean.append(det)
    #wasserstein.append(dwt)
    manhattan.append(dmt)
    
   
de = np.array(euclidean)
#dw = np.array(wasserstein)
dm = np.array(manhattan)

max_de = np.amax(de)
#max_dw = np.amax(dw)
max_dm = np.amax(dm)


# normalisation of metrics

den = de/max_de
#dwn = dw/max_dw
dmn = dm/max_dm



euclidean = []
#wasserstein = []
manhattan = []

ref = X_c_st[irs]

for j in range(0, nos):
    test = X_c_st[j]
    det =  distance.euclidean(test, ref)
    #dwt = wasserstein_distance(test, ref)
    dmt = cityblock(test, ref)
    euclidean.append(det)
    #wasserstein.append(dwt)
    manhattan.append(dmt)

   
de = np.array(euclidean)
#dw = np.array(wasserstein)
dm = np.array(manhattan)

max_de = np.amax(de)
#max_dw = np.amax(dw)
max_dm = np.amax(dm)


# normalisation of metrics

dem = de/max_de
#dwm = dw/max_dw
dmm = dm/max_dm





euclidean = []
#wasserstein = []
manhattan = []

ref = X_s_st[irs]

for j in range(0, nos):  
    test = X_s_st[j]
    det =  distance.euclidean(test, ref)
    #dwt = wasserstein_distance(test, ref)
    dmt = cityblock(test, ref)
    euclidean.append(det)
    #wasserstein.append(dwt)
    manhattan.append(dmt)
    
   
de = np.array(euclidean)
#dw = np.array(wasserstein)
dm = np.array(manhattan)

max_de = np.amax(de)
#max_dw = np.amax(dw)
max_dm = np.amax(dm)


# normalisation of metrics

deq = de/max_de
#dwq = dw/max_dw
dmq = dm/max_dm



# Create a figure with subplots
fig, axs = plt.subplots(3, 1, sharex=False, figsize=(8, 10))

y_ticks = [0.0, 0.25, 0.50, 0.75, 1.0]

# Plot each curve in a different subplot
axs[0].plot(tc, den, '-k',  marker ='s', markersize=11, linewidth=1.8)
axs[1].plot(tc, dem, '-k',  marker ='o', markersize=11, linewidth=1.8)
axs[2].plot(tc, deq, '-k',  marker ='^', markersize=11, linewidth=1.8)


# Customize the plot
for i, ax in enumerate(axs):
    ax.grid(True)
    ax.set_ylim(0, 1)  # Set y-axis limit to [0, 1]
    ax.set_xlim(22, 86)  # Set x-axis limit to [22, 86]
    ax.set_ylabel(f'Euclidean', fontsize=15) #{i+1}
    ax.set_xlabel(r'T($ ^\circ$C )',fontsize=15)
    # Set x-axis ticks and labels for each subplot
    x_ticks = np.arange(22, 87, 4)  # Ticks from 22 to 86, each at distance 4
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(x_ticks, fontsize=14)
    ax.set_yticklabels(y_ticks, fontsize=14)





# Manually set legends for each panel
axs[0].legend(['Fibrinogen'],  loc='lower right', fontsize=16)
axs[1].legend(['Fib + CNP'], loc='lower right',  fontsize=16)
axs[2].legend(['Fib + SiNP'], loc='lower right', fontsize=16)


# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.5)  # Adjust the value as needed for your desired spacing



# Show the plot
plt.show()



# 1) Compute the minimum of each row and print the values
row_minima = np.min(X, axis=1)
print("Row minima:", row_minima)

# 2) Compute the minima of the row minima (Lambda)
Lambda = np.min(row_minima)
print("Lambda:", Lambda)



# Add Lambda to each entry of X
X_modified = X - 2 * Lambda
#print("X after adding Lambda to each entry:\n", X_modified)

print(np.all(X_modified > 0))

# 3) Normalize each row by dividing each entry by the total sum of its respective row
row_sums = np.sum(X_modified, axis=1, keepdims=True)
X_normalized = X_modified / row_sums
#print("X after row-wise normalization:\n", X_normalized)



# 1) Compute the minimum of each row and print the values
row_minima = np.min(X_c, axis=1)
print("Row minima:", row_minima)

# 2) Compute the minima of the row minima (Omega)
Omega = np.min(row_minima)
print("Omega:", Omega)



# Add Omega to each entry of X
X_c_modified = X_c - 2 * Omega
#print("X after adding Omega to each entry:\n", X_c_modified)

print(np.all(X_c_modified > 0))

# 3) Normalize each row by dividing each entry by the total sum of its respective row
row_sums = np.sum(X_c_modified, axis=1, keepdims=True)
X_c_normalized = X_c_modified / row_sums
#print("X after row-wise normalization:\n", X_c_normalized)


# 1) Compute the minimum of each row and print the values
row_minima = np.min(X_s, axis=1)
print("Row minima:", row_minima)

# 2) Compute the minima of the row minima (Rho)
Rho = np.min(row_minima)
print("Rho:", Rho)



# Add Rho to each entry of X
X_s_modified = X_s - 2 * Rho
#print("X after adding Rho to each entry:\n", X_s_modified)

print(np.all(X_s_modified > 0))

# 3) Normalize each row by dividing each entry by the total sum of its respective row
row_sums = np.sum(X_s_modified, axis=1, keepdims=True)
X_s_normalized = X_s_modified / row_sums
#print("X after row-wise normalization:\n", X_s_normalized)




X_st = X_normalized
X_c_st = X_c_normalized
X_s_st = X_s_normalized




print(np.all(X_st > 0))




print(X_st.shape)



support = np.arange(856)


wasserstein = []




ref = X_st[irs]

for j in range(0, nos):  
    test = X_st[j]
    dwt = wasserstein_distance(support, support, test, ref)
    wasserstein.append(dwt)
    
   

dw = np.array(wasserstein)


max_dw = np.amax(dw)

# normalisation of metrics

dwn = dw/max_dw



wasserstein = []


ref = X_c_st[irs]

for j in range(0, nos):
    test = X_c_st[j]
    dwt = wasserstein_distance(support, support, test, ref)
    wasserstein.append(dwt)
    
   

dw = np.array(wasserstein)



max_dw = np.amax(dw)


# normalisation of metrics


dwm = dw/max_dw


wasserstein = []


ref = X_s_st[irs]

for j in range(0, nos):
    test = X_s_st[j]
    dwt = wasserstein_distance(support, support, test, ref)
    wasserstein.append(dwt)
    
   

dw = np.array(wasserstein)



max_dw = np.amax(dw)


# normalisation of metrics


dwq = dw/max_dw


print(dwn)
print(dwq)

#sys.exit()




# Create a figure with subplots
fig, axs = plt.subplots(3, 1, sharex=False, figsize=(8, 10))

# Plot each curve in a different subplot
axs[0].plot(tc, dwn, '-k',  marker ='s', markersize=11, linewidth=1.8)
axs[1].plot(tc, dwm, '-k',  marker ='o', markersize=11, linewidth=1.8)
axs[2].plot(tc, dwq, '-k',  marker ='^', markersize=11, linewidth=1.8)


# Customize the plot
for i, ax in enumerate(axs):
    #ax.legend()
    ax.grid(True)
    ax.set_ylim(0, 1)  # Set y-axis limit to [0, 1]
    ax.set_xlim(22, 86)  # Set x-axis limit to [22, 86]
    ax.set_ylabel(f'Wasserstein', fontsize=15) #{i+1}
    ax.set_xlabel(r'T($ ^\circ$C)',fontsize=15)
    # Set x-axis ticks and labels for each subplot
    x_ticks = np.arange(22, 87, 4)  # Ticks from 22 to 86, each at distance 4
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(x_ticks, fontsize=14)
    ax.set_yticklabels(y_ticks, fontsize=14)


   





# Manually set legends for each panel
axs[0].legend(['Fibrinogen'],  loc='lower right',  fontsize=16)
axs[1].legend(['Fib + CNP'], loc='lower right',  fontsize=16)
axs[2].legend(['Fib + SiNP'], loc='lower right',  fontsize=16)


# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.5)  # Adjust the value as needed for your desired spacing




plt.savefig('wasserstein_2025.png', bbox_inches='tight')

# Show the plot
plt.show()




# Create a figure with subplots
fig, axs = plt.subplots(3, 1, sharex=False, figsize=(8, 10))

# Plot each curve in a different subplot
axs[0].plot(tc, dmn, '-k',  marker ='s', markersize=11, linewidth=1.8)
axs[1].plot(tc, dmm, '-k',  marker ='o', markersize=11, linewidth=1.8)
axs[2].plot(tc, dmq, '-k',  marker ='^', markersize=11, linewidth=1.8)


# Customize the plot
for i, ax in enumerate(axs):
    ax.grid(True)
    ax.set_ylim(0, 1)  # Set y-axis limit to [0, 1]
    ax.set_xlim(22, 86)  # Set x-axis limit to [22, 86]
    ax.set_ylabel(f'Manhattan', fontsize=15) #{i+1}
    ax.set_xlabel(r'T($ ^\circ$C)', fontsize=15)
    # Set x-axis ticks and labels for each subplot
    x_ticks = np.arange(22, 87, 4)  # Ticks from 22 to 86, each at distance 4
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(x_ticks, fontsize=14)
    ax.set_yticklabels(y_ticks, fontsize=14)




# Manually set legends for each panel
axs[0].legend(['Fibrinogen'],  loc='lower right',  fontsize=16)
axs[1].legend(['Fib + CNP'], loc='lower right',  fontsize=16)
axs[2].legend(['Fib + SiNP'], loc='lower right',  fontsize=16)


# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.5)  # Adjust the value as needed for your desired spacing


plt.savefig('manhattan_2025.png', bbox_inches='tight')

# Show the plot
plt.show()




print(deq)
print(dmq)















