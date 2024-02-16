import sys
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.colors import ListedColormap


# Set the default font for the entire plot
plt.rcParams['font.family'] = 'Arial'  # Replace 'Arial' with the desired font
#plt.rcParams['font.size'] = 12  # Replace 12 with the desired font size




path = '/Users/gionni/Desktop/dataset_fibrinogen/'


# Check current working directory.
retval = os.getcwd()

print("Current working directory %s" % retval)

# Now change the directory
os.chdir( path )

# Check current working directory.
retval = os.getcwd()

print("Directory changed successfully %s" % retval)




rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=False)
plt.rcParams.update({'font.size': 20})

plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)



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



# number of PCA components

n_pca = 2

# number of t-SNE components

n_sne = 2


# Initialization of KMeans (seed is used for reproducibility only)

seed = 356


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
            #print(x0[index],y0[index])


length = len(intensity)
lenr = float(length)

split = lenr/float(noc)   
splitin = int(split)

ing = [intensity[x:x+splitin] for x in range(0, length, splitin)]




# Construction of data matrix whose rows are multi-component bulk spectra 

joinList = []

for i in range(0,nos):
    j = 2 * i
    intot = ina[i] + inb[i] + inc[i] + ind[i] + ine[i] + inf[j] + ing[j]
    joinList.append(intot)


X = np.array(joinList)





# Construction of multi-componente spectra in presence of CNP


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




# Construction of multi-componente spectra in presence of SIO2NP


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
            #print(x0[index],y0[index])


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


# form the data matrix 


X_t = np.concatenate((X, X_c), axis=0)
X_to = np.concatenate((X_t, X_s), axis=0)





# Standardise the data


scaler = StandardScaler()
X_train = scaler.fit_transform(X_to)


# Perform PCA

pca = PCA(n_components = n_pca) 
pca.fit(X_train) 


variance_preserved = pca.explained_variance_ratio_

pca1 = variance_preserved[0]
pca2 = variance_preserved[1]




print("Variance Preserved (%) by first component PCA1: {}".format(pca1))
print("Variance Preserved (%) by second component PCA2: {}".format(pca2))



# PCA through Singular  Value Decomposition (SVD)

U1, s1, Vt1 = np.linalg.svd(X_train)



W2 = Vt1.T[:, :2]

X_2d = X_train.dot(W2)

x, y = X_2d.T




# Splitting the arrays into three groups
nos = 17
x_groups = [x[:nos], x[nos:2 * nos], x[2 * nos:]]
y_groups = [y[:nos], y[nos:2 * nos], y[2 * nos:]]
t_groups = [tc, tc, tc]



# Create a scatter plot for each group with the custom colormap
markers = ['s', 'o', '^']
labels = ['Fib', 'Fib + CNP', 'Fib + SiNP']

# Increase marker size
marker_size = 55


for i in range(3):
    plt.scatter(x_groups[i], y_groups[i], c=t_groups[i], marker=markers[i], s=marker_size, cmap='jet', edgecolors='black', linewidths=1,label=labels[i])

# Adding labels and title
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA')

# Adding a color bar using the scatter plot of the first group
scatter = plt.scatter(x_groups[0], y_groups[0], c=t_groups[0], cmap='jet')  # Use the first group for colorbar
plt.colorbar(scatter, label='Temperature')

# Adjusting the legend size and position
legend = plt.legend(loc='upper right', frameon=False, fontsize=10)  # Set legend location to lower left

# Display the plot
plt.show()


t = y



 
plt.scatter(x, y,  s=8, c=t, cmap='winter') 

for i, txt in enumerate(tc):
    plt.annotate(txt, (x[i]+0.1, y[i]+0.1), color='blue',  fontsize=8)
    j = i + nos
    plt.annotate(txt, (x[j]+0.2, y[j]+0.2), color='brown', fontsize=8)
    h = i + 2 * nos
    plt.annotate(txt, (x[h]+0.1, y[h]+0.1),  color='green', fontsize=8)


plt.colorbar()

plt.title(r'PCA')

plt.xlabel('PCA1')

plt.ylabel('PCA2')



plt.show()



#Computing scores with Silhouette method for PCA
#Finding the optimal number of clusters 

range_n_clusters = list (range(2,10))
#print ("Number of clusters from 2 to 9: \n", range_n_clusters)




z = X_2d

silhouette_scores = []

for n_c in range_n_clusters:
    print(n_c)
    cluster = KMeans(n_clusters=n_c, random_state=seed, n_init=10)  # Adjust n_init as needed
    preds = cluster.fit_predict(z)
    score = silhouette_score(z, preds, metric='euclidean')
    silhouette_scores.append(score)
    print("For n_c = {}, silhouette score is {})".format(n_c, score))

# Print the maximum silhouette score and corresponding number of clusters
optimal_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
max_silhouette_score = max(silhouette_scores)

print("\nOptimal number of clusters (PCA):", optimal_n_clusters)
print("Maximum silhouette score (PCA):", max_silhouette_score)



# Use the optimal number of clusters for KMeans
optimal_cluster = KMeans(n_clusters=optimal_n_clusters, random_state=seed, n_init=10)
optimal_preds = optimal_cluster.fit_predict(z)
optimal_centers = optimal_cluster.cluster_centers_


plt.scatter(x, y, c=optimal_preds, cmap='viridis', alpha=0.5)
plt.scatter(optimal_centers[:, 0], optimal_centers[:, 1], c='red', marker='X', s=200, label='Centroids')
#plt.title(f'K-Means Clustering (PCA) (Optimal Clusters: {optimal_n_clusters})')
plt.title(f'K-Means Clustering (PCA)')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()





plt.show()

#Finding optimal number of clusters and value of silhouette score
#Data from t-SNE




# Assuming neig and neig_up are defined
neig = 5
neig_up = 20

# Your data X_train should be defined

# Fixed threshold value for perplexity
p_th = max_silhouette_score   # You can adjust this threshold as needed

# Calculate the silhouette score
range_n_clusters = list(range(2, 10))

# Initialization of KMeans (seed is used for reproducibility only)
seed = 356

# Initialize variables to store optimal values
optimal_perplexity = None
optimal_num_clusters = None
optimal_silhouette_score = float('-inf')

# Silhouette scores calculation
for i in range(neig, neig_up):
    tsne = TSNE(n_components=n_sne, init="pca", perplexity=i)
    z = tsne.fit_transform(X_train)

    for n_c in range_n_clusters:
        cluster = KMeans(n_clusters=n_c, random_state=seed, n_init="auto")
        preds = cluster.fit_predict(z)
        score = silhouette_score(z, preds, metric='euclidean')

        # Check if conditions are met
        if i >= p_th and score > optimal_silhouette_score:
            optimal_perplexity = i
            optimal_num_clusters = n_c
            optimal_silhouette_score = score

# Print optimal values
if optimal_perplexity is not None:
    print("\nOptimal Values (t-SNE):")
    print("Optimal Perplexity: {}, Optimal Number of Clusters: {}, Optimal Silhouette Score: {}".format(
        optimal_perplexity, optimal_num_clusters, optimal_silhouette_score))
else:
    print("\nNo optimal values found within the specified conditions.")






#Comparison with PCA
# Use optimal perplexity value but optimal number of clusters from PCA

per_op = optimal_perplexity
n_opt_pca = optimal_n_clusters



tsne_clusters = TSNE(n_components=n_sne, init="pca", perplexity=per_op)
z_clusters = tsne_clusters.fit_transform(X_train)
cluster = KMeans(n_clusters=n_opt_pca, random_state=seed, n_init="auto")
preds_clusters = cluster.fit_predict(z_clusters)
score_clusters = silhouette_score(z_clusters, preds_clusters, metric='euclidean')
print("\nFor perplexity = {} and n_c = {}, silhouette score is {}".format(per_op, n_opt_pca, score_clusters))



# Use the optimal number of clusters for KMeans of t-SNE



tsne = TSNE(n_components=n_sne, init="pca", perplexity=per_op)
X_tsne = tsne.fit_transform(X_train)


optimal_cluster = KMeans(n_clusters=n_opt_pca, random_state=seed, n_init="auto")
optimal_preds = optimal_cluster.fit_predict(X_tsne)
optimal_centers = optimal_cluster.cluster_centers_


# Create a scatter plot for K-Means clustering with empty markers and colors on the edges
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=optimal_preds, cmap='viridis', alpha=0.5, edgecolor='black', linewidths=1)
plt.scatter(optimal_centers[:, 0], optimal_centers[:, 1], c='red', marker='X', s=200, label='Centroids', edgecolor='black', linewidths=1)

# Adding labels and title
#plt.title(f'K-Means Clustering (t-SNE) (Optimal Clusters: {n_opt_pca})')
plt.title(f'K-Means Clustering (t-SNE)')
plt.xlabel('comp-1')
plt.ylabel('comp-2')
plt.legend()




# Display the plot
plt.show()

x = X_tsne[:, 0]
y = X_tsne[:, 1]

# Splitting the arrays into three groups
nos = 17
x_groups = [x[:nos], x[nos:2 * nos], x[2 * nos:]]
y_groups = [y[:nos], y[nos:2 * nos], y[2 * nos:]]
t_groups = [tc, tc, tc]


# Create a scatter plot for each group with the custom colormap
markers = ['s', 'o', '^']
labels = ['Fib', 'Fib + CNP', 'Fib + SiNP']



# Increase marker size
marker_size = 55


for i in range(3):
    plt.scatter(x_groups[i], y_groups[i], c=t_groups[i], marker=markers[i], s=marker_size, cmap='jet', edgecolors='black', linewidths=1,label=labels[i])

# Adding labels and title
plt.xlabel('comp-1')
plt.ylabel('comp-1')
#plt.title('t-SNE')
plt.title(f't-SNE, p={per_op}')

# Adding a color bar using the scatter plot of the first group
scatter = plt.scatter(x_groups[0], y_groups[0], c=t_groups[0], cmap='jet')  # Use the first group for colorbar
plt.colorbar(scatter, label='Temperature')

# Adjusting the legend size and position
legend = plt.legend(loc='upper right', frameon=False, fontsize=10)  # Set legend location to lower left
#plt.setp(legend.get_texts(), fontsize='10')  # Set legend text size




# Display the plot
plt.show()


t = y



 
plt.scatter(x, y,  s=8, c=t, cmap='winter') 

for i, txt in enumerate(tc):
    plt.annotate(txt, (x[i]+0.1, y[i]+0.1), color='blue',  fontsize=8)
    j = i + nos
    plt.annotate(txt, (x[j]+0.2, y[j]+0.2), color='brown', fontsize=8)
    h = i + 2 * nos
    plt.annotate(txt, (x[h]+0.1, y[h]+0.1),  color='green', fontsize=8)


plt.colorbar()

# Adding labels and title
plt.xlabel('comp-1')
plt.ylabel('comp-1')
#plt.title('t-SNE')
plt.title(f't-SNE, p={per_op}')




plt.show()


