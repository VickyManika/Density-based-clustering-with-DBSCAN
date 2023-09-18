# #Μέσα στον κώδικα ακολοθούνται τα βήματα της εργασίας.
# Ενσωμάτωση των απαραίτητων βιβλιοθηκών
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

#Βήμα 1
mat_file = scipy.io.loadmat('mydata.mat') # Φόρτωση των δεδομένων
X = np.array(mat_file['X']) # Δημιουργία πίνακα Χ με τα δεδομένα του αρχείου

#Βήμα 2
dbscan = DBSCAN(eps = 0.5, min_samples = 15).fit(X) # Εφαρμογή της μεθόδου DBSCAN στον πίνακα Χ με eps=0.5 και min_samples(MinPts)=15
IDX = dbscan.labels_ # Ετικέτες κάθε σημείου της μεθόδου DBSCAN
#Bήμα 3
plt.figure(1)
plt.scatter(X[:,0],X[:,1]) #Γράφημα διασποράς των τιμών του πίνακα Χ 
plt.show()

#Βήμα 4
plt.figure(2)
plt.scatter(X[:,0],X[:,1], c=IDX) # Γράφημα με τις συστάδες στις οποίες χώρισε τα δεδομένα η μέθοδος DBSCAN και ο θόρυβος.
plt.show()

