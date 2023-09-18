# #Μέσα στον κώδικα ακολοθούνται τα βήματα της εργασίας.
# #Βήμα 1
# from sklearn.datasets import load_iris # Ενσωμάτωση βιβλιοθήκης
# means=load_iris().data # Φόρτωση δεδομένων iris που υπάρχουν στο αρχείο
# X = means[:,[2,3]] # Επιλογή των διαστάσεων [2,3] του πίνακα Χ

# #Βήμα 2
# from sklearn.datasets import load_iris 
# means=load_iris().data # Φόρτωση δεδομένων iris που υπάρχουν στο αρχείο
# X = means[:,[0,1]] # Επιλογή των δύο πρώτων διαστάσεων του πίνακα Χ

# from sklearn.cluster import DBSCAN 

# dbscan = DBSCAN(eps = 0.1, min_samples= 5).fit(X) # Eφαρμογή της μεθόδου DBSCAN χρησιμοποιώντας τις παραμέτρoυς του αλγορίθμου eps=0.1 και min_samples(MinPts)=5.  
# IDX = dbscan.labels_ # Ετικέτες κάθε σημείου της μεθόδου DBSCAN

# #Βήμα 3 
# import matplotlib.pyplot as plt  # Ενσωμάτωση βιβλιοθήκης matplotlib

# plt.figure(1)
# plt.scatter(X[:,0],X[:,1]) #Γράφημα διασποράς των τιμών του πίνακα Χ 
# plt.show()

# #Βήμα 4 
# plt.figure(2)
# plt.scatter(X[:,0],X[:,1], c=IDX) # Γράφημα με τις συστάδες στις οποίες χώρισε τα δεδομένα η μέθοδος DBSCAN και ο θόρυβος.
# plt.show()

#Βήμα 5 
#Eνσωμάτωση των απαραίτητων βιβλιοθηκών
from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore

means=load_iris().data # Φόρτωση των δεδομένων iris

X = means[:, [0, 1]] # Eπιλογή των δύο πρώτων διαστάσεων
xV1 = zscore(X[:, 0]) # Κανονικοποίηση των δεδομένων για την πρώτη διάσταση 
xV2 = zscore(X[:, 1]) # Κανονικοποίηση των δεδομένων για την δεύτερη διάσταση

X = np.array([xV1, xV2]).T # Δημιουργία πίνακα με τα κανονικοποιημένα δεδομένα και μετατροπή αυτού απο 2x150 σε 150x2 διάσταση

dbscan = DBSCAN(eps=0.6, min_samples=14).fit(X) # Eφαρμογή της μεθόδου DBSCAN χρησιμοποιώντας τις παραμέτρoυς του αλγορίθμου eps=0.1 και min_samples(MinPts)=5.

IDX = dbscan.labels_ # Ετικέτες κάθε σημείου της μεθόδου DBSCAN

plt.figure(1)
plt.scatter(xV1,xV2) #Γράφημα διασποράς των τιμών του πίνακα Χ 

plt.show()

plt.figure(2)
plt.scatter(xV1,xV2, c=IDX) # Γράφημα με τις συστάδες στις οποίες χώρισε τα δεδομένα η μέθοδος DBSCAN και ο θόρυβος.
plt.show()









