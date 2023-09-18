# #Μέσα στον κώδικα ακολοθούνται τα βήματα της εργασίας.
# #Ενσωμάτωση των απαραίτητων βιβλιοθηκών
# import scipy.io
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN

# #Βήμα 1
# mat_file = scipy.io.loadmat('xV.mat') # Φόρτωση δεδομένων xV που υπάρχουν στο αρχείο
# xV = np.array(mat_file['xV']) # Δημιουργία πίνακα xV με τα δεδομένα του αρχείου

# #Bήμα 2 
# X=xV[:,[0,1]] # Απο τον πίνακα xV χρησιμοποιούμε τα δύο πρώτα χαρακτηριστικά


# dbscan = DBSCAN(eps = 0.3, min_samples = 50).fit(X) # Eφαρμογή της μεθόδου DBSCAN χρησιμοποιώντας τις παραμέτορoυς του αλγορίθμου eps=0.3 και min_samples(MinPts)=50. 
# IDX = dbscan.labels_ # Ετικέτες κάθε σημείου της μεθόδου DBSCAN

# plt.figure(1)
# plt.scatter(X[:,0],X[:,1]) #Γράφημα διασποράς των τιμών του πίνακα Χ 
# plt.show()

# plt.figure(2)
# plt.scatter(X[:,0],X[:,1], c=IDX) # Γράφημα με τις συστάδες στις οποίες χώρισε τα δεδομένα η μέθοδος DBSCAN και ο θόρυβος.
# plt.show()

# #Bήμα 3
# #Ενσωμάτωση των απαραίτητων βιβλιοθηκών
# import scipy.io
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN

# mat_file = scipy.io.loadmat('xV.mat') # Φόρτωση δεδομένων xV που υπάρχουν στο αρχείο
# xV = np.array(mat_file['xV']) # Δημιουργία πίνακα xV με τα δεδομένα του αρχείου

# X=xV[:,[0,1]] # Απο τον πίνακα xV χρησιμοποιούμε τα δύο πρώτα χαρακτηριστικά


# dbscan = DBSCAN(eps = 0.2, min_samples =9.5).fit(X) # Eφαρμογή της μεθόδου DBSCAN χρησιμοποιώντας τις παραμέτρoυς του αλγορίθμου eps=0.1 και min_samples(MinPts)=5.
# IDX = dbscan.labels_ # Ετικέτες κάθε σημείου της μεθόδου DBSCAN

# plt.figure(1)
# plt.scatter(X[:,0],X[:,1]) #Γράφημα διασποράς των τιμών του πίνακα Χ 
# plt.show()

# plt.figure(2)
# plt.scatter(X[:,0],X[:,1], c=IDX) # Γράφημα με τις συστάδες στις οποίες χώρισε τα δεδομένα η μέθοδος DBSCAN και ο θόρυβος.
# plt.show()

#Βήμα 4
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

mat_file = scipy.io.loadmat('xV.mat')
xV = np.array(mat_file['xV'])

X=xV[:,[467,468]] # Απο τον πίνακα xV χρησιμοποιούμε τα δύο τελευταία χαρακτηριστικά


dbscan = DBSCAN(eps = 0.03, min_samples =6).fit(X) # Eφαρμογή της μεθόδου DBSCAN χρησιμοποιώντας διάφορες τιμες στις παραμέτρoυς του αλγορίθμου eps, min_samples(MinPts) 
IDX = dbscan.labels_ # Ετικέτες κάθε σημείου της μεθόδου DBSCAN

plt.figure(1)
plt.scatter(X[:,0],X[:,1]) #Γράφημα διασποράς των τιμών του πίνακα Χ 
plt.show()

plt.figure(2)
plt.scatter(X[:,0],X[:,1], c=IDX) # Γράφημα με τις συστάδες στις οποίες χώρισε τα δεδομένα η μέθοδος DBSCAN και το θόρυβος.
plt.show()












