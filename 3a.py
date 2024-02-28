import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


dataset = pd.read_csv('spaceship-titanic.csv')
print(len(dataset))
print(dataset.head())

zero_not_accepted = ['HomePlanet','CryoSleep','Cabin','Destination','VIP','Name',
                    'Age','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']
stringss = ['HomePlanet','CryoSleep','Cabin','Destination','VIP','Name']
for column in zero_not_accepted:
        if column in stringss:
                dataset[column] = dataset[column].replace('', np.NaN)
                if column == 'HomePlanet': dataset[column] = dataset[column].replace(np.NaN,'Earth')
                if column == 'CryoSleep': dataset[column] = dataset[column].replace(np.NaN,'FALSE')
                if column == 'Cabin': dataset[column] = dataset[column].replace(np.NaN,'E/0/S')
                if column == 'Destination': dataset[column] = dataset[column].replace(np.NaN,'TRAPPIST-1e')
                if column == 'VIP': dataset[column] = dataset[column].replace(np.NaN,'FALSE')
                if column == 'Name': dataset[column] = dataset[column].replace(np.NaN,'No name')
        else:
                dataset[column] = dataset[column].replace(0, np.NaN)
                mean = int(dataset[column].mean(skipna=True))
                dataset[column] = dataset[column].replace(np.NaN,mean)

X = dataset.iloc[:,7:9] #za trening
y = dataset.iloc[:,13]  #za odgovor
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.2)
#print(dataset.head())
nb_train = len(y_train)
nb_test = len(y_test)
print(nb_train, nb_test)
#skaliranje vrednosti
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#definisanje modela knn, n_neighbors ----> K
knn = KNeighborsClassifier(n_neighbors=15, p = 2, metric='euclidean')
#fit modela
knn.fit(X_train,y_train)
#predpostavi test-set resenje
y_pred = knn.predict(X_test)
#proceniti model - testiranje modela
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('f1 score')
print(f1_score(y_test,y_pred))
print('accuracy')
print(accuracy_score(y_test,y_pred))

h = 0.02  # step size in the mesh
x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Predict the class labels for all points in the mesh grid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
# Reshape the predicted labels into a mesh grid
Z = Z.reshape(xx.shape)
# Plot the decision boundary and the data points
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdYlBu_r)
plt.xlabel('plavo = transportovani   crveno = netransportovani')
plt.ylabel(' ')
plt.title('KNN 2D grafik na kome su prikazani trening podaci')
plt.show()