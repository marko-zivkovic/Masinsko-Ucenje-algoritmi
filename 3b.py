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

zero_not_accepted= ['HomePlanet','CryoSleep','Cabin','Destination','VIP','Name',
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
X_train,X_test,y_train,y_test= train_test_split(X,y,random_state=0,test_size=0.2)
#print(dataset.head())
nb_train = len(y_train)
nb_test = len(y_test)
#skaliranje vrednosti
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#definisanje modela knn, n_neighbors ----> K
acc = []
ks = []
for x in range(50):
   k = x + 1
   knn = KNeighborsClassifier(n_neighbors=k, p = 2, metric='euclidean')
   #fit modela
   knn.fit(X_train,y_train)
   #predpostavi test-set resenje
   y_pred = knn.predict(X_test)
   y_pred
   #  proceniti model - testiranje modela
   cm = confusion_matrix(y_test, y_pred)
   acc.append(accuracy_score(y_test,y_pred))
   ks.append(k)

plt.scatter(ks,acc)
plt.ylabel('Accuracy')
plt.xlabel('K')
plt.title('Example 2D graph')
plt.show()
#KOMENTAR
#Na garfu vidimo da je najmanja vrednost upravo u k=1 koja iznoxi oko 67% preciznosti, 
#to se desava jer Kada je premalo k više se hvata za outliere pa pravi greške.
#Najbolje resenje je sigurno u k=15 koje iznosi oko 71% preciznosti, 
#takodje su dobri i k = [14,22,45] => preko 70%