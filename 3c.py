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
                if column == 'HomePlanet': 
                      dataset[column] = dataset[column].replace(np.NaN,'Earth')
                      dataset[column] = dataset[column].replace('Earth', 1)
                      dataset[column] = dataset[column].replace('Europa', 2)
                      dataset[column] = dataset[column].replace('Mars', 3)
                if column == 'CryoSleep': 
                      dataset[column] = dataset[column].replace(np.NaN,'FALSE')
                      dataset[column] = dataset[column].replace('FALSE', 1)
                      dataset[column] = dataset[column].replace('TRUE', 2)
                if column == 'Cabin': 
                      dataset[column] = dataset[column].replace(np.NaN,'E/0/S')
                      for text in dataset[column]:
                            x = text.split("/")
                            a = x[0]
                            b = x[1]
                            c = x[2]
                            if(a == 'A'): a = 1
                            if(a == 'B'): a = 2
                            if(a == 'C'): a = 3
                            if(a == 'D'): a = 4
                            if(a == 'E'): a = 5
                            if(a == 'F'): a = 6
                            if(a == 'G'): a = 7
                            if(a == 'T'): a = 7
                            if(c == 'S'): c = 1
                            if(c == 'P'): c = 2
                            br = int(a)*100 + int(b)*10 + int(c)
                            dataset[column] = dataset[column].replace(text, br)
                if column == 'Destination': 
                      dataset[column] = dataset[column].replace(np.NaN,'TRAPPIST-1e')
                      dataset[column] = dataset[column].replace('TRAPPIST-1e',1)
                      dataset[column] = dataset[column].replace('PSO J318.5-22',2)
                      dataset[column] = dataset[column].replace('55 Cancri e',3)
                if column == 'VIP': 
                      dataset[column] = dataset[column].replace(np.NaN,'FALSE')
                      dataset[column] = dataset[column].replace('FALSE', 11)
                      dataset[column] = dataset[column].replace('TRUE', 22)
                if column == 'Name': 
                      dataset[column] = dataset[column].replace(np.NaN,'No name')
        else:
                dataset[column] = dataset[column].replace(0, np.NaN)
                mean = int(dataset[column].mean(skipna=True))
                dataset[column] = dataset[column].replace(np.NaN,mean)

X = dataset.iloc[:,0:12] #za trening
y = dataset.iloc[:,13]  #za odgovor
X_train,X_test,y_train,y_test= train_test_split(X,y,random_state=0,test_size=0.2)

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
# lose stanje je k=1 sa preciznoscu 71,8%
# najbolje stanje je k=13 sa preciznoscu 76%
# Kada povecamo parametre za trening dobijamo jos preciznije testiranje, u 
# ovo slucaju imamo da je precizost veca u odnosu na prosli zadatak.
# Primecujemo grupu tacaka od k=4 do k=50 koje se krecu od 74% do 76%,
# dok je predhodna grupa bila tek pocela od k=12 do k=50 -> od 70% do 71%.
# Time vidimo da se u ovo primeru desava da nase k dolazi do idealnog stanje
# tek nakon 4 tacke, a prethodno je pocelo tek od 12 tacke, sve ispot toga su greske.
# Sto vise parametra, to bolje resenje.