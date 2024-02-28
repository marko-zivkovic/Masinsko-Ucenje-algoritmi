%matplotlib inline
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Pomocna funkcija koja od niza trening primera pravi feature matricu (m X n).
def create_feature_matrix(x, nb_features):
    tmp_features = []
    for deg in range(1, nb_features + 1):
        tmp_features.append(np.power(x, deg))
    return np.column_stack(tmp_features)



# Učitavanje i obrada podataka.
filename = 'funky.csv'
all_data = np.loadtxt(filename, delimiter=',', usecols=(0, 1), dtype='float32')
data = dict()
data['x'] = all_data[:, 0]
data['y'] = all_data[:, 1]

# Nasumično mešanje.
nb_samples = data['x'].shape[0]
indices = np.random.permutation(nb_samples)
data['x'] = data['x'][indices]
data['y'] = data['y'][indices]

# Normalizacija (obratiti pažnju na axis=0).
data['x'] = (data['x'] - np.mean(data['x'], axis=0)) / np.std(data['x'], axis=0)
data['y'] = (data['y'] - np.mean(data['y'])) / np.std(data['y'])

nb_features = 3

data['x'] = create_feature_matrix(data['x'], nb_features)

# Iscrtavanje.
plt.scatter(data['x'][:, 0], data['y'])
plt.xlabel('x')
plt.ylabel('y')

# Model i parametri.
w = tf.Variable(tf.zeros(nb_features))
b = tf.Variable(0.0)

learning_rate = 0.001
nb_epochs = 50

lambdas = [0, 0.001, 0.01, 0.1, 1, 10, 100]


def pred(x, w, b):
    w_col = tf.reshape(w, (nb_features, 1))
    hyp = tf.add(tf.matmul(x, w_col), b)
    return hyp


# Funkcija troška i optimizacija.
def loss(x, y, w, b, l, reg=None):
    prediction = pred(x, w, b)

    y_col = tf.reshape(y, (-1, 1))
    mse = tf.reduce_mean(tf.square(prediction - y_col))

    if reg == 'l1':
        l1_reg = l * tf.reduce_mean(tf.abs(w))
        loss = tf.add(mse, l1_reg)
    elif reg == 'l2':
        l2_reg = l * tf.reduce_mean(tf.square(w))
        loss = tf.add(mse, l2_reg)
    else:
        loss = mse

    return loss


# Računanje gradijenta
def calc_grad(x, y, w, b, l):
    with tf.GradientTape() as tape:
        loss_val = loss(x, y, w, b, l, reg='l2')

    w_grad, b_grad = tape.gradient(loss_val, [w, b])

    return w_grad, b_grad, loss_val



adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)


# Trening korak
def train_step(x, y, w, b, l):
    w_grad, b_grad, loss = calc_grad(x, y, w, b, l)

    adam.apply_gradients(zip([w_grad, b_grad], [w, b]))

    return loss


costs = []

for lam in lambdas:
    for epoch in range(nb_epochs):

        # Stochastic Gradient Descent.
        epoch_loss = 0
        for sample in range(nb_samples):
            x = data['x'][sample].reshape((1, nb_features))
            y = data['y'][sample]

            curr_loss = train_step(x, y, w, b, lam)
            epoch_loss += curr_loss

        # U svakoj stotoj epohi ispisujemo prosečan loss.
        epoch_loss /= nb_samples

    cost = 0
    for red in range(nb_samples):
        tr_x = data['x'][red].reshape((1, nb_features))
        tr_y = data['y'][red]

        cost += train_step(tr_x, tr_y, w, b, lam)

    costs.append(cost)

    xs = create_feature_matrix(np.linspace(-2, 4, 100, dtype='float32'), nb_features)
    hyp_val = pred(xs, w, b)  # samo umesto color prosledim niz boja
    plt.plot(xs[:, 0].tolist(), hyp_val.numpy().tolist(), color='g')

plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.show()

plt.plot(lambdas, costs, color='yellow')
plt.show()