import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers.merge import concatenate
from keras.utils import plot_model


#loading each data set
def load_dataset_x(filename):
    data = pd.read_csv(filename, header=None)
    dataset = data.values
    X = dataset.astype(str)
    return X

def load_dataset_y(filename):
    data = pd.read_csv(filename, header=None)
    dataset = data.values
    y = dataset.reshape((len(dataset),1))
    return y

#preparing input data
def prepare_inputs(X_train, X_test):
    X_train_enc, X_test_enc = list(),list()
    #label encode each columns
    for i in range(X_train.shape[1]):
        le = LabelEncoder()
        le.fit(X_train[:,i])
        #encode
        train_enc = le.transform(X_train[:,i])
        test_enc = le.transform(X_test[:,i])
        #store
        X_train_enc.append(train_enc)
        X_test_enc.append(test_enc)
    return X_train_enc, X_test_enc

#prepare targets
def prepare_targets(Y_train, Y_test):
    le = LabelEncoder()
    le.fit(Y_train)
    Y_train_enc = le.transform(Y_train)
    Y_test_enc = le.transform(Y_test)
    return Y_train_enc, Y_test_enc

#load data
X_train = load_dataset_x('x_train.csv')
X_test = load_dataset_x('x_test.csv')
Y_train = load_dataset_y('y_train.csv')
Y_test = load_dataset_y('y_test.csv')
#prepare input
X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
#prepare output
Y_train_enc, Y_test_enc = prepare_targets(Y_train, Y_test)
#make output 3d
y_train_enc = Y_train_enc.reshape((len(Y_train_enc)),1,1)
y_test_enc = Y_test_enc.reshape((len(Y_test_enc)),1,1)
#preparing input head
in_layers = list()
em_layers = list()
for i in range(len(X_train_enc)):
    #number of unique inputs
    n_labels = len(np.unique(X_train_enc[i]))
    #define input layer
    in_layer = Input(shape=(1,))
    #embedding layer
    em_layer = Embedding(n_labels, 10)(in_layer)
    #store layers
    in_layers.append(in_layer)
    em_layers.append(em_layer)
#concat all embedded layers
merge = concatenate(em_layers)
dense = Dense(10, activation='relu', kernel_initializer='he_normal')(merge)
output = Dense(1, activation='sigmoid')(dense)
model = Model(inputs=in_layers, outputs=output)
#compile keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#plot graph
plot_model(model, show_shapes=True, to_file='embeddings.png')
#fit model on data set
model.fit(X_train_enc, y_test_enc, epochs=20, batch_size=16, verbose=2)
_, accuracy = model.evaluate(X_train_enc, y_test_enc, verbose=0)
print('Accuracy: %.2f' % (accuracy*100))