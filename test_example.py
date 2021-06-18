import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from tensorflow.keras import layers, Sequential ,callbacks,metrics,losses,optimizers
from keras.utils.np_utils import to_categorical
from plot_assest import Plot_Classification


x,y = make_circles(n_samples=300,noise=0.02,random_state=42)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)

yc_train = to_categorical(y_train)
yc_test = to_categorical(y_test)

model = Sequential()
model.add(layers.InputLayer(input_shape=(x_train.shape[1],)))
model.add(layers.Dense(10,activation='tanh'))
model.add(layers.Dense(10,activation='tanh'))
model.add(layers.Dense(2, activation='softmax'))


weights_history = []
weights_call = callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: weights_history.append(model.get_weights()))
loss = losses.CategoricalCrossentropy()
Optimizer = optimizers.Adam(learning_rate=0.01)
earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=50, min_delta=0.01)
model.compile(optimizer=Optimizer, loss=loss, metrics=['accuracy'])
history = model.fit(x_train, yc_train, epochs=50, batch_size=20,validation_split=0.2, verbose=0,shuffle = False,callbacks=[weights_call,earlyStopping])


Plot_Classification(model,x_train,y_train,weights_history,history)