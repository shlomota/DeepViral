import matplotlib.pyplot as plt

train_acc = [0.7802, 0.8695, 0.8837, 0.8922, 0.9011, 0.9074, 0.9149, 0.9209, 0.9252, 0.9309]
train_loss = [0.4578, 0.3221, 0.2869, 0.2618, 0.2422, 0.2269, 0.2122, 0.1991, 0.1890, 0.1778]
val_acc = [0.8461, 0.8406, 0.8687, 0.8689, 0.8683, 0.8733, 0.8766, 0.8780, 0.8764, 0.8766]
val_loss = [0.3720, 0.4333, 0.2349, 0.2149, 0.2173, 0.2169, 0.1938, 0.2018, 0.1670, 0.1641]

plt.plot(train_acc)
plt.plot(val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(train_loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

