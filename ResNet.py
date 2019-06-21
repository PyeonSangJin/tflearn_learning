import pickle
import tflearn

X = pickle.load(open("train_x.pickle","rb"))
Y = pickle.load(open("train_y.pickle","rb"))
test_x = pickle.load(open("test_x.pickle","rb"))
test_y = pickle.load(open("test_y.pickle","rb"))

X = X/255.0

X = X.reshape([-1, 227, 227, 3])
# 잔차 네트워크 구성
net = tflearn.input_data(shape=[None, 227, 227,3])
net = tflearn.conv_2d(net, 64, 3, activation='relu', bias=False)
# 잔차 블록
net = tflearn.residual_bottleneck(net, 3, 16, 64)
net = tflearn.residual_bottleneck(net, 1, 32, 128, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 32, 128)
net = tflearn.residual_bottleneck(net, 1, 64, 256, downsample=True)
net = tflearn.residual_bottleneck(net, 2, 64, 256)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
net = tflearn.fully_connected(net, 6, activation='softmax')
net = tflearn.regression(net, optimizer='momentum',
	loss='categorical_crossentropy', learning_rate=0.1)

# 학습
model = tflearn.DNN(net, checkpoint_path='model_regnet',
	max_checkpoints=10, tensorboard_verbose=3)

model.load("model_regnet-9009")

# model.fit(X, Y, n_epoch=100, validation_set=0.1, shuffle=True,
# 	show_metric=True, batch_size=16, run_id='regnet_trash5')
#
# model.save('regnet-CNN.model')

acc_train = model.evaluate(X,Y,16)
acc_test = model.evaluate(test_x, test_y,16)
print("train : " + str(acc_train) + ", " + "test : " + str(acc_test))