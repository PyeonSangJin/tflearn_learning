import pickle
import tflearn
from tflearn.data_augmentation import ImageAugmentation

X = pickle.load(open("train_x.pickle","rb"))
y = pickle.load(open("train_y.pickle","rb"))
test_x = pickle.load(open("test_x.pickle","rb"))
test_y = pickle.load(open("test_y.pickle","rb"))

X = X/255.0

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_flip_updown()
img_aug.add_random_rotation(max_angle=25.)


def inception(input, n1, n2, n3, n4, n5, n6) :
	reduce1 = tflearn.conv_2d(input, n1, 1, activation='relu')
	reduce2 = tflearn.conv_2d(input, n2, filter_size=1, activation='relu')
	pool = tflearn.max_pool_2d(input, kernel_size=3, strides=1)
	conv1 = tflearn.conv_2d(input, n3, 1, activation='relu')
	conv2 = tflearn.conv_2d(reduce1, n4, filter_size=3, activation='relu')
	conv3 = tflearn.conv_2d(reduce2, n5, filter_size=5, activation='relu')
	conv4 = tflearn.conv_2d(pool, n6, filter_size=1, activation='relu')
	output = tflearn.merge([conv1, conv2, conv3, conv4], mode='concat', axis=3)
	return output

input = tflearn.input_data(shape=[None, 227, 227, 3], data_augmentation=img_aug)
conv1 = tflearn.conv_2d(input, 64, 7, strides=2, activation='relu')
m1 = tflearn.max_pool_2d(conv1, 3, strides=2)
l1 = tflearn.local_response_normalization(m1)
conv2 = tflearn.conv_2d(l1, 64, 1, activation='relu')
conv3 = tflearn.conv_2d(conv2, 192, 3, activation='relu')
l2 = tflearn.local_response_normalization(conv3)
m2 = tflearn.max_pool_2d(l2, kernel_size=3, strides=2)
i1 = inception(m2, 96, 16, 64, 128, 32, 32)
i2 = inception(i1, 128, 32, 128, 192, 96, 64)
m3 = tflearn.max_pool_2d(i2, kernel_size=3, strides=2)
i3 = inception(m3, 96, 16, 192, 208, 48, 64)
i4 = inception(i3, 112, 24, 160, 224, 64, 64)
i5 = inception(i4, 128, 24, 128, 256, 64, 64)
i6 = inception(i5, 144, 32, 112, 288, 64, 64)
i7 = inception(i6, 160, 32, 256, 320, 128, 128)
m4 = tflearn.max_pool_2d(i7, kernel_size=3, strides=2)
i8 = inception(m4, 160, 32, 256, 320, 128, 128)
i9 = inception(i8, 192, 48, 384, 384, 128, 128)
a1 = tflearn.avg_pool_2d(i9, kernel_size=7, strides=1)
d1 = tflearn.dropout(a1, 0.4)
f1 = tflearn.fully_connected(d1, 6, activation='softmax')
network = tflearn.regression(f1, optimizer='momentum',
	loss='categorical_crossentropy', learning_rate=0.001)

# 학습
model = tflearn.DNN(network, checkpoint_path='model_googlenet',
	max_checkpoints=1, tensorboard_verbose=3)

model.load("model_googlenet-24000")

model.fit(X, y, n_epoch=100, validation_set=0.1, shuffle=True,
	show_metric=True, batch_size=16, snapshot_step=200,
	snapshot_epoch=False, run_id='googlenet_trash6')

model.save('googlenet-CNN.model')

acc_train = model.evaluate(X,y,16)
acc_test = model.evaluate(test_x, test_y,16)
print("train : " + str(acc_train) + ", " + "test : " + str(acc_test))