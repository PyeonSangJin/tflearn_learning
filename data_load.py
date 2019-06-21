import numpy as np
import os
import cv2
import pickle
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

DATADIR = ".\\dataset-original"
CATEGORIES = ["cardboard","glass","metal","paper","plastic","trash"]
IMG_SIZE = 227

training_data = []
training_label = []

label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)

integer_encoded = label_encoder.fit_transform(CATEGORIES)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        # class_num = onehot_encoded[category]
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, onehot_encoded[class_num]])
                print(len(training_data))
            except Exception as e:
                pass

create_training_data()
random.shuffle(training_data)

x = []
y = []

for feature, label in training_data:
    x.append(feature)
    y.append(label)
    print(len(x))

x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
y = np.array(y).reshape(-1, 6)

pickle_out = open("X.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)

pickle_out = open("train_x.pickle","wb")
pickle.dump(x_train, pickle_out)
pickle_out.close()

pickle_out = open("test_x.pickle","wb")
pickle.dump(x_test, pickle_out)
pickle_out.close()

pickle_out = open("train_y.pickle","wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

pickle_out = open("test_y.pickle","wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()

#tensorboard --logdir=logs/
