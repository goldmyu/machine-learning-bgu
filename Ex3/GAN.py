import gc
import os
import time

import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras import backend as K
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import initializers
from sklearn.model_selection import train_test_split

# =========================== configurations ===============================

dataset_dir = "./data-sets"
label_column = 'label'


# ======================================= Class definitions ===================================================


class GanGenerator:

    def __init__(self, train_x, rsize, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.rsize = rsize
        self.batch_size = 128
        self.epochs = 200
        self.random_noise_vector_dim = 100

    def generator(self, optimizer, output_shape):
        generator = Sequential()
        generator.add(Dense(256, input_dim=self.random_noise_vector_dim,
                            kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        generator.add(LeakyReLU(0.2))

        generator.add(Dense(512))
        generator.add(LeakyReLU(0.2))

        generator.add(Dense(1024))
        generator.add(LeakyReLU(0.2))

        generator.add(Dense(output_shape, activation='tanh'))
        generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        return generator

    def discriminator(self, optimizer, input_dim_shape):
        discriminator = Sequential()
        discriminator.add(
            Dense(1024, input_dim=input_dim_shape, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Dense(512))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Dense(256))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))

        discriminator.add(Dense(1, activation='sigmoid'))
        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
        return discriminator

    def generate(self):
        K.set_session(tf.Session(config=tf.ConfigProto()))

        self.train_x = self.train_x.values
        x_train, y_train, x_test, y_test = train_test_split(self.train_x, self.train_y, test_size=0.3)
        print("The model train data shape is {} ".format(x_train.shape))

        # Split the training data into batches of 128
        num_of_iterations = self.train_x.shape[0] // self.batch_size

        # create the GAN netowrk
        adam = Adam(lr=0.0002, beta_1=0.5)
        generator = self.generator(adam, x_train.shape[1])
        discriminator = self.discriminator(adam, x_train.shape[1])
        gan = self.create_gan_network(discriminator, generator, adam)

        # Start training loop, every epoch traines on a batch of data
        for epoch in range(1, self.epochs + 1):
            print('=' * 20, 'Starting Epoch %d/%d' % (epoch, self.epochs), '=' * 20)
            for iteration in range(num_of_iterations):
                # Get a random set of input noise and images
                noise = np.random.normal(0, 1, size=[self.batch_size, self.random_noise_vector_dim])
                real_data_batch = x_train[np.random.randint(0, x_train.shape[0], size=self.batch_size)]

                generated_data_batch = generator.predict(noise)
                X = np.concatenate([real_data_batch, generated_data_batch])

                # Labels for generated and real data
                y_discriminator = np.zeros(2 * self.batch_size)
                # One-sided label smoothing
                y_discriminator[:self.batch_size] = 0.9

                # Train discriminator
                discriminator.trainable = True
                discriminator.train_on_batch(X, y_discriminator)

                # Train generator
                noise = np.random.normal(0, 1, size=[self.batch_size, self.random_noise_vector_dim])
                y_gen = np.ones(self.batch_size)
                discriminator.trainable = False
                gan.train_on_batch(noise, y_gen)

                print('=' * 20, 'Iteration %d/%d' % (iteration, num_of_iterations), '=' * 20)

        noise = np.random.normal(0, 1, size=[int(self.rsize), int(self.random_noise_vector_dim)])
        generated_x = generator.predict(noise)

        del generator
        del gan
        del discriminator
        gc.collect()
        return pd.DataFrame(generated_x)

    def create_gan_network(self, discriminator, generator, optimizer):

        # set trainable to False as we want to train only the generator or discriminator at a time
        discriminator.trainable = False

        # gan random noise input will be a 100-dim vector
        generator_input = Input(shape=(self.random_noise_vector_dim,))

        # the output of the generator - generated data
        generated_data = generator(generator_input)

        # get the output of the discriminator (probability if the image is real or not)
        disc_output = discriminator(generated_data)

        gan = Model(inputs=generator_input, outputs=disc_output)
        gan.compile(loss='binary_crossentropy', optimizer=optimizer)
        return gan


# ========================== Main code ================================================


def generate_samples(num_samples_to_add, x_train, y_train, index_start):
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_scale = scaler.transform(x_train)

    gan_generator = GanGenerator(train_x=pd.DataFrame(x_scale), rsize=int(num_samples_to_add), train_y=y_train)
    x_train_gan = gan_generator.generate()

    x_train_gan = scaler.inverse_transform(x_train_gan)
    x_train_gan = pd.DataFrame(x_train_gan)
    x_train_gan.index = x_train_gan.index + index_start
    return x_train_gan


# def enumerate_classes(name):
#     return {'iris_csv.csv': {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2},
#             }.get(name, 'no_mapping')


def factorize_dataset(data):
    x = data.copy()
    categorical_columns = x.select_dtypes(include='object').columns
    for i, col in enumerate(categorical_columns):
        label_encoder = LabelEncoder()
        label_encoder.fit(x[col])
        x[col] = label_encoder.transform(x[col])
    return pd.DataFrame(x)


def main(rsize):
    all_files = os.listdir(dataset_dir)
    for file_name in all_files:
        print("Start working with data-set: {}".format(file_name))
        file_path = os.path.join(dataset_dir, file_name)
        if os.path.isfile(file_path):
            dataset_start_train_time = time.time()
            data = pd.read_csv(file_path)
            # mapping = enumerate_classes(file_name)
            # if mapping != 'no_mapping':
            #     data = data.replace(mapping)
            data = factorize_dataset(data)

            y = data[label_column]
            x = data.drop(label_column, axis=1)
            if x.select_dtypes(include=[np.object]).empty:
                cols = x.columns
                x_new_samples = generate_samples(rsize * len(x) * 10, x, y, len(x))
                x_new_samples.columns = cols

                if not os.path.exists("generated_data/"):
                    os.makedirs("generated_data/")

                x_new_samples.to_csv("generated_data/generated_{}".format(file_name), index=False)
                print("Finished training GAN and generating data for dataset %s\ntime it took was: %.3f" % (
                    file_name, time.time() - dataset_start_train_time))


# all_files = os.listdir(dataset_dir)
# for file_name in all_files:
#     if file_name == 'Admission_Predict.csv':
#         print("Start working with data-set: {}".format(file_name))
#         file_path = os.path.join(dataset_dir, file_name)
#         if os.path.isfile(file_path):
#             data = pd.read_csv(file_path)
#             data.label = data.label.round().astype(int)
#             data.to_csv(path_or_buf="./data-sets/Admission_Predict_classification.csv",index=False)

print("Finished synthesizing data using GAN")
