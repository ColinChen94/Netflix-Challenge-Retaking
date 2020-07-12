
import tensorflow as tf
import hyperopt
import numpy as np
import os
import datetime
import pickle
import argparse
import glob

class IntAccuracy(tf.keras.metrics.Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_float = tf.cast(y_true, tf.float32)
        y_pred_round = tf.round(tf.clip_by_value(y_pred, 1, 5))
        self.total.assign_add(tf.reduce_sum(tf.cast(tf.math.equal(y_true_float, y_pred_round), tf.float32)))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.total / self.count

class ResultSaverCallback(tf.keras.callbacks.Callback):
    def __init__(self, model_dir):
        super(ResultSaverCallback, self).__init__()
        self.model_dir = model_dir

    def on_epoch_end(self, epoch, logs=None):
        val_rmse = logs["val_root_mean_squared_error"]
        open(os.path.join(self.model_dir, "{:03d}-{:.4f}.result".format(epoch, val_rmse))).close()

def _get_parallel_dataset(table, dataset_dir="./train", num_files=50, num_readers=20, num_threads=20, shuffle_size=100000,
                          batch_size=128, input_type="none", prefetch=1):
    filepath_list = [filepath.path for filepath in os.scandir(dataset_dir) if filepath.is_file()]
    filepaths_ds = tf.data.Dataset.list_files(filepath_list).shuffle(num_files)
    textline_ds = filepaths_ds.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath),
        cycle_length=num_readers, num_parallel_calls=num_threads)
    preprocessor = _preprocessor_generator(table, input_type=input_type)
    ds = textline_ds.map(preprocessor, num_parallel_calls=num_threads).shuffle(shuffle_size)
    return ds.batch(batch_size).prefetch(prefetch)

def _preprocessor_generator(table, input_type="user-movie"):

    @tf.function
    def preprocessor(line):
        defs = [tf.constant(17770, dtype=tf.int32, shape=[]),
                tf.constant(480189, dtype=tf.int32, shape=[]),
                tf.constant([], dtype=tf.int32),
                tf.constant("2005-12-31", tf.string)]
        fields = tf.io.decode_csv(line, record_defaults=defs)
        user_id = table.lookup(fields[1])
        movie_id = fields[0] - 1
        if input_type == "user-movie":
            X = (tf.constant(1.0), user_id, movie_id)
        elif input_type == "user-movie-continuous_time" or input_type == "user_t-movie_t":
            time_tensor = tf.strings.to_number(tf.strings.split(fields[3], sep="-"), tf.float32)
            time = (tf.reduce_sum(tf.constant([1., 1. / 12., 1. / 360.]) * (time_tensor - 1)) - 2001.5) / 3.5
            X = (tf.constant(1.0), user_id, movie_id, time)
        else:
            time_tensor = tf.strings.to_number(tf.strings.split(sep="-").to_tensor(), dtype=tf.float32)
            X = (tf.constant(1.0), user_id, movie_id, time_tensor[0]-1998, time_tensor[1]-1, time_tensor[2]-1)
        y = tf.stack(fields[2])
        return X, y
    return preprocessor

def _run(train_ds, val_ds, *args, **kwargs):

    l1_regularizer = tf.keras.regularizers.l1(l=kwargs["regularized_factor"])
    model = _get_model(480189, 17770, embedding_regularizer=l1_regularizer,
                       latent_dim=kwargs["latent_dim"], input_type=kwargs["input_type"],
                       network=kwargs["network"], num_neurons=kwargs["num_neurons"])

    init_epoch = 0
    if kwargs["load_dir"] is not None:
        files = sorted(glob.glob(os.path.join(kwargs["load_dir"], "*.result")))
        if len(files) != 0:
            init_epoch = int(files[-1].split("/")[-1].split("-")[-2])
        trial_path = kwargs["load_dir"]
    else:
        now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        trial_id = "model-" + now
        trial_path = os.path.join(kwargs["job_dir"], trial_id)
        if not os.path.exists(trial_path):
            os.mkdir(trial_path)
        print("SAVE MODEL in {}.".format(trial_path))

    if os.path.exists(os.path.join(kwargs["load_dir"], "mymodel.h5")):
        model.load_weights(os.path.join(kwargs["load_dir"], "mymodel.h5"))
        print("LOAD MODEL in {}.".format(trial_path))

    model.summary()

    patience = 3
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(trial_path, "mymodel.h5"),
                                                       save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
    result_saver_cb = ResultSaverCallback(trial_path)

    model.compile(optimizer="Adam", loss="mse",
                  metrics=[IntAccuracy(), tf.keras.metrics.RootMeanSquaredError()])
    history = model.fit(train_ds, initial_epoch=init_epoch, epochs=kwargs["epoches"], validation_data=val_ds,
                        callbacks=[checkpoint_cb, early_stopping_cb, result_saver_cb])

    print("VAL RMSE: {}.".format(history.history["val_root_mean_squared_error"][-(patience + 1)]))
    return {"loss": history.history["val_root_mean_squared_error"][-(patience + 1)],
            "model_dir": trial_path,
            "status": hyperopt.STATUS_OK,
            "attachment": {
                "return": pickle.dumps(history.history)
            }}


def _get_model(num_users, num_movies, latent_dim=10, embedding_regularizer=None, num_neurons=(40, 20, 10),
               input_type="user-movie", network="wide-deep"):
    # TODO: Implement a custom layer to make it organized

    input_dummy = tf.keras.Input(shape=(1), dtype=tf.float32)
    input_users = tf.keras.Input(shape=(), dtype=tf.int32)
    input_movies = tf.keras.Input(shape=(), dtype=tf.int32)
    # one_hot_users = tf.keras.layers.Lambda(lambda user_id: tf.one_hot(user_id, num_users))(input_users)
    # one_hot_movies = tf.keras.layers.Lambda(lambda movie_id: tf.one_hot(movie_id, num_movies))(input_movies)
    # embedding_users = tf.keras.layers.Dense(latent_dim, use_bias=False, kernel_regularizer=embedding_regularizer)(
    #     one_hot_users)
    # embedding_movies = tf.keras.layers.Dense(latent_dim, use_bias=False, kernel_regularizer=embedding_regularizer)(
    #     one_hot_movies)

    weight_users = tf.keras.layers.Embedding(num_users, 1, embeddings_regularizer=embedding_regularizer)(input_users)
    weight_movies = tf.keras.layers.Embedding(num_movies, 1, embeddings_regularizer=embedding_regularizer)(input_movies)
    bias = tf.keras.layers.Dense(1, use_bias=False, kernel_regularizer=embedding_regularizer)(input_dummy)

    embedding_users = tf.keras.layers.Embedding(num_users, latent_dim, embeddings_regularizer=embedding_regularizer)(
        input_users)
    embedding_movies = tf.keras.layers.Embedding(num_movies, latent_dim, embeddings_regularizer=embedding_regularizer)(
        input_movies)

    if input_type == "user-movie":
        input_list = [input_dummy, input_users, input_movies]
        # one_hot_input_list = [one_hot_users, one_hot_movies]

        add_input_list = [weight_users, weight_movies, bias]
        embedding_list = [embedding_users, embedding_movies]
        embedding_time_weight_list = []
        deep_input_list = embedding_list

    elif input_type == "user-movie-continuous_time":
        input_time = tf.keras.Input(shape=(1), dtype=tf.float32)
        input_list = [input_dummy, input_users, input_movies, input_time]

        weight_time = tf.keras.layers.Dense(1, use_bias=False, kernel_regularizer=embedding_regularizer)(input_time)
        add_input_list = [weight_users, weight_movies, weight_time, bias]
        # one_hot_input_list = [one_hot_users, one_hot_movies, input_time]

        embedding_time = tf.keras.layers.Dense(latent_dim, use_bias=False, kernel_regularizer=embedding_regularizer)(input_time)
        embedding_list = [embedding_users, embedding_movies, embedding_time]
        embedding_time_weight_list = []
        deep_input_list = embedding_list

    elif input_type == "user_t-movie_t":
        input_time = tf.keras.Input(shape=(1), dtype=tf.float32)
        input_list = [input_dummy, input_users, input_movies, input_time]
        bias_time = tf.keras.layers.Dense(1, use_bias=False, kernel_regularizer=embedding_regularizer)(input_time)
        weight_user_time_slope = tf.keras.layers.Embedding(num_users, 1, embeddings_regularizer=embedding_regularizer)(input_users)
        weight_movie_time_slope = tf.keras.layers.Embedding(num_movies, 1, embeddings_regularizer=embedding_regularizer)(
            input_movies)
        weight_user_time = tf.keras.layers.Dot(axes=1)([weight_user_time_slope, input_time])
        weight_movie_time = tf.keras.layers.Dot(axes=1)([weight_movie_time_slope, input_time])
        add_input_list = [weight_users, weight_movies, weight_user_time, weight_movie_time, bias, bias_time]
        embedding_list = [embedding_users, embedding_movies]
        embedding_user_time_weight = tf.keras.layers.Dense(latent_dim, use_bias=False, kernel_regularizer=embedding_regularizer)(
            input_time)
        embedding_movie_time_weight = tf.keras.layers.Dense(latent_dim, use_bias=False, kernel_regularizer=embedding_regularizer)(
            input_time)
        embedding_time_weight_list = [embedding_user_time_weight, embedding_movie_time_weight]

        deep_input_list = [embedding_users, embedding_movies, input_time]

    else:
        input_year = tf.keras.Input(shape=(), dtype=tf.int32)
        input_month = tf.keras.Input(shape=(), dtype=tf.int32)
        input_day = tf.keras.Input(shape=(), dtype=tf.int32)
        input_list = [input_dummy, input_users, input_movies, input_year, input_month, input_day]

        weight_year = tf.keras.layers.Embedding(8, 1, embeddings_regularizer=embedding_regularizer)(input_users)
        weight_month = tf.keras.layers.Embedding(12, 1, embeddings_regularizer=embedding_regularizer)(input_movies)
        weight_day = tf.keras.layers.Embedding(31, 1, embeddings_regularizer=embedding_regularizer)(input_users)
        add_input_list = [weight_users, weight_movies, weight_year, weight_month, weight_day, bias]

        # one_hot_year = tf.keras.layers.Lambda(lambda year: tf.one_hot(year, 8))(input_year)
        # one_hot_month = tf.keras.layers.Lambda(lambda month: tf.one_hot(month, 12))(input_month)
        # one_hot_day = tf.keras.layers.Lambda(lambda day: tf.one_hot(day, 31))(input_day)
        # one_hot_input_list = [one_hot_users, one_hot_movies, one_hot_year, one_hot_month, one_hot_day]

        embedding_year = tf.keras.layers.Embedding(8, latent_dim, embeddings_regularizer=embedding_regularizer)(
        input_year)
        embedding_month = tf.keras.layers.Embedding(12, latent_dim, embeddings_regularizer=embedding_regularizer)(
        input_month)
        embedding_day = tf.keras.layers.Embedding(31, latent_dim, embeddings_regularizer=embedding_regularizer)(
        input_day)
        embedding_list = [embedding_users, embedding_movies, embedding_year, embedding_month, embedding_day]
        deep_input_list = embedding_list
        embedding_time_weight_list = []

    if network == "wide":
        order_1 = tf.keras.layers.Add()(add_input_list)
        dot_list = []
        for i in range(len(embedding_list)):
            for j in range(i+1, len(embedding_list)):
                dot_list.append(tf.keras.layers.Dot(axes=1)([embedding_list[i], embedding_list[j]]))
        dot_list.append(order_1)

        if len(embedding_time_weight_list) == len(embedding_list):
            for embedding, time_weight in zip(embedding_list, embedding_time_weight_list):
                dot_list.append(tf.keras.layers.Dot(axes=1)([embedding, time_weight]))
        raw_rating = tf.keras.layers.Add()(dot_list)

    elif network == "deep":
        previous_output = tf.keras.layers.Concatenate(axis=1)(deep_input_list)
        for num in num_neurons:
            previous_output = tf.keras.layers.Dense(num, activation="relu", kernel_regularizer=embedding_regularizer)(previous_output)
        raw_rating = tf.keras.layers.Dense(1)(previous_output)

    else:
        # one_hot_input = tf.keras.layers.Concatenate(axis=1)(add_input_list)
        order_1 = tf.keras.layers.Add()(add_input_list)
        dot_list = []
        for i in range(len(embedding_list)):
            for j in range(i+1, len(embedding_list)):
                dot_list.append(tf.keras.layers.Dot(axes=1)([embedding_list[i], embedding_list[j]]))
        dot_list.append(order_1)

        if len(embedding_time_weight_list) == len(embedding_list):
            for embedding, time_weight in zip(embedding_list, embedding_time_weight_list):
                dot_list.append(tf.keras.layers.Dot(axes=1)([embedding, time_weight]))

        wide_rating = tf.keras.layers.Add()(dot_list)

        previous_output = tf.keras.layers.Concatenate(axis=1)(deep_input_list)
        for num in num_neurons:
            previous_output = tf.keras.layers.Dense(num, activation="relu", kernel_regularizer=embedding_regularizer)(previous_output)
        deep_output = tf.keras.layers.Dense(1)(previous_output)
        raw_rating = tf.keras.layers.Add()([wide_rating, deep_output])

    model = tf.keras.Model(inputs=input_list, outputs=[raw_rating])
    return model

def _cml_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job_dir',
        help='Directory where to save the given model',
        type=str,
        default='./models')
    parser.add_argument(
        '--batch_size',
        help='Batch size for sampling and training model',
        type=int,
        default=128)
    parser.add_argument(
        "--input_type",
        help="Inputs to be considered",
        choices=["user-movie", "user-movie-continuous_time", "user-movie-embedding_time", "user_t-movie_t"],
        default="user-movie-continuous_time"
    )
    parser.add_argument("--network", help="The network architecture", choices=["wide", "deep", "wide-deep"], default="wide-deep")
    parser.add_argument("--latent_dim", help="Latent space dimension", type=int, default=20)
    parser.add_argument('--yaml_file', help='Optimization space', type=str, default="../hyperparam.yaml")
    parser.add_argument("--train_dir", help="Training set directory", type=str, default="./train")
    parser.add_argument("--val_dir", help="Validation set directory", type=str, default="./val")
    parser.add_argument("--dict_path", help="File path of mapping dictionary of user id", type=str, default="./dict.pickle")
    parser.add_argument("--prefetch_size", help="Batch size prefetched", type=int, default=1)
    parser.add_argument("--num_neurons", help="The number of neurons in each layer of the deep part", type=int, nargs="*", default=[60, 40, 20])
    parser.add_argument("--regularized_factor", help="l1 regularization factor", type=float, default=0.0)
    parser.add_argument("--steps_per_epoch", help="The number of steps in each epoch", type=int, default=10000)
    parser.add_argument("--shuffle_size", help="buffer size for shuffling", type=int, default=1000000)
    parser.add_argument("--load_dir", help="Reload saved model directory", type=str, default=None)
    parser.add_argument("--epoches", help="Total epoches", type=int, default=50)
    result, _ = parser.parse_known_args()
    return result

if __name__ == "__main__":

    parser = _cml_parse()
    users_id_map = pickle.load(open(parser.dict_path, "rb"))
    keys = list(users_id_map.keys())
    values = list(users_id_map.values())
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(tf.constant(keys), tf.constant(values)), 480189)

    num_train_files = len([file for file in os.scandir(parser.train_dir) if file.is_file()])
    num_val_files = len([file for file in os.scandir(parser.val_dir) if file.is_file()])

    train_ds = _get_parallel_dataset(table,
                                     dataset_dir=parser.train_dir,
                                     input_type="user-movie-continuous_time",
                                     batch_size=parser.batch_size,
                                     prefetch=parser.prefetch_size,
                                     num_readers=1,
                                     num_threads=1,
                                     num_files=num_train_files,
                                     shuffle_size=parser.shuffle_size).take(parser.steps_per_epoch)
    val_ds = _get_parallel_dataset(table,
                                   dataset_dir=parser.val_dir,
                                   input_type="user-movie-continuous_time",
                                   batch_size=8192,
                                   prefetch=parser.prefetch_size,
                                   num_readers=num_val_files,
                                   num_threads=num_val_files,
                                   num_files=num_val_files,
                                   shuffle_size=10000)
    print("FINISH INITIALIZING DATASET.")

    if not os.path.exists(parser.job_dir):
        os.mkdir(parser.job_dir)

    _run(train_ds, val_ds, **vars(parser))
