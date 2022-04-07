import tensorflow as tf
from clearml import Task

task = Task.init(
    project_name='cifar-10',
    task_name='Single Experiment',
    task_type=Task.TaskTypes.training,
    reuse_last_task_id=False
)

def train():
    
    (x_train, y_train) = get_training_data()
    (x_test, y_test) = get_test_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    model = create_model(dropout_1=0.25, 
                            dropout_2=0.25, 
                            dropout_3=0.5, 
                            dense_1=1024)
                            
    model = compile_model(model, 
                            optimizer='adam', 
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    model = fit_model(model, x_train, y_train)
    metric = evaluate_model(model, x_test, y_test)
    
    print(metric["val_loss"], metric["val_acc"])
    model.save("./modlel/")
    return model

def get_training_data():
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (_, _) = cifar10.load_data()
    return (x_train, y_train)

def get_test_data():
    cifar10 = tf.keras.datasets.cifar10
    (_, _), (x_test, y_test) = cifar10.load_data()
    return (x_test, y_test)

def create_model(dropout_1, dropout_2, dropout_3, dense_1):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(dropout_1))

    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(dropout_2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(dense_1, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_3))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    
    return model

def compile_model(model, optimizer, loss, metrics):
    model.compile(optimizer, loss, metrics)
    return model

def fit_model(model, x, y):
    model.fit(x, y, batch_size=128, shuffle=True, epochs=1)
    return model

def evaluate_model(model, x, y):
    val_loss, val_acc = model.evaluate(x, y)
    return {"val_loss":val_loss, "val_acc":val_acc}

if __name__ == "__main__":
    model = train()