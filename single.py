from clearml import Task, Logger
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import cifar_10
import os
import tempfile

task = Task.init(
    project_name='cifar-10',
    task_name='Single Experiment',
    task_type=Task.TaskTypes.training,
    reuse_last_task_id=False
)

args = {'batch_size': 128,
        'epochs': 1,
        'dropout_1': 0.25,
        'dropout_2': 0.25,
        'dropout_3': 0.5,
        'dense_1': 1024,
        }

args = task.connect(args)

(x_train, y_train) = cifar_10.get_training_data()
(x_test, y_test) = cifar_10.get_test_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = cifar_10.create_model(dropout_1=args["dropout_1"], 
                        dropout_2=args["dropout_2"], 
                        dropout_3=args["dropout_3"], 
                        dense_1=args["dense_1"])
                        
model = cifar_10.compile_model(model, 
                        optimizer='adam', 
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])


output_folder = os.path.join(tempfile.gettempdir(), 'keras_example')

board = TensorBoard(log_dir=output_folder, write_images=False)
model_store = ModelCheckpoint(filepath=os.path.join(output_folder, 'weight.hdf5'))

model.fit(x_train, y_train, 
        batch_size=args["batch_size"], 
        callbacks=[board, model_store],
        epochs=args["epochs"],
        validation_data=(x_test, y_test))

# score = cifar_10.evaluate_model(model, x_test, y_test)
score = model.evaluate(x_test, y_test)

print('Test score:', score[0])
print('Test accuracy:', score[1])

Logger.current_logger().report_scalar(title='evaluate', series='score', value=score[0], iteration=args['epochs'])
Logger.current_logger().report_scalar(title='evaluate', series='accuracy', value=score[1], iteration=args['epochs'])

model.save("./modlel/")
