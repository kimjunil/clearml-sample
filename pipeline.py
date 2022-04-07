from clearml.automation.controller import PipelineDecorator
from clearml import TaskTypes

@PipelineDecorator.component(return_values=['X_train', 'X_test', 'y_train', 'y_test'], cache=True, task_type=TaskTypes.data_processing)
def step_one():
    import tensorflow as tf

    cifar10 = tf.keras.datasets.cifar10
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    return X_train, X_test, y_train, y_test

@PipelineDecorator.component(
  return_values=['X_train'], cache=True, task_type=TaskTypes.data_processing)
def step_two_1(X_train):
    import tensorflow as tf
    
    return tf.keras.utils.normalize(X_train, axis=1)

@PipelineDecorator.component(
  return_values=['X_test'], cache=True, task_type=TaskTypes.data_processing)
def step_two_2(X_test):
    import tensorflow as tf
    
    return tf.keras.utils.normalize(X_test, axis=1)

@PipelineDecorator.component(return_values=['model'], cache=True, task_type=TaskTypes.training)
def step_three(X_train, y_train):
    print('step_three')
    # make sure we have pandas for this step, we need it to use the data_frame
    import cifar_10

    model = cifar_10.create_model()
    model = cifar_10.fit_model(model, X_train, y_train)
    
    return model

@PipelineDecorator.component(return_values=['accuracy'], cache=True, task_type=TaskTypes.qc)
def step_four(model, X_data, Y_data):

    val_loss, val_acc = model.evaluate(X_data, Y_data)

    return val_acc

@PipelineDecorator.pipeline(name='pipeline', project='examples', version='0.1')
def main():
    X_train, X_test, y_train, y_test  = step_one()
    X_train = step_two_1(X_train)
    X_test = step_two_2(X_test)
    model = step_three(X_train, y_train)
    accuracy = 100 * step_four(model, X_data=X_test, Y_data=y_test)
    print(f"Accuracy={accuracy}%")

if __name__ == '__main__':
    # run the pipeline on the current machine, for local debugging
    # for scale-out, comment-out the following line and spin clearml agents
    PipelineDecorator.run_locally()

    main()