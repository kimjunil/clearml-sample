from clearml import Task
from clearml.automation import UniformParameterRange, UniformIntegerParameterRange
from clearml.automation import HyperParameterOptimizer
from clearml.automation.optuna import OptimizerOptuna

task = Task.init(
    project_name='cifar-10',
    task_name='Hpyer Parameter Optimization',
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False
)

optimizer = HyperParameterOptimizer(
      # specifying the task to be optimized, task must be in system already so it can be cloned
      base_task_id="e50b86071b594e86aab35a8a00d586a3",  
      # setting the hyper-parameters to optimize
      hyper_parameters=[
          UniformIntegerParameterRange('epochs', min_value=2, max_value=12, step_size=2),
          UniformIntegerParameterRange('batch_size', min_value=128, max_value=256, step_size=8),
          UniformParameterRange('dropout_1', min_value=0, max_value=0.5, step_size=0.05),
          UniformParameterRange('dropout_2', min_value=0, max_value=0.5, step_size=0.05),
          UniformParameterRange('dropout_3', min_value=0, max_value=0.5, step_size=0.05),
          UniformIntegerParameterRange('dense_1', min_value=512, max_value=2048, step_size=8),
          ],
      # setting the objective metric we want to maximize/minimize
      objective_metric_title='accuracy',
      objective_metric_series='total',
      objective_metric_sign='max',  

      # setting optimizer  
      optimizer_class=OptimizerOptuna,
  
      # configuring optimization parameters
      execution_queue='k8s-queue2',  
      max_number_of_concurrent_tasks=5,  
      optimization_time_limit=60., 
      compute_time_limit=120, 
      total_max_jobs=20,  
      min_iteration_per_job=15000,  
      max_iteration_per_job=150000,  
      )


def job_complete_callback(
    job_id,                 # type: str
    objective_value,        # type: float
    objective_iteration,    # type: int
    job_parameters,         # type: dict
    top_performance_job_id  # type: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('WOOT WOOT we broke the record! Objective reached {}'.format(objective_value))

optimizer.start(job_complete_callback=job_complete_callback)

optimizer.set_time_limit(in_minutes=120.0)

optimizer.wait()

top_exp = optimizer.get_top_experiments(top_k=3)
print([t.id for t in top_exp])

optimizer.stop()