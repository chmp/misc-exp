## `chmp.experiment`

Extensions around mlflow to track experiments.


### `chmp.experiment.active_run`
`chmp.experiment.active_run()`

Get the currently active `Run`, or None if no such run exists.


### `chmp.experiment.log_artifact`
`chmp.experiment.log_artifact(local_path, artifact_path=None)`

Log a local file or directory as an artifact of the currently active run.

#### Parameters

* **local_path** (*any*):
  Path to the file to write.
* **artifact_path** (*any*):
  If provided, the directory in `artifact_uri` to write to.


### `chmp.experiment.log_artifacts`
`chmp.experiment.log_artifacts(local_dir, artifact_path=None)`

Log all the contents of a local directory as artifacts of the run.

#### Parameters

* **local_dir** (*any*):
  Path to the directory of files to write.
* **artifact_path** (*any*):
  If provided, the directory in `artifact_uri` to write to.


### `chmp.experiment.log_param`
`chmp.experiment.log_param(key, value)`

Log a parameter under the current run, creating a run if necessary.

#### Parameters

* **key** (*any*):
  Parameter name (string)
* **value** (*any*):
  Parameter value (string, but will be string-ified if not)


### `chmp.experiment.log_metric`
`chmp.experiment.log_metric(key, value)`

Log a metric under the current run, creating a run if necessary.

#### Parameters

* **key** (*any*):
  Metric name (string).
* **value** (*any*):
  Metric value (float).


### `chmp.experiment.set_tag`
`chmp.experiment.set_tag(key, value)`

Set a tag under the current run, creating a run if necessary.

#### Parameters

* **key** (*any*):
  Tag name (string)
* **value** (*any*):
  Tag value (string, but will be string-ified if not)


### `chmp.experiment.delete_experiment`
`chmp.experiment.delete_experiment(run_uuid, store=None)`


### `chmp.experiment.get_all_infos`
`chmp.experiment.get_all_infos(store=None)`


### `chmp.experiment.get_all_infos_df`
`chmp.experiment.get_all_infos_df()`


### `chmp.experiment.get_all_metrics`
`chmp.experiment.get_all_metrics(run_uuid, store=None)`


### `chmp.experiment.get_all_params`
`chmp.experiment.get_all_params(run_uuid, store=None)`


### `chmp.experiment.get_artifact_repository`
`chmp.experiment.get_artifact_repository(run_uuid, store=None)`


### `chmp.experiment.get_infos`
`chmp.experiment.get_infos(run_uuid, store=None)`


### `chmp.experiment.get_metric`
`chmp.experiment.get_metric(run_uuid, metric_key, store=None)`


### `chmp.experiment.get_metric_history`
`chmp.experiment.get_metric_history(run_uuid, metric_key, store=None)`


### `chmp.experiment.get_param`
`chmp.experiment.get_param(run_uuid, param_key, store=None)`


### `chmp.experiment.get_store`
`chmp.experiment.get_store(uri=None)`


### `chmp.experiment.get_structured`
`chmp.experiment.get_structured(run_uuid, store=None)`


### `chmp.experiment.get_tempdir`
`chmp.experiment.get_tempdir()`

Get a tempdir, after the run it will be logged as an artifact.


### `chmp.experiment.get_trash_folder`
`chmp.experiment.get_trash_folder(store=None)`


### `chmp.experiment.get_run`
`chmp.experiment.get_run(run_uuid, store=None)`


### `chmp.experiment.has_active_run`
`chmp.experiment.has_active_run()`


### `chmp.experiment.is_local_artifact_respository`
`chmp.experiment.is_local_artifact_respository(artifact_repo)`


### `chmp.experiment.list_artifacts`
`chmp.experiment.list_artifacts(run_uuid, path='.', store=None)`


### `chmp.experiment.list_all_runs`
`chmp.experiment.list_all_runs(store=None)`

Get the ids of all runs


### `chmp.experiment.log_all_metrics`
`chmp.experiment.log_all_metrics(**metrics)`


### `chmp.experiment.log_all_params`
`chmp.experiment.log_all_params(**params)`


### `chmp.experiment.log_structured`
`chmp.experiment.log_structured(**values)`

Log a stream of structured values to an artifact inside mlflow.


### `chmp.experiment.log_tempdir`
`chmp.experiment.log_tempdir()`


### `chmp.experiment.new_run`
`chmp.experiment.new_run(*args, **kwds)`


### `chmp.experiment.open_artifact`
`chmp.experiment.open_artifact(*args, **kwds)`


### `chmp.experiment.restore_experiment`
`chmp.experiment.restore_experiment(run_uuid, store=None)`


### `chmp.experiment.set_all_tags`
`chmp.experiment.set_all_tags(**tags)`

