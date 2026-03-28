# Lab: Managing Real-Time Endpoints

## The Scenario
Your company has successfully packaged its PyTorch model into a `model.tar.gz` artifact and stored it in S3. Now, you need to write the infrastructure code that deploys this artifact to a live, production HTTP Endpoint. Because this endpoint serves real customers, you must configure the deployment to use a secure Execution Role and properly scale out instances to handle web traffic. When you are done testing, you must tear down the endpoint so you don't receive a massive AWS bill at the end of the month!

## Core Tasks

1. Navigate to the `starter_code/` directory.
2. Open `e038-endpoint_lab.py`.
3. Complete the `deploy_production_model` function:
   - Configure the `MockPyTorchModel`. Pass `model_data` as the S3 URI (`s3://my-cloud-bucket/models/resnet-v1/model.tar.gz`), `role` as the `iam_role` variable, `entry_point` as `"inference.py"`, `framework_version` as `"2.0.0"`, and `py_version` as `"py310"`.
   - Call `.deploy()` on your configured `model`. You must request `initial_instance_count=2` and an `instance_type` of `"ml.m5.large"`. Name your endpoint `"production-endpoint-v1"`. Set `serializer="JSONSerializer()"` and `deserializer="JSONDeserializer()"`. Return the resulting `predictor` object.
4. Complete the `test_and_cleanup` function:
   - Call `.predict()` on the `predictor` object, passing the `dummy_payload`. Store the result in `response`.
   - Before exiting the script, call `.delete_endpoint()` on the `predictor` object to terminate the instances and stop billing!

## Definition of Done
- The script executes successfully locally.
- The output clearly demonstrates the Endpoint Boot sequence for `"production-endpoint-v1"`.
- The console prints the successful HTTP Response parsing.
- The console definitively prints that the endpoint was torn down and instances were terminated.
