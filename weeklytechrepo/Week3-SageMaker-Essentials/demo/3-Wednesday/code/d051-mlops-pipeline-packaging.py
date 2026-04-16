"""
Demo: SageMaker MLOps Pipeline Orchestrator (Live)
This script demonstrates how to programmatically define and register 
a Directed Acyclic Graph (DAG) for automated ML lifecycles.
It bridges together an SKLearn Data Processing Step, followed by a PyTorch Model Training Step.
"""

import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingOutput
from sagemaker.pytorch import PyTorch
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterInteger, ParameterFloat
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model

def demonstrate_pipelines_and_registry():
    print("--- SageMaker MLOps: Extended Pipeline Orchestrator ---")
    
    # =====================================================================
    # SECTION 1: SageMaker Initialization
    # =====================================================================
    # PipelineSession allows you to queue execution logic up without running it immediately
    pipeline_session = PipelineSession()
    
    # Grab the current IAM Role operating the Studio Space (so containers have S3 privileges)
    try:
        role = sagemaker.get_execution_role()
    except ValueError:
        role = "arn:aws:iam::407975137156:role/2371-SM-Execution-Test"
        
    # =====================================================================
    # SECTION 2: Pipeline Variable Bindings
    # =====================================================================
    # These parameters can be modified via the SageMaker GUI at runtime execution, 
    # meaning project managers can run new tests without rewriting any python scripts natively!
    epochs = ParameterInteger(
        name="TrainingEpochs",
        default_value=100
    )
    learning_rate = ParameterFloat(
        name="LearningRate",
        default_value=0.01
    )

    # =====================================================================
    # SECTION 3: Step A - Data Preprocessing Node
    # =====================================================================
    # 1. Define the computing instance required to process the data
    processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.t3.medium",  # Small instance suitable for basic data operations
        instance_count=1,
        base_job_name="demo-processing-job",
        role=role,
        sagemaker_session=pipeline_session
    )
    
    # 2. Wrap the configuration up into a DAG node entity (ProcessingStep)
    step_process = ProcessingStep(
        name="ProcessDataStep",
        processor=processor,
        inputs=[], # If we had raw data in S3, we'd list ProcessingInput mappings here
        
        # We capture the data dumped natively into /opt/ml/processing/train internally 
        # and store it back into S3 logically under the "train" label internally for pipeline passing.
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation")
        ],
        code="process.py" 
    )

    # =====================================================================
    # SECTION 4: Step B - Model Training Node
    # =====================================================================
    # 1. Provision a PyTorch Deep learning container mapping to train.py
    pt_estimator = PyTorch(
        entry_point="train.py",
        role=role,
        framework_version="2.1",
        py_version="py310",
        instance_count=1,
        instance_type="ml.m5.xlarge", # Upgrading power slightly for Training workload handling
        
        # Pass the global pipeline config parameters right into the py_script's argparse directly
        hyperparameters={
            "epochs": epochs,
            "learning_rate": learning_rate
        },
        sagemaker_session=pipeline_session
    )

    # 2. Wrap into a DAG node entity bridging the previous step outputs 
    step_train = TrainingStep(
        name="TrainPyTorchModel",
        estimator=pt_estimator,
        inputs={
            # Here is the magic linkage! We command the TrainingJob's "train" input channel
            # to utilize the physical S3 Object URI generated securely by the ProcessDataStep!
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv"
            )
        }
    )

    # =====================================================================
    # SECTION 5: Step C - Artifact Versioning & Registry
    # =====================================================================
    # Create the high level AWS Model object referring to the zipped state dictionaries from training
    model = Model(
        image_uri=pt_estimator.training_image_uri(),
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=pipeline_session,
        role=role
    )

    # Log this iteration into a governed SageMaker "Model Registry Group", applying metadata.
    # Marking it as PendingManualApproval ensures a human engineer guarantees quality before deployment
    step_register = ModelStep(
        name="RegisterPyTorchModel",
        step_args=model.register(
            content_types=["text/csv"],
            response_types=["text/csv"],
            inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
            transform_instances=["ml.m5.xlarge"],
            model_package_group_name="PyTorchLinearRegressionDemoGroup",
            approval_status="PendingManualApproval"
        )
    )

    # =====================================================================
    # SECTION 6: DAG Assembly & Push Execution
    # =====================================================================
    print("\nAssembling Pipeline DAG...")
    
    # We compile the graph with all variables and step node dependencies logically mapped.
    pipeline = Pipeline(
        name="PyTorchLinearRegressionPipeline",
        parameters=[epochs, learning_rate],
        steps=[step_process, step_train, step_register],
        sagemaker_session=pipeline_session
    )

    # Upsert validates the structural integrity of your pipeline JSON model and ships 
    # it back to AWS, showing it off cleanly on the Amazon SageMaker Studio Dashboards.
    print("\n[Action] Registering Pipeline with AWS SageMaker...")
    try:
        pipeline.upsert(role_arn=role)
        print("✅ Pipeline registered successfully!")
        print(f"Name: {pipeline.name}")
        print("Note: The steps are NOT running yet. You must initialize an execution session in the dashboard!")
    except Exception as e:
        print(f"❌ Registration skipped/failed: {e}")

    print("-" * 50)

if __name__ == "__main__":
    demonstrate_pipelines_and_registry()
