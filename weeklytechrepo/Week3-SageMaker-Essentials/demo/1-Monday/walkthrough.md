# Lab Walkthrough: SageMaker JumpStart Fine-Tuning

This guide covers the end-to-end workflow for adaptive fine-tuning of a foundation model using the SageMaker Studio UI.

## Phase 1: Creating your Workspace (JupyterLab Space)

In the new SageMaker Studio experience, you work in **Spaces**.

1. **Access Studio:** Open the [SageMaker Console](https://console.aws.amazon.com/sagemaker/) and click **Studio** in the left sidebar.
2. **Open User:** Click **Open Studio** next to your User Profile.
3. **Create Space:** On the Home page, scroll to the **JupyterLab** section and click **Create JupyterLab Space**.
4. **Configure:**
   - **Name:** `Lab-Transfer-Learning`
   - **Instance:** Start with a `ml.t3.medium` (low cost) for coding. You can scale up to a GPU (`ml.g4dn`) later when you run the actual training.
5. **Run Space:** Once created, click **Run Space**, and then **Open JupyterLab**.

---

## Phase 2: Selecting a Pretrained Model (JumpStart)

Instead of writing training code from scratch, we use **SageMaker JumpStart**.

1. **Open JumpStart:** Inside your JupyterLab space, click the **Home** icon (or the SageMaker logo) and select **JumpStart**.
2. **Browse Models:** You will see categories like "Image Classification" or "Text Generation."
3. **Select Model:** Search for a popular model like **BERT Base Uncased** (for text) or **MobileNet** (for images).
4. **Model Hub:** Click on the model card. This page shows you the model details, licensing, and a "Deploy" vs. "Fine-Tune" option.

---

## Phase 3: Fine-Tuning the Model

Fine-tuning adapts the "brain" of the model to your specific data.

1. **Click Fine-Tune:** On the JumpStart model page, select the **Fine-Tune** tab.
2. **Prepare Data:** Upload your custom dataset to an **S3 Bucket**.
   > [!TIP]
   > You can find your default bucket by running `print(sagemaker.Session().default_bucket())` in a notebook.
3. **Set Hyperparameters:** Use the defaults for your first run (Epochs, Learning Rate).
4. **Launch Training:** Scroll to the bottom and click **Train**.

---

## Phase 4: Deploying to an Endpoint

Once your training job status is "Completed," it's time to host it.

1. **Go to Training Jobs:** In the SageMaker Console, under **Training**, find your completed job.
2. **Create Endpoint:** Click the **Deploy** button at the top of the job details page.
3. **Configure Endpoint:**
   - **Instance Type:** `ml.m5.xlarge` (CPU is usually enough for simple inference).
   - **Endpoint Name:** `my-finetuned-model-v1`.
4. **Scale to 1:** Set the initial instance count to 1.
5. **Deploy:** Click **Deploy**. This takes 3-5 minutes to "provision" the server.

---

## Phase 5: Testing with Predictions

Now we ask the model a question using the **SageMaker Python SDK**.

1. Open a new Jupyter Notebook (`.ipynb`) in your space.
2. Paste and run the following code:

```python
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer

# Initialize the predictor
predictor = Predictor(
    endpoint_name="my-finetuned-model-v1", 
    serializer=JSONSerializer()
)

# Send a test prediction
input_data = {"inputs": "The quick brown fox jumps over the lazy dog."}
response = predictor.predict(input_data)

print(f"Model Prediction: {response}")
```

> [!CAUTION]
> **Cleanup:** Always delete your endpoint when finished testing to stop hourly billing!
> `predictor.delete_endpoint()`
