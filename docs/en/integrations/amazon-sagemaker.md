---
title: YOLO26 on Amazon SageMaker Endpoints
comments: true
description: Deploy Ultralytics YOLO26 on Amazon SageMaker Endpoints with AWS CDK and CloudFormation, from environment setup to endpoint testing and monitoring.
keywords: YOLO26, Amazon SageMaker, AWS, Ultralytics, machine learning, computer vision, model deployment, AWS CloudFormation, AWS CDK, real-time inference
---

# A Guide to Deploying YOLO26 on Amazon SageMaker Endpoints

Amazon SageMaker Endpoints provide managed, scalable hosting for [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) models like [Ultralytics YOLO26](../models/yolo26.md) in production [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) applications. A successful [model deployment](https://www.ultralytics.com/glossary/model-deployment) depends on understanding the setup, configuration, and deployment steps on both the model and the AWS side.

This guide will take you through the process of deploying YOLO26 [PyTorch](https://www.ultralytics.com/glossary/pytorch) models on Amazon SageMaker Endpoints step by step. You'll learn the essentials of preparing your AWS environment, configuring the model appropriately, and using tools like AWS CloudFormation and the AWS Cloud Development Kit (CDK) for deployment.

## Amazon SageMaker

<p align="center">
  <img width="640" src="https://cdn.ul.run/i/ebafd2d72a07d5a323ae495d5f1fc623.avif" alt="Amazon SageMaker ML platform architecture">
</p>

[Amazon SageMaker](https://aws.amazon.com/sagemaker/) is a machine learning service from Amazon Web Services (AWS) that simplifies the process of building, training, and deploying machine learning models. It provides a broad range of tools for handling various aspects of machine learning workflows. This includes automated features for tuning models, options for training models at scale, and straightforward methods for deploying models into production. SageMaker supports popular machine learning frameworks, offering the flexibility needed for diverse projects. Its features also cover data labeling, workflow management, and performance analysis.

## Deploying YOLO26 on Amazon SageMaker Endpoints

Deploying YOLO26 on Amazon SageMaker lets you use its managed environment for [real-time inference](https://www.ultralytics.com/glossary/real-time-inference) and take advantage of features like autoscaling. Take a look at the AWS architecture below.

<p align="center">
  <img width="640" src="https://cdn.ul.run/i/8d98b503db95f4426bfd35e313166a1b.avif" alt="AWS architecture for hosting YOLO on an Amazon SageMaker endpoint">
</p>

### Step 1: Set Up Your AWS Environment

First, ensure you have the following prerequisites in place:

- An AWS Account: If you don't already have one, sign up for an AWS account.

- Configured IAM Roles: You'll need an IAM role with the necessary permissions for Amazon SageMaker, AWS CloudFormation, and Amazon S3. This role should have policies that allow it to access these services.

- AWS CLI: If not already installed, download and install the AWS Command Line Interface (CLI) and configure it with your account details. Follow [the AWS CLI instructions](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) for installation.

- AWS CDK: If not already installed, install the AWS Cloud Development Kit (CDK), which will be used for scripting the deployment. Follow [the AWS CDK instructions](https://docs.aws.amazon.com/cdk/v2/guide/#getting_started_install) for installation.

- Adequate Service Quota: Confirm that you have sufficient quotas for two separate resources in Amazon SageMaker: one for `ml.m5.4xlarge` for endpoint usage and another for `ml.m5.4xlarge` for notebook instance usage. Each of these requires a minimum of one quota value. If your current quotas are below this requirement, it's important to request an increase for each. You can request a quota increase by following the detailed instructions in the [AWS Service Quotas documentation](https://docs.aws.amazon.com/servicequotas/latest/userguide/request-quota-increase.html#quota-console-increase).

### Step 2: Clone the YOLO26 SageMaker Repository

The next step is to clone the specific AWS repository that contains the resources for deploying YOLO26 on SageMaker. This repository, hosted on GitHub, includes the necessary CDK scripts and configuration files.

- Clone the GitHub Repository: Execute the following command in your terminal to clone the host-yolov8-on-sagemaker-endpoint repository:

    ```bash
    git clone https://github.com/aws-samples/host-yolov8-on-sagemaker-endpoint.git
    ```

- Navigate to the Cloned Directory: Change your directory to the cloned repository:

    ```bash
    cd host-yolov8-on-sagemaker-endpoint/yolov8-pytorch-cdk
    ```

### Step 3: Set Up the CDK Environment

Now that you have the necessary code, set up your environment for deploying with AWS CDK.

- Create a Python Virtual Environment: This isolates your Python environment and dependencies. Run:

    ```bash
    python3 -m venv .venv
    ```

- Activate the Virtual Environment:

    ```bash
    source .venv/bin/activate
    ```

- Install Dependencies: Install the required Python dependencies for the project:

    ```bash
    pip3 install -r requirements.txt
    ```

- Upgrade AWS CDK Library: Ensure you have the latest version of the AWS CDK library:

    ```bash
    pip install --upgrade aws-cdk-lib
    ```

### Step 4: Create the AWS CloudFormation Stack

- Synthesize the CDK Application: Generate the AWS CloudFormation template from your CDK code:

    ```bash
    cdk synth
    ```

- Bootstrap the CDK Application: Prepare your AWS environment for CDK deployment:

    ```bash
    cdk bootstrap
    ```

- Deploy the Stack: This will create the necessary AWS resources and deploy your model:

    ```bash
    cdk deploy
    ```

### Step 5: Deploy the YOLO Model

Before deploying, review the available [YOLO26 models](https://platform.ultralytics.com/ultralytics/yolo26) to choose the size and task that fit your accuracy and latency requirements.

After creating the AWS CloudFormation Stack, the next step is to deploy YOLO26.

- Open the Notebook Instance: Go to the AWS Console and navigate to the Amazon SageMaker service. Select "Notebook Instances" from the dashboard, then locate the notebook instance that was created by your CDK deployment script. Open the notebook instance to access the Jupyter environment.

- Access and Modify inference.py: After opening the SageMaker notebook instance in Jupyter, locate the inference.py file. Edit the output_fn function in inference.py as shown below and save your changes to the script, ensuring that there are no syntax errors.

    ```python
    import json


    def output_fn(prediction_output, content_type):
        """Formats model outputs as JSON string, extracting attributes like boxes, masks, keypoints."""
        print("Executing output_fn from inference.py ...")
        infer = {}
        for result in prediction_output:
            if result.boxes is not None:
                infer["boxes"] = result.boxes.numpy().data.tolist()
            if result.masks is not None:
                infer["masks"] = result.masks.numpy().data.tolist()
            if result.keypoints is not None:
                infer["keypoints"] = result.keypoints.numpy().data.tolist()
            if result.obb is not None:
                infer["obb"] = result.obb.numpy().data.tolist()
            if result.probs is not None:
                infer["probs"] = result.probs.numpy().data.tolist()
        return json.dumps(infer)
    ```

- Deploy the Endpoint Using 1_DeployEndpoint.ipynb: In the Jupyter environment, open the 1_DeployEndpoint.ipynb notebook located in the sm-notebook directory. Follow the instructions in the notebook and run the cells to download the YOLO26 model, package it with the updated inference code, and upload it to an Amazon S3 bucket. The notebook will guide you through creating and deploying a SageMaker endpoint for the YOLO26 model.

### Step 6: Testing Your Deployment

Now that your YOLO26 model is deployed, it's important to test its performance and functionality.

- Open the Test Notebook: In the same Jupyter environment, locate and open the 2_TestEndpoint.ipynb notebook, also in the sm-notebook directory.

- Run the Test Notebook: Follow the instructions within the notebook to test the deployed SageMaker endpoint. This includes sending an image to the endpoint and running inferences. Then, you'll plot the output to visualize the model's performance and [accuracy](https://www.ultralytics.com/glossary/accuracy), as shown below.

<p align="center">
  <img width="640" src="https://cdn.ul.run/i/373b4db68fb4b27bd989f229905114c2.avif" alt="YOLO26 detection results returned by the SageMaker test endpoint">
</p>

- Clean-Up Resources: The test notebook will also guide you through the process of cleaning up the endpoint and the hosted model. This is an important step to manage costs and resources effectively, especially if you do not plan to use the deployed model immediately.

### Step 7: Monitoring and Management

After testing, continuous monitoring and management of your deployed model are essential.

- Monitor with Amazon CloudWatch: Regularly check the performance and health of your SageMaker endpoint using [Amazon CloudWatch](https://aws.amazon.com/cloudwatch/).

- Manage the Endpoint: Use the SageMaker console for ongoing management of the endpoint. This includes scaling, updating, or redeploying the model as required.

By completing these steps, you will have deployed and tested a YOLO26 model on Amazon SageMaker Endpoints. The same workflow applies to deploying other models on AWS.

## Summary

This guide took you step by step through deploying YOLO26 on Amazon SageMaker Endpoints using AWS CloudFormation and the AWS Cloud Development Kit (CDK). The process includes cloning the necessary GitHub repository, setting up the CDK environment, deploying the model using AWS services, and testing its performance on SageMaker.

For more technical details, read [Hosting YOLOv8 PyTorch models on Amazon SageMaker Endpoints](https://aws.amazon.com/blogs/machine-learning/hosting-yolov8-pytorch-model-on-amazon-sagemaker-endpoints/) on the AWS Machine Learning Blog. The official [Amazon SageMaker real-time endpoints documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html) covers endpoint features and configuration in depth.

To explore other YOLO26 integrations, visit the [Ultralytics integrations guide](../integrations/index.md).

## FAQ

### How do I deploy the Ultralytics YOLO26 model on Amazon SageMaker Endpoints?

To deploy the Ultralytics YOLO26 model on Amazon SageMaker Endpoints, follow these steps:

1. **Set Up Your AWS Environment**: Ensure you have an AWS Account, IAM roles with necessary permissions, and the AWS CLI configured. Install AWS CDK if not already done (refer to the [AWS CDK instructions](https://docs.aws.amazon.com/cdk/v2/guide/#getting_started_install)).
2. **Clone the YOLO26 SageMaker Repository**:
    ```bash
    git clone https://github.com/aws-samples/host-yolov8-on-sagemaker-endpoint.git
    cd host-yolov8-on-sagemaker-endpoint/yolov8-pytorch-cdk
    ```
3. **Set Up the CDK Environment**: Create a Python virtual environment, activate it, install dependencies, and upgrade AWS CDK library.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install -r requirements.txt
    pip install --upgrade aws-cdk-lib
    ```
4. **Deploy using AWS CDK**: Synthesize and deploy the CloudFormation stack, bootstrap the environment.
    ```bash
    cdk synth
    cdk bootstrap
    cdk deploy
    ```

For further details, see [Step 5: Deploy the YOLO Model](#step-5-deploy-the-yolo-model).

### What are the prerequisites for deploying YOLO26 on Amazon SageMaker?

To deploy YOLO26 on Amazon SageMaker, ensure you have the following prerequisites:

1. **AWS Account**: Active AWS account ([sign up for AWS](https://aws.amazon.com/)).
2. **IAM Roles**: Configured IAM roles with permissions for SageMaker, CloudFormation, and Amazon S3.
3. **AWS CLI**: Installed and configured AWS Command Line Interface ([AWS CLI installation guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)).
4. **AWS CDK**: Installed AWS Cloud Development Kit ([CDK setup guide](https://docs.aws.amazon.com/cdk/v2/guide/#getting_started_install)).
5. **Service Quotas**: Sufficient quotas for `ml.m5.4xlarge` instances for both endpoint and notebook usage ([request a quota increase](https://docs.aws.amazon.com/servicequotas/latest/userguide/request-quota-increase.html#quota-console-increase)).

For detailed setup, see [Step 1: Set Up Your AWS Environment](#step-1-set-up-your-aws-environment).

### Why should I use Ultralytics YOLO26 on Amazon SageMaker?

Using Ultralytics YOLO26 on Amazon SageMaker offers several advantages:

1. **Scalability and Management**: SageMaker provides a managed environment with features like autoscaling, which helps in real-time inference needs.
2. **Integration with AWS Services**: Integrate with other AWS services, such as S3 for data storage, CloudFormation for infrastructure as code, and CloudWatch for monitoring.
3. **Ease of Deployment**: Simplified setup using AWS CDK scripts and streamlined deployment processes.
4. **Performance**: Use Amazon SageMaker's managed infrastructure to run large-scale inference workloads.

Learn more about SageMaker in the [Amazon SageMaker overview](#amazon-sagemaker).

### Can I customize the inference logic for YOLO26 on Amazon SageMaker?

Yes, you can customize the inference logic for YOLO26 on Amazon SageMaker:

1. **Modify `inference.py`**: Locate and customize the `output_fn` function in the `inference.py` file to tailor output formats.

    ```python
    import json


    def output_fn(prediction_output, content_type):
        """Formats model outputs as JSON string, extracting attributes like boxes, masks, keypoints."""
        infer = {}
        for result in prediction_output:
            if result.boxes is not None:
                infer["boxes"] = result.boxes.numpy().data.tolist()
            # Add more processing logic if necessary
        return json.dumps(infer)
    ```

2. **Deploy Updated Model**: Ensure you redeploy the model using Jupyter notebooks provided (`1_DeployEndpoint.ipynb`) to include these changes.

Refer to the [detailed steps](#step-5-deploy-the-yolo-model) for deploying the modified model.

### How can I test the deployed YOLO26 model on Amazon SageMaker?

To test the deployed YOLO26 model on Amazon SageMaker:

1. **Open the Test Notebook**: Locate the `2_TestEndpoint.ipynb` notebook in the SageMaker Jupyter environment.
2. **Run the Notebook**: Follow the notebook's instructions to send an image to the endpoint, perform inference, and display results.
3. **Visualize Results**: Use the notebook's plotting code to draw the predicted bounding boxes on the test image.

For comprehensive testing instructions, visit the [testing section](#step-6-testing-your-deployment).
