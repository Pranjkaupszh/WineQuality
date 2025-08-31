##mlflow with deployment

(here)
##MLflow on Streamlit
Make sure you have Python 3.8+ installed. Then install the required packages:

pip install -r requirements.txt

🚀 Run the Streamlit App

In your terminal or command prompt:

streamlit run app.py

This will start a local Streamlit server and open the app in your browser at:

http://localhost::

📊 Use MLflow UI to Track Experiments

This project uses MLflow to log:
Parameters
Metrics
Model artifacts (saved models)

To view your MLflow experiments:
write this command:
mlflow ui

## MLflow on AWS Setup:

1. Login to AWS console.
2. Create IAM user with AdministratorAccess
3. Export the credentials in your AWS CLI by running "aws configure"
4. Create a s3 bucket
5. Create EC2 machine (Ubuntu) & add Security groups 5000 port

Run the following command on EC2 machine
```bash
sudo apt update

sudo apt install python3-pip

sudo apt install pipenv

sudo apt install virtualenv

mkdir mlflow

cd mlflow

pipenv install mlflow

pipenv install awscli

pipenv install boto3

pipenv shell


## Then set aws credentials
aws configure


#Finally 
mlflow server -h 0.0.0.0 --default-artifact-root s3://mlflowtracking1

#open Public IPv4 DNS to the port 5000


#set uri in your local terminal and in your code 

export MLFLOW_TRACKING_URI=http://ec2-54-158-152-207.compute-1.amazonaws.com:5000/
