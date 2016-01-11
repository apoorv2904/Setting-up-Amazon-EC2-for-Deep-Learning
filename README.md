# Setting-up-Amazon-EC2-for-Deep-Learning
Instructions to install cuda, theano, nolearn, sklearn, skimage, lasagne, cudamat for Deep Learning

Setting up Amazon EC2 for deep learning can be a very painful task and can take a while to just setup Theano and other libraries to start with the actual interesting work.<br />
The following steps can help reduce the time and efforts required to get started on Amazon EC2.<br /><br />

## Some Background
1. Go to Amazon.com and signup for Amazon Web Services.<br />
https://aws.amazon.com/console/<br />

2. Amazon EC2 is the cloud computing resource that we will be primarily using to run the code on the cloud. Amazon offers 2 type of instances :- <br />
  1. Reserved Instance - This as the name says will be allocated to you at a fixed price/hour. A Gpu g2.2x large instance which we will use typically costs $0.65 an hour.
  2. Spot Instance - These are machines which are available for bidding. The advantage is that this costs around $0.07- $0.1 an hour depending upon the region ( will tell about the regions later ) but the machine can be taken away at a 2 min notice when someone outbids you so the work has to be saved periodically to ensure no loss. <br />
We will be using these Spot instances as a part of the guidelines and will look at how to save the files on a Amazon S3 Machine.<br />

3. AMI - Suppose you install a bunch of software on an EC2 instance and use it for a couple of hours and then terminate the instance. It is painful to install these software every time you run the system. So Amazon offers you to save the image of the machine which can boot up with the software installed but this will cost some additional money.<br />
Alternatively you can select from community AMI to select a Machine with pre-installed software to save the money and hassle.

4. Regions - Amazon offers its services in different regions and data within a region is duplicated to ensure robustness. AMI discussed above are restricted to a region and an AMI in one region might not be available in another.

## Setting up Instructions.
### Connecting to Amazon EC2 instance.
1. Log in to Amazon Console.
2. Select a region from top right. For this tutorial we will use N.California.
3. Go to Services -> EC2 -> Spot Requests ( Under Instances. You can have a look at the pricing history to get a guide on the bid )
4. Choose an AMI. To show a complete setup, I will go with AMI Ubuntu Server 14.04 LTS (HVM), SSD Volume Type. ( I will recommend you to try out complete installation once. Later you can go with the AMI I have mentioned.)
5. From the available options go GPU g2.2xlarge instance. (This will cost you money. Small amount :) )
6. Press Next and Enter your bidding price based on the history suggested. At the time of writing this current price is around $0.07/hour so I am going to bid $0.1/hour to have a safety margin.
7. Press Review and Launch. Press Launch. You will come to a screen where it will ask for a Key-Pair. If you are using it for the first time, then please create a new key-pair and make sure you keep it safely. If lost it cannot be retrieved later on.
8. If on Linux use this command to connect to EC2 instance<br />
  ```
  **ssh -i EC2KeyPair.pem ubuntu@[your instance ip address]**
  ```
  If on windows use putty to connect to the instance. The instructions can be found here.<br />
  http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/putty.html<br />
  In nutshell, do the following.<br />
  1. Open PuTTYgen
  2. Click load and load <key>.pem 
  3. Press save private key
  4. After saving private key in ppk format open PuTTy
  5. In host enter ubuntu@DNSname. This can be seen by cliking the instance in AWS services and copying Public DNS column
  6. Now SSH -> Auth Tab and browse private key. and press open. You will be connected to the Amazon EC2 Server

### Installing Theano and other libraries.
1. Run these commands to install basic libraries<br />
  ```
  **sudo apt-get update**<br />
  **sudo apt-get -y dist-upgrade**<br />
  **sudo apt-get install -y gcc g++ gfortran build-essential git wget linux-image-generic libopenblas-dev python-dev python-pip python-nose python-numpy python-scipy**<br />
  **sudo apt-get install -y liblapack-dev**<br />
  **sudo apt-get install -y libblas-dev**<br />
  ```
2. Install Cuda by running these commands.**THIS IS THE MOST TIME CONSUMING STEP- USE COMMUNITY AMI TO AVOID THIS (Steps at last - USING PREINSTALLED AMI)** Install latest cuda- cuda-repo-ubuntu1404_7.5-18_amd64.deb ( This is cuda 7.5)<br />
  ```
  **wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb**<br />
  **sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb**<br />
  **sudo apt-get update**<br />
  **sudo apt-get install cuda**<br />
  ```
3. Reboot and reconnect<br />
  **sudo reboot**<br />
  
4. Add paths<br />
  **CUDA_ROOT=`find /usr/local/ -type d -name "cuda-[0-9]\.[0-9]" -print`**<br />
  The above command will find the cuda path and for cuda-7.5 it should be */usr/local/cuda-7.5<br />
  Run following commands to add paths to the library. Many issues that usually come are due to wrong path settings. Make sure you import path in the environment. <br />
  
  **export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH} **<br />
  **export PATH=${CUDA_ROOT}/bin:${PATH}**<br />
  **echo "PATH=${CUDA_ROOT}/bin:${PATH}" >> .bashrc**<br />
  **echo "export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH}" >> .bashrc**<br />
  **source ~/.bashrc**<br />

5. Verify Cuda is working. In the commands set right version for cuda. Here it was 7.5<br />
  
  **cuda-install-samples-7.5.sh  ~**<br />
  **cd ~/NVIDIA_CUDA-7.5_Samples**<br />
  **cd 1_Utilities/deviceQuery**<br />
  **make**<br />
  **./deviceQuery**<br />

6. Go to home folder and install sklearn and nolearn<br />
  **cd ~**<br />
  **sudo pip install scikit-learn nolearn**<br />
  
7. Install cudamat<br />
  **git clone https://github.com/cudamat/cudamat**<br />
  **cd cudamat**<br />
  **pip install --user .**<br />
  
8. Install theano<br />
  **sudo pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt**<br />

9. Configure Theano - Right now if you run python and import theano it will not use GPU. In order to use GPU we need to configure theanorc<br />
  Create file ~/.theanorc and copy<br />
  
  **[global]**<br />
  **floatX = float32**<br />
  **device = gpu0**<br />
  <br />
  **[nvcc]**<br />
  **fastmath = True**<br />

10. Verify if GPU is used. Open Python and import theano, if GPU is being used it will print the GPU information.<br />


### Install Lasagne and skimage
1. Intall Lasagne <br />
  **sudo pip install Lasagne==0.1**<br />
  **sudo apt-get update**<br />
  **sudo pip install scikit-learn**<br />
  **sudo apt-get install python-matplotlib**<br />
  **sudo pip install scikit-image**<br />

2. Install Cudadnn after making an account at nvidia<br />
  https://developer.nvidia.com/cudnn<br />
  1. Register and click download
  2. Right cuDNN v4 Library for Linux click and click copy link address
  3. Now there are 2 options either download file to your local machine and then upload to S3 and download from S3 and run next commands or to download directly to the ec2 machine follow the steps in Download dataset from kaggle (next section long term helpful ) and use the cookies.txt generated in the cudnn download link page
  4. copy cookies.txt (on nvdia link page ) to the ec2 machine and run<br />
    
    **mkdir data**<br />
    **wget -x --load-cookies cookies.txt -P data -nH --cut-dirs=5 https://developer.nvidia.com/rdp/assets/cudnn-70-linux-x64-v40**<br />
    **cd data**<br />
    **mv cudnn-70-linux-x64-v40 cudnn-70-linux-x64-v40.tar.gz**<br />
    **tar -xvzf cudnn-70-linux-x64-v40.tar.gz**<br />
    **CUDA_ROOT=`find /usr/local/ -type d -name "cuda-[0-9]\.[0-9]" -print`**<br />
    **cd cuda/include/**<br />
    **sudo cp `ls *.h` "$CUDA_ROOT"/include/**<br />
    **cd ../lib64/**<br />
    **sudo cp `ls *`  "$CUDA_ROOT"/lib64/**<br />
    **export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH}**<br />
    **export PATH=${CUDA_ROOT}/bin:${PATH}**<br />
  
  5. For faster use in future upload cudnn-70-linux-x64-v40.tar.gz to S3 bucket and retrieve from there to use.
  6. At times when lasagne gives some error, one of the three commands solves the purpose<br />
    **sudo pip install -r https://raw.githubusercontent.com/dnouri/kfkd-tutorial/master/requirements.txt**<br />
    **sudo pip install -r https://raw.githubusercontent.com/dnouri/nolearn/master/requirements.txt**<br />
    **sudo pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt**<br />

### Download Dataset from Kaggle<br />
Steps :-<br />
1. Export your cookies from your browser, when you are logged in at kaggle and put your cookies.txt on your server. Then run:
  mkdir data<br />
  **wget -x --load-cookies cookies.txt -P data -nH --cut-dirs=5 http://www.kaggle.com/c/dogs-vs-cats/download/test1.zip**<br />
  
## S3 Setup for persistent storage<br />
Use S3 to temporarily save your models when using Spot Instances<br />

###Install awscli to move files to and from S3<br />

1. Create S3 bucket<br />
  https://www.youtube.com/watch?v=wODEO2Tvmik<br />
2. Set up Access Key<br />
  Read How to Retrieve Root Access Keys (first one)<br />
  http://www.cloudberrylab.com/blog/how-to-find-your-aws-access-key-id-and-secret-access-key-and-register-with-cloudberry-s3-explorer/ <br />
3. On on your ec2 instance, Install awscli and configure it<br />
  **sudo apt-get install awscli**<br />
  **aws configure**<br />
  
  Configure will ask 4 questions<br />
  AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE<br />
  AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY<br />
  Default region name [None]: us-west-2<br />
  Default output format [None]: ENTER<br />
  
  In third question you can see the region by having a look at region properties. For N.California bucket it is us-west-1<br />
  You can connect to any bucket in any region.<br />

4. Transfer file/folder to S3<br />
  ####Make test folder<br />
  **mkdir TestTransfer**<br />
  **aws s3 cp TestTransfer s3://apoorvfirstbucket/DeepLearning/SpatialTransformerNetwork/CatDog/TestTransfer --recursive**<br />
  
  ####Create a test file<br />
  **vim test.txt**<br />
  **aws s3 cp test.txt s3://apoorvfirstbucket/DeepLearning/SpatialTransformerNetwork/CatDog/test.txt**<br />

5. Retrieve from file/folder from S3<br />
  **aws s3 cp s3://apoorvfirstbucket/DeepLearning/SpatialTransformerNetwork/CatDog/TestTransfer TestTransfer --recursive**<br />
  **aws s3 cp  s3://apoorvfirstbucket/DeepLearning/SpatialTransformerNetwork/CatDog/test.txt test.txt**<br />
  
## IMPORTANT - USING Preinstalled AMI TO AVOID THEANO INSTALLATION TIME & EFFORTS<br />
1. When selecting an AMI, go to Commnity AMI and search for theano or deep learning.<br />
2. Select one of the AMI's and proceed.<br />
3. I usually select the AMI with theano and cuda 7 installed and install Lasagne myself.<br />

## OPTIONAL DEEP LEARNING TUTORIAL
A. Install Deep Learning Tutorial to playaround.<br />
  **cd ~**<br />
  **git clone https://github.com/lisa-lab/DeepLearningTutorials.git**<br />

B. Install datasets and play around<br />
  **cd DeepLearningTutorials/data/**<br />
  **chmod +x download.sh**<br />
  **./download.sh**<br />

C. Run convolutional_mlp.py<br />
  **cd ../code/**<br />
  **time python convolutional_mlp.py**<br />
  
  It takes around 39-40 minutes to run on GPU instance <br />

  
