# Setting-up-Amazon-EC2-for-Deep-Learning
Instructions to install cuda, theano, nolearn, sklearn, skimage, lasagne, cudamat for Deep Learning

Setting up Amazon EC2 for deep learning can be a very painful task and can take a while to just setup Theano and other libraries to start with the actual interesting work.
The following steps can help reduce the time and efforts required to get started on Amazon EC2.

## Some Backgound
1. Go to Amazon.com and signup for Amazon Web Services.
https://aws.amazon.com/console/

2. Amazon EC2 is the cloud computing resource that we will be primarily using to run the code on the cloud. Amazon offers 2 type of instances :- 
  1. Reserved Instance - This as the name says will be allocated to you at a fixed price/hour. A Gpu g2.2x large instance which we will use typicall costs $0.65 an hour.
  2. Spot Instance - These are machines which are available for bidding. The advantage is that this costs around $0.07- $0.1 an hour depending upon the region ( will tell about the regions later ) but the machine can be taken away at a 2 min notice when someone outbids you so the work has to be saved periodically to ensure no loss. We will be using these Spot instances as a part of the guidelines and will look at how to save the files on a Amazon S3 Machine.

3. AMI - Suppose you install a bunch of softwares on an EC2 instance and use it for a couple of hours and then terminate the instance. It is painful to install these softwares everytime you run the system. So Amazon offers you to save the image of the machine which can boot up with the sofwares installed but this will cost some additional money. Alternatively you can select from community AMI to select a Machine with pre-installed softwares to save the money and hassle.

4. Regions - Amazon offers its services in different regions and data within a region is duplicated to ensure robustness. AMI discussed above are restricted to a region and an AMI in one region might not be available in another.

* Setting up Instructions.
** Connecting to Amazon EC2 instance.
1. Log in to Amazon Console.
2. Select a region from top right. For this tutorial we will use N.California.
3. Go to Services -> EC2 -> Spot Requests ( Under Instances. You can have a look at the pricing history to get a guide on the bid )
4. Choose an AMI. To show a complete setup, I will go with AMI Ubuntu Server 14.04 LTS (HVM), SSD Volume Type. ( I will recommend you to try out complete installation once. Later you can go with the AMI I have mentioned.)
5. From the available options go GPU g2.2xlarge instance. (This will cost you money. Small amout :) )
6. Press Next and Enter your bidding price based on the history suggested. At the time of writing this current price is around $0.07/hour so I am going to bid $0.1/hour to have a safety margin.
7. Press Review and Launch. Press Launch. You will come to a screen where it will ask for a Key-Pair. If you are using it for the first time, then please create a new key-pair and make sure you keep it safely. If lost it cannot be retrieved later on.
8. If on Linux use this command to connect to EC2 instace
  ssh -i EC2KeyPair.pem ubuntu@<your instance ip address>
  If on windows use putty to connect to the instance. The instructions can be found here.
  http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/putty.html
  In nutshell, do the following.
  1. Open PuTTYgen
  2. Click load and load <key>.pem 
  3. Press save private key
  4. After saving private key in ppk format open PuTTy
  5. In host enter ubuntu@DNS name. This can be seen by cliking the instance in AWS services and copying Public DNS column
  6. Now SSH -> Auth Tab and browse private key. and press open. You will be connected to the Amazon EC2 Server

** Installing Theano and other libraries.
1. Run these commands to install basic libraries
  sudo apt-get update
  sudo apt-get -y dist-upgrade
  sudo apt-get install -y gcc g++ gfortran build-essential git wget linux-image-generic libopenblas-dev python-dev python-pip python-nose python-numpy python-scipy
  sudo apt-get install -y liblapack-dev
  sudo apt-get install -y libblas-dev

2. Install Cuda Running these commands. Install latest cuda- cuda-repo-ubuntu1404_7.5-18_amd64.deb ( This is cuda 7.5)
  wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
  sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
  sudo apt-get update
  sudo apt-get install cuda

3. Reboot and reconnect
  sudo reboot
  
4. Add paths
  CUDA_ROOT=`find /usr/local/ -type d -name "cuda-[0-9]\.[0-9]" -print`
  The above command will find the cuda path and for cuda-7.5 it should be */usr/local/cuda-7.5
  Run following commands to add paths to the library. Many issues that usually come are due to wrong path settings. Make sure you import path in the environment. 
  
  export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH} 
  export PATH=${CUDA_ROOT}/bin:${PATH} 
  echo "PATH=${CUDA_ROOT}/bin:${PATH}" >> .bashrc
  echo "export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH}" >> .bashrc
  source ~/.bashrc

5. Verify Cuda is working. In the commands set right version for cuda. Here it was 7.5
  
  cuda-install-samples-7.5.sh  ~ 
  cd ~/NVIDIA_CUDA-7.5_Samples 
  cd 1_Utilities/deviceQuery 
  make
  ./deviceQuery
  
6. Go to home folder and install sklearn and nolearn
  cd ~
  sudo pip install scikit-learn nolearn
  
7. Install cudamat
  git clone https://github.com/cudamat/cudamat
  cd cudamat
  pip install --user .
  
8. Install theano
  sudo pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt

9. Configure Theano - Right now if you run python and import theano it will not use GPU. In order to use GPU we need to configure theanorc
  Create file ~/.theanorc and copy
  
  [global]
  floatX = float32
  device = gpu0
  
  [nvcc]
  fastmath = True

10. Verify if GPU is used. Open Python and import theano, if GPU is being used it will print the GPU information.

** OPTIONAL DEEP LEARNING TUTORIAL
A. Install Deep Learning Tutorial to playaround.
  cd ~
  git clone https://github.com/lisa-lab/DeepLearningTutorials.git

B. Install datasets and play around
  cd DeepLearningTutorials/data/
  chmod +x download.sh
  ./download.sh

C. Run convolutional_mlp.py
  cd ../code/
  time python convolutional_mlp.py
  
  It takes around 39-40 minutes to run on GPU instance 

*** Download Dataset from Kaggle
Steps :-
1. Export your cookies from your browser, when you are  logged in at kaggle and put your cookies.txt on your server. Then run:
  mkdir data
  wget -x --load-cookies cookies.txt -P data -nH --cut-dirs=5 http://www.kaggle.com/c/dogs-vs-cats/download/test1.zip

** Install Lasagne and skimage
1. Intall Lasagne 
  sudo pip install Lasagne==0.1
  sudo apt-get update
  sudo pip install scikit-learn
  sudo apt-get install python-matplotlib
  sudo pip install scikit-image

2. Install Cudadnn after making an account at nvidia
  https://developer.nvidia.com/cudnn
  1. Register and click download
  2. Right cuDNN v4 Library for Linux click and click copy link address
  3. Now there are 2 options either download file to your local machine and then upload to S3 and download from S3 and run next commands or to download directly to the ec2 machine follow the steps Download dataset from kaggle and use the cookies.txt generated in the cudnn download link page
  4. copy cookies.txt (on nvdia link page ) to the ec2 machine and run
    
    mkdir data
    wget -x --load-cookies cookies.txt -P data -nH --cut-dirs=5 https://developer.nvidia.com/rdp/assets/cudnn-70-linux-x64-v40
    cd data
    mv cudnn-70-linux-x64-v40 cudnn-70-linux-x64-v40.tar.gz
    tar -xvzf cudnn-70-linux-x64-v40.tar.gz
    CUDA_ROOT=`find /usr/local/ -type d -name "cuda-[0-9]\.[0-9]" -print`
    cd cuda/include/
    sudo cp `ls *.h` "$CUDA_ROOT"/include/
    cd ../lib64/
    sudo cp `ls *`  "$CUDA_ROOT"/lib64/
    export LD_LIBRARY_PATH=${CUDA_ROOT}/lib64:${LD_LIBRARY_PATH} 
    export PATH=${CUDA_ROOT}/bin:${PATH}  
  
  5. For faster use in future upload cudnn-70-linux-x64-v40.tar.gz to S3 bucket and retrieve from there to use.
  6. At times when lasagne gives some error, one of the three commands solves the purpose
    sudo pip install -r https://raw.githubusercontent.com/dnouri/kfkd-tutorial/master/requirements.txt
    sudo pip install -r https://raw.githubusercontent.com/dnouri/nolearn/master/requirements.txt
    sudo pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
  
** Install awscli to move files to and from S3

1. Create S3 bucket
  https://www.youtube.com/watch?v=wODEO2Tvmik
2. Set up Access Key
  Read How to Retrieve Root Access Keys (first one)
  http://www.cloudberrylab.com/blog/how-to-find-your-aws-access-key-id-and-secret-access-key-and-register-with-cloudberry-s3-explorer/
3. On on your ec2 instance, Install awscli and configure it
  sudo apt-get install awscli
  aws configure
  
  Configure will ask 4 questions
  AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
  AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
  Default region name [None]: us-west-2
  Default output format [None]: ENTER
  
  In third question you can see the region by having a look at region properties. For N.California bucket it is us-west-1
  You can connect to any bucket in any region.

4. Transfer file/folder to S3
  # Make test folder
  mkdir TestTransfer
  aws s3 cp TestTransfer s3://apoorvfirstbucket/DeepLearning/SpatialTransformerNetwork/CatDog/TestTransfer --recursive
  
  # Create a test file
  vim test.txt
  aws s3 cp test.txt s3://apoorvfirstbucket/DeepLearning/SpatialTransformerNetwork/CatDog/test.txt 

5. Retrieve from file/folder from S3
  aws s3 cp s3://apoorvfirstbucket/DeepLearning/SpatialTransformerNetwork/CatDog/TestTransfer TestTransfer --recursive
  aws s3 cp  s3://apoorvfirstbucket/DeepLearning/SpatialTransformerNetwork/CatDog/test.txt test.txt
  

  

  
