# CAP5516-Programming-Assignment-01

## Download Dataset

Pediatric Chest X-Ray Images (Pneumonia) can be downloaded from the following Kaggle link:

https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

--------------------------------------------------------------------------------------------------------------------

## Download Trained Models and Results

The trained models along with the Accuracy and Loss Curves for all experiments can be downloaded from the folllowing OneDrive link. The folder also contains CAM visualizations with Original and Predicted Classes on the complete test dataset.

https://drive.google.com/file/d/1n1Ntze68f6jIodmqzF3OppHTmRfoigv9/view?usp=sharing

--------------------------------------------------------------------------------------------------------------------

## Setting Up environment and 
To replicate the results perform the following steps:

--------------------------------------------------------------------------------------------------------------------

-To create and setup conda environment using the following commands:

conda create -n my_env

conda activate my_env

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip3 install opencv-python

pip3 install tqdm

pip3 install matplotlib

--------------------------------------------------------------------------------------------------------------------

-To execute the training experiments:

python3 train.py

--------------------------------------------------------------------------------------------------------------------

-To execute the CAM visualizations:

python3 cam.py -mp RN18_pretrained_ALLaugmented_LR0.01_Ep15 -ip cam_inputs/test/PNEUMONIA -op cam_outputs/CAM_on_test

--------------------------------------------------------------------------------------------------------------------


