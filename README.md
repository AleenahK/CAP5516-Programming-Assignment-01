# CAP5516-Programming-Assignment-01

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


