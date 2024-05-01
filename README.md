# Pneumonia Detector

Our project is to use machine learning and train the resnet-18 imageNet model using transfer learning to make a model that differentiates between healthy lungs and lungs infected with pneumonia. We have acquired a dataset in Kaggle with around 4,000 images for training, testing, and validation. However, due to hardware limitations, we will be forced to select only a portion of these images for testing. It can be expected for the recognition model to give results within the range of 40-50% confidence in identifying the correct results, but with more powerful hardware it could be possible to reach above 95% confidence when identifying infected lungs.

## Before running the program

To use the model on your device, it is important to make sure your hardware is powerful enough to run the model. A computing device made specifically for machine learning is recommended. The [jetson-inference](https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md) library must be already installed before use. 

## To run the program

1. Verify that your hardware can handle the GPU requirements of the recognition model.
2. Verify that the [jetson-inference](https://github.com/dusty-nv/jetson-inference/blob/master/docs/building-repo-2.md) library is installed on your device.
3. Download the 'model' directory and 'pneumoscanv2' and place them in the same directory.
4. cd into the directory containing the files before execution.
5. Run the program by executing the following command on the terminal:

    ```python3 pneumoscanv2.py <PATH TO SAMPLE IMAGE> <PATH TO OUTPUT IMAGE>```

The class type and confidence will be displayed on the terminal once the program has finished running.# pneumoscan
