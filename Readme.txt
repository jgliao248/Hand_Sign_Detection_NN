
# Libraries:

Install these libraries prior to running the program.

- opencv-python
- mediapipe
    - note:
        apple-silicon requires the use of a third party version: pip install mediapipe-silicon
        Must be running Python 3.7 - 3.10 https://github.com/google/mediapipe/blob/master/docs/getting_started/troubleshooting.md#python-pip-install-failure
- tensorflow
    - note:
        apply-silicon version
        https://developer.apple.com/metal/tensorflow-plugin/



# Notes:
- performance was not great with convolution based neural network. Perhaps, the low resolution made it hard to properly
train the model. The model gets trained within 1 or 2 epochs of train batch of 128 and test batch of 64.

- when using landmarks, it is difficult to find landmarks for certain letter photo examples.


Resources
- https://www.learnpytorch.io/04_pytorch_custom_datasets/#5-option-2-loading-image-data-with-a-custom-dataset
