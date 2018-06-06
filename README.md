# keras-semantic-segmentation-example
Example of semantic segmentation in Keras

## Single class example:
Generated data: random ellipse with random color on random color background and with random noise added.

Result: 1st images is input image, 2nd image is ground truth mask, 3rd image is probability, 4th image is probability thresholded at 0.5.
![alt tag](https://github.com/mrgloom/keras-semantic-segmentation-example/blob/master/misc/binary_crossentropy_result_binary_segmentation.png)

## Multi-class example:
Generated data: first class random is ellipse with random color and second class is random rectangle with random color on random color background and with random noise added.

Result: 1st images is input image, 2nd image is ground truth mask, 3rd image is probability, 4th image is probability thresholded at 0.5.
![alt tag](https://github.com/mrgloom/keras-semantic-segmentation-example/blob/master/misc/binary_crossentropy_result_multilabel_segmentation.png)
