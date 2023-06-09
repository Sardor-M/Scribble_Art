# Scribble_Art

- This is the OpenCv implementation space.

## Description

- This is a small application that allows you to convert the given input image(s)
  to a cartoonized form of look. Here, i used the following methods :

  - Median Blurring,
  - Adaptive Threshholding,
  - Bilateral Blurring,
  - Stylization,
  - Color Quantization

Please refer to image results to learn more about it

## Results

### Converted Images

<p align ="left"> <img width=47% src="result_images/after-median-blurring.png "/><img align = "right"width=47% src="result_images/after-adaptive-thresholding.png"/></p>
<p align ="left"> <img width=47% src="result_images/after-bilateral-filtering.png "/><img align = "right"width=47% src="result_images/after-stylization.png"/></p>

### Final Result Image

- Final result image result comparison with the original image.

|   Original Image    |          Final Result Image (1)           |
| :-----------------: | :---------------------------------------: |
| ![](Bald-Eagle.jpg) | ![](result_images/after-quantization.png) |

### Final Result Comparison

- This is the final result image comparison with ChatGPT generated code example:

|       Final Result Image (2)       |      ChatGPT generated code's result       |
| :--------------------------------: | :----------------------------------------: |
| ![](result_images/final-image.png) | ![](result_images/ChatGPT_code_result.png) |
