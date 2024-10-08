# Lumbar Spine Degenerative Condition Detection using MRI Images

## Overview

This project is developed for the [Kaggle competition](https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/overview) aimed at enhancing the detection and classification of degenerative spine conditions through the analysis of lumbar spine MRI images. The competition is hosted by the Radiological Society of North America (RSNA) in collaboration with the American Society of Neuroradiology (ASNR).

### Dataset (Data)
The competition provides a comprehensive dataset sourced from eight medical sites across five continents. A single data point consists of a study. Each study has several series of images, and each series of images is made up of several different images at different levels. Each study maps to an inference on several conditions (Left and right neural_foraminal_narrowing for each vertebrae from L1-L2 to L5-s1, left and right subarticular stenosis for each of the same set of vertebrae, and spinal canal stenosis). 
- **Study**: A single study is the primary data point that is mapped to a set of labels. Each study corresponds to a unique patient case with varying levels of spine degeneration.
- **Series**: Each study is comprised of multiple series of images, representing different sequences of MRI scans taken from the same patient. Each series captures the lumbar spine from various angles or sequences, aiding in comprehensive analysis.
- **Instance**: Each series consists of multiple images captured at different vertebral levels. These images provide detailed information necessary for diagnosing conditions affecting the lumbar spine.


### Dataset (Labels)
We are making inferences about 5 different conditions. 
- **Left Neural Foraminal Narrowing**
- **Right Neural Foraminal Narrowing**
- **Left Subarticular Stenosis**
- **Right Subarticular Stenosis**
- **Spinal Canal Stenosis**

We are making inferences for each of the above 5 conditions for each of the following vertebrae
- **L1-L2**
- **L2-L3**
- **L3-L4**
- **L4-L5**
- **L5-s1**

The inferences for each pair of condition and vertebrae, will be one of three classes. 
- **Normal/Mild**
- **Moderate**
- **Severe**

### Project Structure

- `train.csv`: Training labels.
- `train_label_coordinates.csv`: Coordinates for labels.
- `train_images/`: Directory containing training MRI images.
- `test_images/`: Directory containing test MRI images.
- `submission.csv`: Format for submission predictions.
- `notebooks/`: Jupyter notebooks containing the modeling process and analysis.
- `requirements.txt`: Dependencies and libraries required for the project.
