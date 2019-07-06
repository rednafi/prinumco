# PriNumco

[![Dataset](https://img.shields.io/badge/Dataset-Prinumco-red.svg)](https://drive.google.com/file/d/1ICR4I5Rmtst8eGFqQFudhnG4YSEIWozq/view?usp=sharing)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
![stability-experimental](https://img.shields.io/badge/stability-experimental-orange.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

- [PriNumco](#PriNumco)
  - [Dataset Summary](#Dataset-Summary)
  - [Sample Images](#Sample-Images)
    - [Original Images](#Original-Images)
    - [Augmented Images](#Augmented-Images)
    - [List of Applied Augmentations](#List-of-Applied-Augmentations)
  - [Requirements](#Requirements)
  - [Script Arrangement & Order of Execution](#Script-Arrangement--Order-of-Execution)
  - [Contributors](#Contributors)

Several initiatives have been taken to label and aggregate Bengali handwritten digit images with the aim of constructing robust digit recognition systems. However, deeplearning models trained on handwritten digit data do not generalize well on images of printed digits. PriNumco is a Compilation of Printed Bengali Digit images which aims to provide an extremely robust training-validation dataset for building recognition systems that can classify images of printed digits, sourced from license plates, billboards, banners and other real-life data sources.

## Dataset Summary

Initially, the script uses 58 different Bengali fonts to generate 2320  (256 x 256) images of 10 digits (232 images per digit) and propels them through an augmentation pipeline to generate 200k train images. A similar procedure with different augmentation pipeline was followed to generate 30k test images.

## Sample Images

### Original Images

<img src="https://user-images.githubusercontent.com/30027932/55688081-62db0c80-5996-11e9-9b85-245ef2e50469.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/30027932/55688087-7a19fa00-5996-11e9-908f-49ab1879b223.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/30027932/55688089-89994300-5996-11e9-866e-4f31c3e1a8c0.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/30027932/55688092-9c137c80-5996-11e9-8400-033de289a7fb.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/30027932/55688104-c2d1b300-5996-11e9-9907-bf4263cd4db9.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/30027932/55688109-d250fc00-5996-11e9-999b-cdcaf01e3f4b.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/30027932/55688114-e0068180-5996-11e9-8fe5-b4011de41083.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/30027932/55688119-f8769c00-5996-11e9-915c-73558b0cb65a.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/30027932/55688127-07f5e500-5997-11e9-85f8-88b6e1136530.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/30027932/55688132-18a65b00-5997-11e9-946f-e4c611a859ef.jpg" width="18%"></img> 

### Augmented Images

<img src="https://user-images.githubusercontent.com/30027932/55688245-5e175800-5998-11e9-84d8-407fa6c98ea4.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/30027932/55688250-6a9bb080-5998-11e9-8a2c-7ff60e8f3e93.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/30027932/55688259-78e9cc80-5998-11e9-8eba-687f011ba9cd.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/30027932/55688269-8acb6f80-5998-11e9-870c-e2abb153fcb8.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/30027932/55688279-9cad1280-5998-11e9-84bd-aeffa354f675.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/30027932/55688287-acc4f200-5998-11e9-836a-c12fc34b0db1.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/30027932/55688293-bd756800-5998-11e9-8672-27c41a5afb4f.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/30027932/55688305-d847dc80-5998-11e9-8e98-c509730c8f25.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/30027932/55688355-69b74e80-5999-11e9-8c7c-ee9b95a83580.jpg" width="18%"></img> <img src="https://user-images.githubusercontent.com/30027932/55688383-b4d16180-5999-11e9-836a-724c300d26d4.jpg" width="18%"></img> 

Generated images are organized in the following hierarchy:

    ------------------------------------------------------------------------------------------------------
    Folder Structure
    ------------------------------------------------------------------------------------------------------

    .
    ├── dataset/
        ├── train/
            ├── 0/
                ├── img0.jpg
                ├── img1.jpg
                ...........
            ├── 1/
                ├── img0.jpg
                ├── img1.jpg
                ...........
            ├── 2/
                .....
        ├── test/
            ├── 0/
                ├── img0.jpg
                ├── img1.jpg
                ...........
            ├── 1/
                ├── img0.jpg
                ├──img1.jpg
                ...........
            ├── 2/
                .....     

### List of Applied Augmentations

In order to mimic real life images of Bengali digits, we generated the images with _white_, _yellow_, _sky blue_ and _teal_ colored backgrounds and used [augmentor](https://github.com/mdbloice/Augmentor) library to apply the following augmentations on both the train and test dataset:

-   **gaussian_noise**(probability=0.3, mean=0, std=20.0 )
-   **blur**(probability=0.6, blur_type='random', radius=(0, 100), fixed_radius=3)
-   **random_filter**(probability=0.9, filter_type='random', size=5)
-   **black_and_white**(probability = 1, threshold = 128)
-   **rotate**(probability=0.3, max_left_rotation=25, max_right_rotation=25)
-   **rotate90**(probability=0.005)
-   **rotate270**(probability=0.005)
-   **zoom**(probability=0.1, min_factor=1.01, max_factor=1.03)
-   **skew_tilt**(probability=0.01, magnitude=1)
-   **skew_left_right**(probability=0.02, magnitude=1)
-   **skew_top_bottom**(probability=0.03, magnitude=1)
-   **skew_corner**(probability=0.03, magnitude=1)
-   **skew**(probability=0.01, magnitude=1)- [PriNumco](#PriNumco)
  - [Dataset Summary](#Dataset-Summary)
  - [Sample Images](#Sample-Images)
    - [Original Images](#Original-Images)
    - [Augmented Images](#Augmented-Images)
    - [List of Applied Augmentations](#List-of-Applied-Augmentations)
  - [Requirements](#Requirements)
  - [Script Arrangement & Order of Execution](#Script-Arrangement--Order-of-Execution)
  - [Contributors](#Contributors)
-   **random_erasing**(probability=0.01, rectangle_area=0.11)
-   **random_brightness**(probability=0.5, min_factor=0.5, max_factor=1.5)
-   **random_color**(probability=0.2, min_factor=0, max_factor=1)
-   **random_contrast**(probability=0.3, min_factor=0.4, max_factor=1)
-   **invert**(probability=0.09)
-   **resize**(probability=1, width=256, height=256)

For further details on individual augmentation operation, please checkout the [documentation](https://augmentor.readthedocs.io/en/master/code.html#Augmentor.Pipeline.Pipeline.crop_random) of augmentor libarary.

## Requirements

-   Ubunutu 18.04 Bionic Beaver

**Image Generation**

    pip install -r requirements.txt

**Running the CNN model**

-   tensorflow 1.12/2.0 alpha
-   keras 2.2.4 (Only applicable for tensorflow 1.12)

## Script Arrangement & Order of Execution

**Image Generation and Augmentation**

    ------------------------------------------------------------------------------------------------------
    Folder Structure
    ------------------------------------------------------------------------------------------------------

    .
    ├── bfonts/
    ├── digit_generation_utils/
        ├── directory_generation_check_font.py (Making necessary directories to contain the dataset)
        ├── digit_generation.py (Generating the actual digits)
        ├── image_augmentation.py (Augmenting the images)
        ├── mixing_aug_image_with_gen_image.py (Mixing the augmented images with the generated images)
    ├── digit_gen_main.py (Script containing the main function)

To run the digit generation and augmentation pipeline,

-   Make a folder name `custom` in the path `/usr/share/fonts/truetype` 
-   Copy the fonts from the `bfonts` folder to `/usr/share/fonts/truetype/custom` path
-   Run the `main.py` file to generate, augment and prepare and the images in their corresponding folders. 
         

**Training and Validating a Baseline CNN model**

    ------------------------------------------------------------------------------------------------------
    Folder Structure
    ------------------------------------------------------------------------------------------------------

    .
    cnn_model/
    	├── train.py
    	├── test.py
    	├── prinumco_mobilenet.h5
    	├── test.png

We used tensorflow 2.0's keras API to construct and train [mobilenetV2](https://arxiv.org/abs/1801.04381) architecture to provide a baseline CNN model for benchmarking purposes. However, we also provided necessary scripts for training and testing the model in tensorflow 1.12.

-   Put the dataset folder in the primary folder (In case you haven't generated the images yourself) 
-   If you are using tensorflow 2.0, run the `train_tf2.0.py` file to train the baseline model
-   For testing out the model (tf 2.0), run the `test_tf2.0.py` file (This will load our pretrained model to predict the class of a sample `test.png` image)
-   Follow a similar procedure for training and testing in tensorflow 1.12.

## Contributors

-   [@manashmndl](https://github.com/manashmndl)
