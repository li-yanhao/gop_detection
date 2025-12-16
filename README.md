# 🎥 A Contrario Detection of H.264 Video Double Compression


This is a project detecting whether a video has been recompressed,
and to estimate the fixed size of the primary Group of Pictures (GOP) of 
a recompressd video. ([github link](https://github.com/li-yanhao/gop_detection/tree/apate))

The project consists of two parts:
* An inspector for H.264 videos that extracts the
intermediate data during the decompression. At present it can
extract the prediction residuals, macroblock types, frame types,
display order and picture coding order. The extractor is based on the
[JM software](https://iphome.hhi.de/suehring/tml/) 
and its [extension](https://vqeg.github.io/software-tools/encoding/modified-avc-codec/).
* An _a Contrario_ detector that detects potential periodic sequence
of residual peaks in P-frames and validate the sequence if the Number
of False Alarms (NFA) is significantly small.


## 🛠️ Before using


1. Clone the project (branch `apate`):

    step 1: Install Git LFS ([docs](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)) if it is not done yet, otherwise skip this step.

    Ubuntu / Debian:
    ```bash
      sudo apt install git-lfs
    ```

    MacOS:
    ```bash
      brew install git-lfs
    ```

    Then initialize Git LFS:
    ```bash
      git lfs install
    ```

    step 2: Clone the repository (branch `apate`):
    ```bash
    git clone -b apate --recurse-submodules https://github.com/li-yanhao/gop_detection.git
    ```

2. Install [ffmpeg](https://ffmpeg.org/). You could use a 3rd-party tool to install ffmpeg:

    Ubuntu / Debian:
    ```bash
    sudo add-apt-repository ppa:savoury1/ffmpeg4 -y
    sudo add-apt-repository ppa:savoury1/ffmpeg5 -y
    sudo apt update
    sudo apt install ffmpeg
    ```
   MacOS:
    ```bash
   brew install ffmpeg
    ```
   
    Or install from the official [website](https://ffmpeg.org/download.html).
    
   Make sure ffmpeg can be run from bash, e.g.:
    ```bash
    $ ffmpeg -version
   
   ffmpeg version 5.1.2 Copyright (c) 2000-2022 the FFmpeg developers
    built with Apple clang version 14.0.0 (clang-1400.0.29.202)
    configuration: --prefix=/opt/homebrew/Cellar/ffmpeg/5.1.2_1 --enable-shared --enable-pthreads --enable-version3 --cc=clang --host-cflags= --host-ldflags= --enable-ffplay --enable-gnutls --enable-gpl --enable-libaom --enable-libbluray --enable-libdav1d --enable-libmp3lame --enable-libopus --enable-librav1e --enable-librist --enable-librubberband --enable-libsnappy --enable-libsrt --enable-libtesseract --enable-libtheora --enable-libvidstab --enable-libvmaf --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxml2 --enable-libxvid --enable-lzma --enable-libfontconfig --enable-libfreetype --enable-frei0r --enable-libass --enable-libopencore-amrnb --enable-libopencore-amrwb --enable-libopenjpeg --enable-libspeex --enable-libsoxr --enable-libzmq --enable-libzimg --disable-libjack --disable-indev=jack --enable-videotoolbox --enable-neon
    libavutil      57. 28.100 / 57. 28.100
    libavcodec     59. 37.100 / 59. 37.100
    libavformat    59. 27.100 / 59. 27.100
    libavdevice    59.  7.100 / 59.  7.100
    libavfilter     8. 44.100 /  8. 44.100
    libswscale      6.  7.100 /  6.  7.100
    libswresample   4.  7.100 /  4.  7.100
    libpostproc    56.  6.100 / 56.  6.100
    ```


3. Compile the H.264 decoder (JM software)

    step 1: Prepare the library dependencies for `libpng`, `libtiff` and `libjpeg`:

    Ubuntu / Debian:
    ```
      sudo apt update
      sudo apt install -y \
        build-essential \
        libpng-dev \
        libtiff-dev \
        libjpeg-dev

    ```
    MacOS:
    ```bash
      brew install libpng libtiff jpeg
    ```

    step 2: Compile the decoder

    ```bash
    cd jm
    make -j ldecod
    ```

4. Install the python requirements for the _a Contrario_ detector.
The code was tested in python 3.10.19
   ```bash
   conda create --name apate python=3.10.19
   conda activate apate
   
   pip install -r requirements.txt

   ```

🎉 Done! Now all the prerequisites are installed.

5. (Optional for GUI users) Install the GUI plugin `tkinter` on your system if not installed yet:

   Ubuntu / Debian:
   ```bash
   sudo apt install python3-tk
   ```

   MacOS:
   ```bash
   brew install python-tk
   ```

## 🖱️ Usage with interactive GUI


The input video file must be encoded in H.264. It can be a video file in extension `.mp4`, `.avi`, `.mkv`, `.mov`, `.qt`, `.264`.

You can run the program with a GUI to select the ROI interactively in order to perform the detection on a specific area of the video.
```bash
python perform_video_analysis_gui.py [-h] [--d D] [--space SPACE] [--epsilon EPSILON] [--out_folder OUT_FOLDER] video_path
```
where:
* `d`: number of neighbors to validate a peak residual (default: 3)
* `space`: color space used for detection, accepted values are `{'Y', 'U', 'V'}`, (default: `'Y'`)
* `epsilon`: threshold for the Number of False Alarms (NFA), (default: 0.05)
* `out_folder`: output folder for results (default: `results/`)

The execution arguments and results will be saved in `<out_folder>/<date_of_execution>/`.

```
results/<date_of_execution>/
├── args.txt         # the input arguments used for the detection
├── detections.txt   # the detected candidates with periodicity, empty if no candidate is found
├── histogram.png    # the histogram of the residuals of all the frames
├── histogram.html   # the histogram of the residuals of all the frames, needs a web browser to open
└── mask.png         # a ROI mask selected interactively
```


### 🧪 Example: detecting double compression in a faceswap video with ROI mask

For instance, in a faceswap video `asset/fake_003.mp4`, the background area originated from a primary authentic video may have been compressed twice, while the face area modified or generated by software may have been compressed only once during the final compression. 

Command:
```bash
python perform_video_analysis_gui.py asset/fake_003.mp4
```

Running the detection on the face area does NOT find any evidence of double compression:
![alt text](asset/003_fake_face_viz.png)
(you can press Left Arrow / Right Arrow to navigate through frames in the GUI)

The histogram of the residuals in P-frames does NOT show any periodic pattern:
![alt text](asset/003_fake_face.png)




Running the detection on the background area results in positive detection of double compression:
![alt text](asset/003_fake_background_viz.png)
``` 
Detected candidates (by A Contrario analysis):
  Periodicity = 15, Offset = 0, NFA = 0.002004693634577336
  Periodicity = 15, Offset = 14, NFA = 0.0006576445075819903
  Periodicity = 30, Offset = 0, NFA = 7.679586275143973e-13
  Periodicity = 30, Offset = 29, NFA = 3.843936739891997e-12
  Periodicity = 60, Offset = 0, NFA = 3.24848871735527e-06
  Periodicity = 60, Offset = 29, NFA = 0.010216038405910469
  Periodicity = 60, Offset = 30, NFA = 0.0031332629116749638
  Periodicity = 60, Offset = 59, NFA = 3.24848871735527e-06
  Periodicity = 90, Offset = 0, NFA = 0.0009634826448104125
  Periodicity = 90, Offset = 89, NFA = 0.0009634826448104125

The most prominent candidate: periodicity = 30, NFA = 7.679586275143973e-13
```
The histogram of the residuals in P-frames shows clear periodic peaks (highlighted in <span style="color:cyan"> cyan</span>):
![alt text](asset/003_fake_background.png)


## 💻 Usage with CLI

If GUI is not desired, you can also run the program with command line arguments to specify the ROI mask and other parameters:
```bash
usage: perform_video_analysis.py [-h] [--d D] [--space SPACE] [--epsilon EPSILON] [--mask_path MASK_PATH] [--out_folder OUT_FOLDER] video_path
```
where:
* `d`: number of neighbors to validate a peak residual (default: 3)
* `space`: color space used for detection, accepted values are `{'Y', 'U', 'V'}` (default: `'Y'`)
* `epsilon`: threshold for the Number of False Alarms (NFA) (default: 0.05)
* `mask_path`: path to the binary ROI mask image in `.png`. If the mask is not in the same shape as the video frames, it will be readjusted to the video frame size by zero padding or cropping (default: `None`)
* `out_folder`: output folder for results (default: `results/`)

The execution arguments and results will be saved in `<out_folder>/<date_of_execution>/`.

### 🧪 Example 1: detecting double compression in a full video
Given a fully recompressed video `asset/office_recompressed.mp4`, we can run:
```bash
python perform_video_analysis.py asset/office_recompressed.mp4
```
which gives detection results:
```
Detected candidates (by A Contrario analysis):
  Periodicity = 31, Offset = 0, NFA = 0.00018751240128970711

The most prominent candidate: periodicity = 31, NFA = 0.00018751240128970711
```

On the other hand, if we run the detection on an original video `asset/office_original.mp4`:
```bash
python perform_video_analysis.py asset/office_original.mp4
```
which gives no detection:
```
No periodicity detected!
```



### 🧪 Example 2: detecting recompressed background in a faceswap video with ROI mask

Given a faceswap video `asset/fake_003.mp4` and a binary ROI mask image `asset/fake_003_background_mask.png` selecting the background area, we can run:
```bash
python perform_video_analysis.py asset/fake_003.mp4 --mask_path asset/fake_003_background_mask.png
```
which gives detection results:
```
Detected candidates (by A Contrario analysis):
  Periodicity = 15, Offset = 0, NFA = 0.002004693634577336
  Periodicity = 15, Offset = 14, NFA = 0.0006576445075819903
  Periodicity = 30, Offset = 0, NFA = 7.679586275143973e-13
  Periodicity = 30, Offset = 29, NFA = 3.843936739891997e-12
  Periodicity = 60, Offset = 0, NFA = 3.24848871735527e-06
  Periodicity = 60, Offset = 29, NFA = 0.010216038405910469
  Periodicity = 60, Offset = 30, NFA = 0.0031332629116749638
  Periodicity = 60, Offset = 59, NFA = 3.24848871735527e-06
  Periodicity = 90, Offset = 0, NFA = 0.0009634826448104125
  Periodicity = 90, Offset = 89, NFA = 0.0009634826448104125

The most prominent candidate: periodicity = 30, NFA = 7.679586275143973e-13
```



## 🗑️ Large temporary files

At each run the program decodes the full frames and prediction residuals in a temporary folder under `gop_detection/tmp/`, then 
detects double compression using these data. The intermediate
data can take several GB depending on the resolution and
the length of the video. You can safely delete the `tmp/` folder after running the program.

## 📖 Citation
If you find this code useful in your research, please cite the following paper:

```@inproceedings{li2023contrario,
  title={A contrario detection of h. 264 video double compression},
  author={Li, Yanhao and Gardella, Marina and Bammey, Quentin and Nikoukhah, Tina and Morel, Jean-Michel and Colom, Miguel and Von Gioi, Rafael Grompone},
  booktitle={2023 IEEE International Conference on Image Processing (ICIP)},
  pages={1765--1769},
  year={2023},
  organization={IEEE}
}
```
