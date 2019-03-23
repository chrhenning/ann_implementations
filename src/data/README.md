# Datasets

Here you can find instructions about how to prepare some of the datasets for automatic processing.

## Large-scale CelebFaces Attributes (CelebA) Dataset

[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) is a dataset with over 200K celebrity images. It can be downloaded from [here](https://drive.google.com/open?id=0B7EVK8r0v71pWEZsZE9oNnFzTm8).

Google Drive will split the dataset into multiple zip-files. In the following, we explain, how you can extract these files on Linux. To decompress the sharded zip files, simply open a terminal, move to the downloaded zip-files and enter:

```
unzip '*.zip'
```

This will create a local folder named `CelebA`.

Afterwards move into the `Img` subfolder: `cd ./CelebA/Img/`.

You can now decide, whether you want to use the JPG or PNG encoded images.

For the jpeg images, you have to enter:

```
unzip img_align_celeba.zip
```

This will create a folder `img_align_celeba`, containing all images in jpeg format.

To save space on your local machine, you may delete the zip file via `rm img_align_celeba.zip`.

The same images are also available in png format. To extract these, you have to move in the corresponding subdirectory via `cd img_align_celeba_png.7z`. You can now extract the sharded 7z files by entering:

```
7z e img_align_celeba_png.7z.001
```

Again, you may now delete the archives to save space via `rm img_align_celeba_png.7z.0*`.

You can proceed similarly if you want to work with the original images located in the folder `img_celeba.7z`.

FYI, there are scripts available (e.g., [here](https://github.com/carpedm20/DCGAN-tensorflow/blob/master/download.py)), that can be used to download the dataset.
