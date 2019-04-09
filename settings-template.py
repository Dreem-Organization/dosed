import os


BASE_DIRECTORY = os.path.dirname(os.path.realpath(__file__)) + "/data/"
DOWNLOAD_DIRECTORY = BASE_DIRECTORY + "downloads/"
H5_DIRECTORY = BASE_DIRECTORY + "h5/"

if not os.path.isdir(BASE_DIRECTORY):
    os.mkdir(BASE_DIRECTORY)

if not os.path.isdir(DOWNLOAD_DIRECTORY):
    os.mkdir(DOWNLOAD_DIRECTORY)

if not os.path.isdir(H5_DIRECTORY):
    os.mkdir(H5_DIRECTORY)

MINIMUM_EXAMPLE_SETTINGS = {
    "download_directory": DOWNLOAD_DIRECTORY + "minimum_example/",
    "h5_directory": H5_DIRECTORY + "minimum_example/",
}
