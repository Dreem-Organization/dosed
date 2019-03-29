docker build -t lol .

docker run --runtime=nvidia -it -v "${PWD}:/workspace" lol bash

