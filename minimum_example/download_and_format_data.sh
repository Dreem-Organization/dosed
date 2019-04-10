#!/usr/bin/env bash

cd ..
cp settings-template.py settings.py
python minimum_example/download_data.py
python minimum_example/to_h5.py