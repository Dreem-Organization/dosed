cd .. & cp settings-template.py settings.py
cd .. & python -m minimum_example/download_data.py
cd .. & python -m minimum_example/to_h5.py
cd .. & python -m minimum_example/to_memmap.py
