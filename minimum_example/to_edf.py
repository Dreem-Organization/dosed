import os
import h5py
import numpy as np
import pyedflib
from tqdm import tqdm
import datetime
   
def convert_to_edf(h5_file):
    """
    A function to convert from H5 files as defined in 
    https://github.com/Dreem-Organization/dreem-learning-open
    back to edf. Might be handy if you want to view certain 
    signals in an external viewer (e.g. EDFBrowser, VisBrain)
    """
    with h5py.File(h5_file, 'r') as f:
        hypnogram = np.array(f['hypnogram'], dtype=int)
        signals = f['signals']
        
        extracted = []
        signal_headers = []
        
        header = {'birthdate': '',
                  'startdate': datetime.datetime.now(),
                  'gender': '',
                  'admincode': '',
                  'equipment': '',
                  'patientcode': '',
                  'patient_additional': '',
                  'patientname': '',
                  'recording_additional': '',
                  'technician': ''}
        
        # extract all signals contained in the h5
        for group_key in signals.keys():
            group = signals[group_key]
            for signal_key in tqdm(group.keys()):
                signal = group[signal_key]
                data = np.array(signal)

                signal_header = {'label': signal_key, 
                                   'dimension': group.attrs['unit'].decode(), 
                                   'sample_rate': group.attrs['fs'], 
                                   'physical_min': data.min(), 
                                   'physical_max': data.max(), 
                                   'digital_min':  -32768, 
                                   'digital_max':  32767, 
                                   'transducer': '', 
                                   'prefilter': ''}
                signal_headers.append(signal_header)
                extracted.append(data)
                
        edf_file = os.path.splitext(h5_file)[0] + '.edf'
     
        # write edf
        with pyedflib.EdfWriter(edf_file, n_channels=len(extracted)) as f:  
            f.setSignalHeaders(signal_headers)
            f.setHeader(header)
            f.writeSamples(extracted, digital=False)
        
        # write hypnogram
        np.savetxt(edf_file + '.txt', hypnogram, fmt='%d')

