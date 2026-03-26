from ultralytics import YOLO
import cv2
import h5py
import numpy as np

model = YOLO('YOUR_TRAINED_MODEL.pt')  # load a trained model

events_file = "YOUR_TEST_EVENTS_FILE.h5"
num_ev_histograms, _, height, width = events_file['data'].shape

out = cv2.VideoWriter('OUTPUT_FOLDER/output.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 20.0, (width, height))
with h5py.File(events_file, 'r') as f:
    for idx in range(num_ev_histograms):
        ev_histo = np.transpose(f['data'][idx], (1, 2, 0))
        results = model(ev_histo)  # return a generator of Results objects
        annotated_frame = results[0].plot()
        out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
out.release()