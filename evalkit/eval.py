#!/usr/bin/python3

import glob
import re
import math
import os


def smooth(detections):
    result = [detections[0]]
    for i, det in enumerate(detections[1:], 1):
        length = int(det.split(':')[1])
        if length <= 10:
            prev_len, prev_score = int(result[-1].split(':')[1]), float(result[-1].split(':')[2])
            score = float(det.split(':')[2])
            result[-1] = result[-1].split(':')[0] + ':' + str(prev_len + length) + ':' + str(prev_score + score) 
        else:
            result.append(det)
    return result


os.system('mkdir -p results; rm -f results/*')
recog_files = sorted(glob.glob('../results/video_test*'))

# read label mapping
mapping = dict()
with open('mapping', 'r') as f:
    lines = f.read().split('\n')[0:-1]
    for line in lines:
        mapping[line.split()[0]] = line.split()[1]

# read fps for each video
fps = dict()
with open('fps.txt', 'r') as f:
    lines = f.read().split('\n')[0:-1]
    for line in lines:
        fps[line.split()[0]] = float(line.split()[1])

# process recognition files
result = []
for recog_file in recog_files:
    base_name = re.sub('\.mp4', '', re.sub('.*/', '', recog_file))
    with open(recog_file, 'r') as f:
        detections = f.read().split('\n')[-2]
        detections = detections.split()
        detections = smooth(detections) # uncomment if smoothing of short segments is desired
        offset = 0
        for detection in detections:
            (label, length, score) = tuple(detection.split(':'))
            length = int(length)
            start_time = offset / fps[base_name]
            end_time = (offset + length) / fps[base_name]
            score = math.exp(float(score) / length)
            if int(label) > 0:
                result += [ "%s %.1f %.1f %s %.7f" % (base_name, start_time, end_time, mapping[label], score) ]
            offset += length

# save detection file
with open('results/detections.txt', 'w') as f:
    f.write('\n'.join([line for line in result]) + '\n')
f.close()

# run matlab script for scoring
for overlap in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    matlab_cmd = "[pr_all,ap_all,map]=TH14evalDet('results/detections.txt','annotation','test'," + str(overlap) + "); \
                  disp(map); \
                  exit();"
    os.system('/usr/bin/matlab -nosplash -nodisplay -nojvm -r "' + matlab_cmd + '" > results/eval.overlap-' + str(overlap) + '.log')

