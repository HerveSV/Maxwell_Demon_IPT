import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience


import pims
import trackpy as tp
import av

from scipy.optimize import curve_fit

import time

import os


def find_complete_intervals(df, p_num):
    p_frames = df.loc[df['particle']==p_num, 'frame']
    p_intervals = [] # array of tuples containing start and end for each continous interval where particle is detected
    start = True
    first_start = True
    prev, curr = 0, 0
    for i in p_frames: # this actually serves as an index, i.e if p0 is not detected during frame 14, then there is no element 14, only an element 13 then 15
        if first_start: prev = i; first_start = False
        curr = i
        if curr-prev > 1:
            # frame jumping
            #print("frame skip: ",prev, curr)
            start = True

        if start:
            if len(p_intervals) != 0: p_intervals[-1][1] = prev
            p_intervals.append([curr, -69]) # error will occur if -69 not overwritten
        start = False
        prev = curr
    #print(p_intervals)
    if len(p_intervals) != 0: p_intervals[-1][1] = curr
    #else: print("Intervals too short", p_intervals)
    #print(p_intervals)
    return p_intervals

#tv.loc[(tv['particle']==0) & (tv['frame']==3), ['speed']] = 0.9 # assigning a specific frame  

def dist(y1, x1, y2, x2): # compute 2D euclidean norm:
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def compute_speeds(df, p_num, p_intervals) : # compute speeds of single particle over a complete interval
    short_count = 0
    p = df.loc[df['particle']==p_num]
    for i_start, i_end in p_intervals:
        if i_end-i_start == 0: 
            #print(f"Interval length too short: {i_end-i_start=}")
            df.loc[(df['particle']==p_num) & (tv['frame']==i_start), 'speed'] == None
            df.loc[(df['particle']==p_num) & (tv['frame']==i_start), 'vx'] == None
            df.loc[(df['particle']==p_num) & (tv['frame']==i_start), 'vy'] == None
            short_count += 1
            continue
        for i in range(i_start, i_end+1): # i is a frame_index
            if i == i_start: # start of the interval, differentiate over single time_step
                df.loc[(df['particle']==p_num) & (tv['frame']==i), 'speed'] = dist(p['y'][i], p['x'][i], p['y'][i+1], p['x'][i+1]) # px/frame
                df.loc[(df['particle']==p_num) & (tv['frame']==i), 'vx'] =  p.loc[i+1, 'x'] - p.loc[i, 'x'] # px/frame 
                df.loc[(df['particle']==p_num) & (tv['frame']==i), 'vy'] =  p.loc[i+1, 'y'] - p.loc[i, 'y'] # px/frame 
                #print(p0)
            elif i == i_end: # end of interval
                df.loc[(df['particle']==p_num) & (tv['frame']==i), 'speed'] = dist(p['y'][i-1], p['x'][i-1], p['y'][i], p['x'][i])
                df.loc[(df['particle']==p_num) & (tv['frame']==i), 'vx'] =  p.loc[i, 'x'] - p.loc[i-1, 'x'] # px/frame 
                df.loc[(df['particle']==p_num) & (tv['frame']==i), 'vy'] =  p.loc[i, 'y'] - p.loc[i-1, 'y'] # px/frame 
            else: # neither start nor beginner, we can average speed over 2 frames for more numerical stability
                df.loc[(df['particle']==p_num) & (tv['frame']==i), 'speed'] = dist(p['y'][i-1], p['x'][i-1], p['y'][i+1], p['x'][i+1])/2
                df.loc[(df['particle']==p_num) & (tv['frame']==i), 'vx'] =  (p.loc[i+1, 'x'] - p.loc[i-1, 'x'])/2 # px/frame 
                df.loc[(df['particle']==p_num) & (tv['frame']==i), 'vy'] = (p.loc[i+1, 'y'] - p.loc[i-1, 'y'])/2 # px/frame 
    return df, short_count

# video filepaths

filepaths = [
    "videos/1.75g_cropped/1.75g@15Hz.mov",
    "videos/1.75g_cropped/1.75g@18Hz.mov",
    "videos/1.75g_cropped/1.75g@20Hz.mov",
    "videos/1.75g_cropped/1.75g@22Hz.mov",
    "videos/1.75g_cropped/1.75g@24Hz.mov",
    "videos/1.75g_cropped/1.75g@26Hz.mov",
    "videos/1.75g_cropped/1.75g@28Hz.mov",
    "videos/1.75g_cropped/1.75g@30Hz.mov",
    "videos/1.75g_cropped/1.75g@32Hz.mov",
    "videos/1.75g_cropped/1.75g@34Hz.mov"
]


tp.quiet()  # Turn off progress reports for best performance

for fp in filepaths:
    PARAMS = {
    "diameter": 13,
    "threshold": 0,
    "minmass": 1000
    }
    print(f"Working on file {fp}")

    frames = pims.as_grey(pims.PyAVReaderTimed(fp))

    print("breakpoint 1")
    f = tp.batch(frames, diameter=PARAMS['diameter'], threshold=PARAMS['threshold'], minmass=PARAMS['minmass'])
    print("breakpoint 2")
    t = tp.link(f, 5, memory=3) # this tracks the location of each particle by establishing continuity from frame to frame

    t1 = tp.filter_stubs(t, 10)

    t2 = t1 # don't do any additional filtering, we can do this afterwards once we have csv - avoid throwing away potentially good data

    # skip the MSD step, we can do that from PSD

    # compute particle velocities
    tv = t2.copy()
    tv.insert(len(tv.columns), 'speed', 0.)
    tv.insert(len(tv.columns), 'vx', 0.)
    tv.insert(len(tv.columns), 'vy', 0.)
    particle_num = np.max(tv['particle'])

    short_count = 0
    for i in range(particle_num+1):
        p = tv[tv['particle']==i]
        p_intervals = find_complete_intervals(tv, p_num=i)
        #print(p_intervals)
        tv, sc = compute_speeds(tv, i, p_intervals)
        short_count += sc

    print(f"Number of short intervals without well defined velocity {short_count}")
    print(f"Particle number: {particle_num}")

    save_dir = "data/1.75g"
    save_name = os.path.splitext(os.path.basename(fp))[0]
    save_path = os.path.join(save_dir, save_name+".csv")
    print(save_path)
    tv.to_csv(save_path)