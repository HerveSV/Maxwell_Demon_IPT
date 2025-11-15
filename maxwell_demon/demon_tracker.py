import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience

import time
import sys
import getopt
import os

import pims
import trackpy as tp
import av

from scipy.optimize import curve_fit


def track_particles(file_path, params, logging):
    frames = pims.as_grey(pims.PyAVReaderTimed(file_path))

    if not logging: tp.quiet()

    if logging:
        plt.imshow(frames[0])
        plt.title("Frame 0")
        plt.show()
    

    # locate bright features on all frames
    f = tp.batch(frames, 
                    diameter=params['diameter'],
                    threshold=params['threshold'],
                    minmass=params['minmass'])
    
    # track location of particles across frames
    t = tp.link(f, 5, memory=params['memory'])

    # filter out particles which appear for too few frames
    t1 = tp.filter_stubs(t, threshold=params['minframes'])

    tv = t1.copy()
    tv.insert(len(tv.columns), 'speed', 0.)
    tv.insert(len(tv.columns), 'vx', 0.)
    tv.insert(len(tv.columns), 'vy', 0.)

    return tv




def main():



    logging = True
    tracked = False

    tracking_params = {
        "diameter":  13,    # tp.locate(diameter)
        "threshold": 0,     # tp.locate(threshold)
        "minmass": 3000,    # tp.locate(minmass)
        "memory": 3,        # tp.link(memory)
        "minframes": 3      # tp.filter_stubs(threshold)
    }

    # video data
    video_filepath = ""
    video_fps = 30
    video_dist_calib = 1. # mm/px
    

    # tracking data
    tracking_df = None
    ensemble_MSD = None


    argv = sys.argv[1:]
    try:
        options, args = getopt.getopt(argv, "w:r:N:l:p:h:s:Q:",
                                ["w =",
                    "r =",
                    "N =",
                    "l =",
                    "p =",
                    "s =",
                    "Q ="])
    except Exception as e:
        print("Error Message: " + str(e))

    for name, value in options:
        if name in ['-v']:
            video_filepath = "videos/"+value
        


    # welcome message
    print("Starting Tracker:")
    print(f"<dmontrakr>: Current video path: {"None" if video_filepath=='' else video_filepath}")

    while (True):
        user_input = input("<dmontrakr>: ")

        if user_input == 'quit' or user_input ==  'q':
            break

        elif user_input == 'track' or user_input == 't':
            print("<dmontrakr>:")
            # track and save particle positions
            print("<dmontrakr>: Tracking particles")
            #time1 = time.time()
            tracking_df = track_particles(video_filepath, tracking_params, logging)
            #time2 = time.time()
            #print(f"<dmontrakr>: Finished tracking, time elapsed {time2-time1} s")
            ensemble_MSD = tp.emsd(tracking_df, video_dist_calib, video_fps)

            if logging:
                _t2 = lambda x, a: a*x**2
                _t = lambda x, a: a*x

                fig, ax = plt.subplots()
                ax.plot(ensemble_MSD.index, ensemble_MSD, 'o', markersize=4)
                ax.plot(ensemble_MSD.index[:10], _t2(ensemble_MSD.index[:10], 5000), label=f"~$t^2$ Ballistic motion", c='green') # Ballistic motion at short timescales
                ax.plot(ensemble_MSD.index[3:], _t(ensemble_MSD.index[3:], 1.1e3), label=f"~$t$ Brownian motion", c='red') # Brownian motion at long timescales
                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set(ylabel=r'$\langle \Delta r^2 \rangle$ [px$^2$]',
                    xlabel='lag time $t$')
                ax.set_title("Ensemble average MSD, 1.5g@35Hz")
                ax.legend()
                #plt.savefig("figs/ensemble_MSD_1.5g@35Hz.png", dpi=300)
                plt.show()

            
        
        elif user_input == 'set_vid' or user_input == 'sv':
            video_filepath = "videos/"+input("<dmontrakr>: video filepath = ")

        elif user_input == 'set_fps':
            video_fps = int(input("<dmontrakr>: video FPS = "))
            tracked = False


        elif user_input == 'help' or user_input == 'h':
            print("<dmontrakr>:")
            print("<dmontrakr>:   Maxwell's Demon Particle Tracker")
            print("<dmontrakr>:   help/h:     help menu")
            print("<dmontrakr>:   track/t:    run tracker")
            print("<dmontrakr>:   set/s:      set video to track")
            print("<dmontrakr>:   config/c:   show the config")
            print("<dmontrakr>:   quit/q:     quit the tracker")

        else:
            print("<sim>:")
            print("<sim>:   Type 'help' for help")
    


if __name__ == "__main__":
    main()