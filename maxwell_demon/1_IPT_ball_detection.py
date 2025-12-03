import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

import pandas as pd
import os

from scipy.optimize import curve_fit


THRESH = 145
MIN_AREA = 20

MAX_DIST = 14  # maximum allowed displacement per frame (tune this)

VERBOSE = True
PLOT = False
COMPUTE_MSD = False

'''
centroid {tuple}
(x, y, contour_area)
'''

'''
all_centroids {list{centroid}}
'''

def detect_centroids(frame_gray):
    '''
    frame_gray: grayscale image frame
    Returns list of tuples, each representing a particle (x, y, contour_area)
    '''
    # threshold bright balls
    _, bw = cv2.threshold(frame_gray, THRESH, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    # find blobs
    contours, _ = cv2.findContours(
        bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    centroids = []
    for c in contours:
        if cv2.contourArea(c) < MIN_AREA:
            continue

        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            centroids.append((float(cx), float(cy), cv2.contourArea(c)))

    return centroids

def match_points(prev_pts, curr_pts):
    """
    prev_pts: list of (x, y) from frame i
    curr_pts: list of (x, y) from frame i+1
    Returns list of (prev_index, curr_index) matches. i.e. it matches the same particle between frame i and i+1
    """

    if len(prev_pts) == 0 or len(curr_pts) == 0:
        return []

    prev_pts = np.array(prev_pts, float)
    curr_pts = np.array(curr_pts, float)

    matches = []
    used_curr = set()

    for i, p in enumerate(prev_pts):
        dists = np.linalg.norm(curr_pts - p, axis=1)
        j = np.argmin(dists)

        if dists[j] <= MAX_DIST and j not in used_curr:
            matches.append((i, j))
            used_curr.add(j)

    return matches

def build_clean_dataset(all_centroids, fps, mm_per_px):
    """
    all_centroids: list of lists
        all_centroids[f] = [(x, y, area), ...] for frame f
    fps: frames per second

    Returns:
        pandas DataFrame with columns:
        frame, particle_id, x, y, area, vx, vy, speed
    """

    # This will store every record
    rows = []

    # Particle tracks: particle_id -> last known (x, y)
    particle_last_pos = {}
    next_particle_id = 0

    # Particle assignment per frame: list of dicts
    # frame_assignments[f][j] = particle_id
    frame_assignments = []

    # ---- PASS 1: assign particle IDs for every frame ----
    for f in range(len(all_centroids)):
        curr_pts = all_centroids[f]
        curr_positions = [(p[0], p[1]) for p in curr_pts]

        frame_map = {}  # maps centroid index j → particle ID

        if f == 0:
            # First frame → every blob is new
            for j, p in enumerate(curr_positions):
                frame_map[j] = next_particle_id
                particle_last_pos[next_particle_id] = p
                next_particle_id += 1
        else:
            prev_pts = all_centroids[f - 1]
            prev_positions = [(p[0], p[1]) for p in prev_pts]

            matches = match_points(prev_positions, curr_positions)

            used_curr = set() # list of particle indices which have already been matched

            # Assign matched particles
            for (i, j) in matches:
                pid = frame_assignments[-1][i]   # same particle ID as in previous frame
                frame_map[j] = pid
                particle_last_pos[pid] = curr_positions[j]
                used_curr.add(j)

            # Unmatched = new particles
            for j, p in enumerate(curr_positions):
                if j not in used_curr:
                    frame_map[j] = next_particle_id
                    particle_last_pos[next_particle_id] = p
                    next_particle_id += 1

        frame_assignments.append(frame_map)

    # ---- PASS 2: compute vx, vy, speed ----
    for f in range(len(all_centroids)):
        for j, (x, y, area) in enumerate(all_centroids[f]):
            pid = frame_assignments[f][j]

            # Velocity = difference with previous frame
            if f == 0 or pid not in frame_assignments[f-1].values():
                vx = np.nan
                vy = np.nan
            else:
                # Find previous index of same particle
                prev_map = frame_assignments[f - 1]
                prev_idx = None
                for k, p_pid in prev_map.items():
                    if p_pid == pid:
                        prev_idx = k
                        break

                if prev_idx is None:
                    vx = vy = np.nan
                else:
                    x_prev, y_prev, _ = all_centroids[f - 1][prev_idx]
                    vx = (x - x_prev)
                    vy = (y - y_prev)

            speed = np.sqrt(vx * vx + vy * vy) if not np.isnan(vx) else np.nan

            rows.append({
                "frame": f,
                "particle_id": pid,
                "area": area,
                "x_px": x,
                "y_px": y,
                "x_mm": x*mm_per_px,
                "y_mm": y*mm_per_px,
                "vx_px/frame": vx,
                "vy_px/frame": vy,
                "speed_px/frame": speed,
                "vx_mm/s": vx*mm_per_px*fps,
                "vy_mm/s": vy*mm_per_px*fps,
                "speed_mm/s": speed*mm_per_px*fps
            })
    

    df = pd.DataFrame(rows)
    return df

# deprecated, slow
def compute_MSD(df, time_conversion=1, length2_conversion=1 ,max_lag=None):
    """
    Computes the Mean Squared Displacement for a tracked dataset.
    
    df: pandas DataFrame with columns ['frame', 'pid', 'x', 'y']
    max_lag: maximum time lag to compute (in frames). 
             Default: full possible range.
    
    Returns: DataFrame with columns ['lag', 'MSD']
    """

    # Ensure sorted by pid then frame
    df = df.sort_values(["pid", "frame"])

    # Determine maximum lag
    if max_lag is None:
        max_lag = df["frame"].max()

    # Store MSD values
    msd_vals = []

    # Loop over lag values
    for lag in range(1, max_lag + 1):
        displacements = []

        # For each particle
        for pid, track in df.groupby("pid"):

            frames = track["frame"].values
            xs = track["x"].values
            ys = track["y"].values

            # For each possible displacement
            for i in range(len(track) - lag):
                if frames[i+lag] == frames[i] + lag:
                    dx = xs[i+lag] - xs[i]
                    dy = ys[i+lag] - ys[i]
                    displacements.append(dx*dx + dy*dy)

        # Average over all particles
        if len(displacements) > 0:
            msd_vals.append(np.mean(displacements))
        else:
            msd_vals.append(np.nan)

    # Return results as a DataFrame
    lags = np.arange(1, max_lag+1)*time_conversion
    msd_vals = np.array(msd_vals)*length2_conversion
    return lags, msd_vals

def compute_velocities(centroids):
    vx = []
    vy = []

    for f in range(len(centroids) - 1):
        prev_pts = centroids[f]
        curr_pts = centroids[f + 1]

        matches = match_points(prev_pts, curr_pts)

        for (i, j) in matches:
            p_prev = np.array(prev_pts[i], float)
            p_curr = np.array(curr_pts[j], float)

            dx = p_curr[0] - p_prev[0]     # x-direction displacement
            dy = p_curr[1] - p_prev[1]
            vx.append(dx)
            vy.append(dy)     
    

    #print("what happens here")
    #print(vx)
    #print("stuff")
    #print(vy)
    return np.array(vx), np.array(vy)#[vx, vy]

def df_to_tensor(df):
    """
    Convert long-form tracking DataFrame into a tensor P × T × 2.
    df must contain columns: particle_id, frame, x_px, y_px
    Missing frames will be NaN.
    """
    pids = df["particle_id"].unique()
    frames = df["frame"].unique()

    P = len(pids)
    T = len(frames)

    positions = np.full((P, T, 2), np.nan, dtype=float)

    pid_to_idx = {pid: i for i, pid in enumerate(pids)}
    frame_to_idx = {frame: i for i, frame in enumerate(frames)}

    for _, row in df.iterrows():
        p = pid_to_idx[row['particle_id']]
        t = frame_to_idx[row['frame']]
        positions[p, t, 0] = row['x_px']
        positions[p, t, 1] = row['y_px']

    return positions

def compute_msd_fast(positions):
    """
    positions: array of shape (P, T, 2)
    returns: MSD for lags 1..T-1
    """
    P, T, _ = positions.shape
    msd = np.zeros(T)

    # For each lag τ, compute (r(t+τ) - r(t))² averaged over all pids and valid t
    for tau in range(1, T):
        diffs = positions[:, tau:, :] - positions[:, :-tau, :]
        squared = np.sum(diffs**2, axis=2)

        # ignore missing data (nan)
        msd[tau] = np.nanmean(squared)

    return msd
# --------------------------------------------

'''video_names = [
    "1.75g@15Hz.mov",
    "1.75g@18Hz.mov",
    "1.75g@20Hz.mov",
    "1.75g@22Hz.mov",
    "1.75g@24Hz.mov",
    "1.75g@26Hz.mov",
    "1.75g@28Hz.mov",
    "1.75g@30Hz.mov",
    "1.75g@32Hz.mov",
    "1.75g@34Hz.mov"
]'''
'''video_names= [
    "1.25g@15Hz.mov",
    "1.25g@20Hz.mov",
    "1.25g@22Hz.mov",
    "1.25g@24Hz.mov",
    "1.25g@26Hz.mov",
    "1.25g@28Hz.mov",
    "1.25g@30Hz.mov",
    "1.25g@32Hz.mov",
    "1.25g@34Hz.mov",
]'''
'''video_names = [
    "1.5g@15Hz.mov",
    "1.5g@18Hz.mov",
    "1.5g@20Hz.mov",
    "1.5g@22Hz.mov",
    "1.5g@24Hz.mov",
    "1.5g@26Hz.mov",
    "1.5g@28Hz.mov",
    "1.5g@30Hz.mov",
    "1.5g@32Hz.mov",
    "1.5g@34Hz.mov",
    "1.5g@36Hz.mov",
    "1.5g@38Hz.mov",
    "1.5g@40Hz.mov",
]'''



video_names = ["20balls, initially off, on at 2:.mov"]
video_dir = '/Users/hervesv/Documents/CloudDrive/IPT/Maxwell_Demon_2025/videos/Demon Operation (27.11.)/cropped'
#video_dir = "/Users/hervesv/Documents/CloudDrive/IPT/Maxwell_Demon_2025/videos/13-11-25/1.5g/cropped"


save_dir = "/Users/hervesv/Documents/CloudDrive/IPT/Maxwell_Demon_2025/data/27-11/"




mm_per_px = 350. / (1000-40) # mm / px, quite rough
fps = 60.
#fps = cap.get(cv2.CAP_PROP_FPS)


# --------- MAIN LOOP: store coordinates --------



for i in range(len(video_names)):
    #if i > 0: break
    video_path = os.path.join(video_dir, video_names[i])
    print(f"Current video {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file.")

    all_centroids = []   # list of list of centroids. Each element represents a frame

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        centroids = detect_centroids(gray)

        all_centroids.append(centroids)
        frame_idx += 1

    cap.release()


    if VERBOSE: print("Number of frames processed:", len(all_centroids))
    #print("Example entry (frame 0):", all_centroids[0])

    
    # -------- COMPUTE DATASET -----------

    print("Building dataset")
    df = build_clean_dataset(all_centroids, fps, mm_per_px)
    print("Dataset built dataset")



    # -------- IDENTIFY BAD PARTICLES --------------

    # particles which do not move
    '''epsilon_rms = 1.3 # px/frame
    rms_speed = df.groupby("particle_id")["speed_px/frame"].mean()
    bad_speed = rms_speed[rms_speed < epsilon_rms].index'''

    # particles which do not appear for many frames
    f_threshold = 10 # 10 frames minimum
    df_pid = df.groupby("particle_id")

    short_tracks = [] # keep pid of particles with frames under threshold
    for pid, track in df_pid:
        if len(track) < f_threshold:
            short_tracks.append(pid)

    print(f"Removed particles which appear for under {f_threshold} frames")

    # particles which have low rms for n_frames consecutive frames
    '''n_frames = 60
    stoppers = []

    for pid, track in df.groupby("particle_id"):
        T = len(track)
        if T < n_frames: continue
        for i in range(T-n_frames):
            subtrack = track[i:i+n_frames]
            sub_rms = subtrack['speed_px/frame'].mean()
            if sub_rms < epsilon_rms:
                stoppers.append(pid)
                break


    
    df_bad = df[df["particle_id"].isin(stoppers)]#df[(df["particle_id"].isin(bad_speed)) & (df["particle_id"].isin(short_tracks))]

    print(f"Bad particles ids: {df_bad["particle_id"]}")
    '''
    df_good = df[~df['particle_id'].isin(short_tracks)]

    # --------- SAVE FILE ----------------

    save_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(save_dir, save_name+".csv")

    #print(df)
    df_good.to_csv(save_path)
    print(f"File saved at {save_path}")

     # --------- COMPUTE MSD --------------
    if COMPUTE_MSD :
       

        print("Computing MSD")

        positions = df_to_tensor(df_good)
        msd_px2 = compute_msd_fast(positions)
        lag_frame = np.arange(0, len(msd_px2))
        
        lag_s = lag_frame/fps
        msd_mm2 = msd_px2 * mm_per_px**2

        df_msd = pd.DataFrame({"lag_frame": lag_frame,
                            "msd_px2": msd_px2,
                            "lag_s": lag_s,
                            "msd_mm2": msd_mm2})
        

        # ---------- SAVE MSD DATA ------------

        msd_path = os.path.join(save_dir, save_name+"_msd.csv")
        df_msd.to_csv(msd_path)
        print(f"MSD saved at {msd_path}")

        # ---------- SAVE MSD FIGURE -----------

        plt.clf()
        
        msd_fig_path = os.path.join(save_dir, save_name+"_msd.pdf")
        
        def lin_model(x, a):
            return a*x

        def quad_model(x, a):
            return a*x**2
        
        lag_nonan = np.array([lag_s[i] for i in range(len(lag_s)) if not np.isnan(msd_mm2[i])])
        msd_nonan = np.array([msd_mm2[i] for i in range(len(msd_mm2)) if not np.isnan(msd_mm2[i])])

        plt.scatter(lag_s, msd_mm2, s=5)

        # fit the front half of dataset
        lag_half1 = lag_nonan[:int(len(lag_nonan)*0.005)]
        msd_half1 = msd_nonan[:int(len(lag_nonan)*0.005)]
        popt1, pcov1 = curve_fit(quad_model, lag_half1, msd_half1)

        #print(popt1)
        plt.plot(lag_nonan[:int(len(lag_nonan)*0.05)], quad_model(lag_nonan[:int(len(lag_nonan)*0.05)], *popt1), label=r"~$\tau^2$")

        # fit the back half of dataset
        lag_half2 = lag_nonan[int(len(lag_nonan)*0.03):int(len(lag_nonan)*0.08)]
        msd_half2 = msd_nonan[int(len(lag_nonan)*0.03):int(len(lag_nonan)*0.08)]
        popt2, pcov2 = curve_fit(lin_model, lag_half2, msd_half2)
        #print(popt2)
        plt.plot(lag_nonan[int(len(lag_nonan)*0.005):], lin_model(lag_nonan[int(len(lag_nonan)*0.005):], *popt2), label=r"~$\tau$")
        
        plt.legend()
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r"Lag time $\tau$ [s]")
        plt.ylabel("MSD [mm$^2$]")
        plt.title(f"MSD {save_name}")
        plt.savefig(msd_fig_path)



    # -------- BUILD VIDEO TRACKING PARTICLE -------

    print("Drawing video")
    output_path = "videos/particle_tracking.mp4"
    if True:
        # =====================================================
        # =============== LOOP 2: DRAW + SAVE =================
        # =====================================================

        cap = cv2.VideoCapture(video_path)

        # Output video writer setup
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"{fps = }")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # get data for frame
            frame_data = df.loc[df['frame']==frame_idx]
            #frame_data = df_bad.loc[df_bad['frame']==frame_idx] # only label bad particles

            # draw circle which pid label on each particle
            for row_i in range(len(frame_data)):
                row = frame_data.iloc[row_i]
                cx, cy, pid = row['x_px'], row['y_px'], row['particle_id']
                #print(f"{cx = }")
                #print(f"{cy = }")
                color = (0,0,255)#(255, 0, 0) if (pid in bad_speed or pid in short_tracks) else (0, 0, 255) # change color if particle is bad

                cv2.circle(frame, (int(cx), int(cy)), 6, color, 2)


                # show particle_id
                pid_text = f"id={pid}"
                cv2.putText(frame, pid_text, (int(cx)+5, int(cy)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)



            

            # Write to output video
            out.write(frame)

            frame_idx += 1

        cap.release()
        out.release()



        print("Annotated video saved as:", output_path)


    # -------- COMPUTE VELOCITIES --------
    
    if False:
        vx_all, vy_all = compute_velocities(all_centroids)
        vx_t, vy_t = vx_all*fps, vy_all*fps


        if PLOT:
            plt.figure(figsize=(8,5))
            plt.hist(vx_t, bins=150)

            plt.title("Distribution of x-direction velocities")
            plt.xlabel("vx (pixels per second)")
            plt.ylabel("Count")


            plt.grid(True)
            plt.show()


            plt.figure(figsize=(8,5))
            plt.hist(vy_t, bins=150)

            plt.title("Distribution of x-direction velocities")
            plt.xlabel("vy (pixels per second)")
            plt.ylabel("Count")


            plt.grid(True)
            plt.show()



   





if False:   #Show video with detections
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not reopen video file.")

    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pts = all_centroids[frame_index]

        # Draw detected points
        for (x, y) in pts:
            cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), 2)

        # Draw velocities (starting from frame 1)
        if frame_index > 0:
            prev_pts = all_centroids[frame_index - 1]
            curr_pts = pts

            matches = match_points(prev_pts, curr_pts)

            for (i, j) in matches:
                x0, y0 = prev_pts[i]
                x1, y1 = curr_pts[j]

                dx = x1 - x0
                dy = y1 - y0

                # velocity vector arrow
                cv2.arrowedLine(
                    frame,
                    (int(x0), int(y0)),
                    (int(x1) + int((x1 - x0) * 3), int(y1) + int((y1 - y0) * 3)),
                    (0, 255, 0),
                    2,
                    tipLength=0.6
                )

                # show vx value
                vx_text = f"vx={dx:.2f}"
                cv2.putText(frame, vx_text, (int(x1)+5, int(y1)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)

        cv2.imshow("Detections + Velocities", frame)

        # Press Q to quit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_index += 1
        time.sleep(0.03)   # slow down playback slightly

    cap.release()
    cv2.destroyAllWindows()






if False:    #Obtain density heatmap
    GRID_SIZE = 6  # pixels

    # Determine frame size
    frame_h, frame_w = None, None
    for frame_pts in all_centroids:
        if len(frame_pts) > 0:
            frame_pts_array = np.array(frame_pts)
            break

    if frame_pts_array is not None:
        frame_h = int(np.max(frame_pts_array[:,1])) + GRID_SIZE
        frame_w = int(np.max(frame_pts_array[:,0])) + GRID_SIZE
    else:
        raise RuntimeError("No points detected, cannot determine frame size.")

    # Grid dimensions
    n_rows = int(np.ceil(frame_h / GRID_SIZE))
    n_cols = int(np.ceil(frame_w / GRID_SIZE))

    # Accumulate counts
    density_accum = np.zeros((n_rows, n_cols), dtype=float)
    for frame_pts in all_centroids:
        for (x, y) in frame_pts:
            row = int(y // GRID_SIZE)
            col = int(x // GRID_SIZE)
            if row < n_rows and col < n_cols:
                density_accum[row, col] += 1

    average_density = density_accum / len(all_centroids)

    # --- Normalize to Nth-highest value ---
    exclude_N = 20
    flattened = average_density.flatten()
    flattened_sorted = np.sort(flattened)[::-1]  # descending
    if len(flattened_sorted) >= exclude_N:
        vmax = flattened_sorted[exclude_N-1]  # 5th-highest
    else:
        vmax = flattened_sorted[0]  # if fewer than 5 cells
    vmin = 0

    # Plot heatmap
    plt.figure(figsize=(10,6))
    plt.imshow(
        average_density,
        origin='upper',         # flip y-axis: 0 at top
        cmap='hot',
        interpolation='nearest',
        extent=[0, frame_w, 0, frame_h],
        vmin=0,
        vmax=vmax
    )
    plt.colorbar(label="Average number of balls per cell")
    plt.title(f"Average Ball Density Heatmap (normalized to {exclude_N}th-highest, y flipped)")
    plt.xlabel("X pixels")
    plt.ylabel("Y pixels")
    plt.show()
