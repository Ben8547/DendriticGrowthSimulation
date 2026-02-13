from numpy import array,hstack,vstack
from launchSimulation import run


gen_matrix = True
gen_sims = True

matrix_Ham = array([float(2**(2*i)) for i in range (3)],dtype=str) # 3
matrix_viscMult = array([1., 1.5 , 2.],dtype=str) # 3

if gen_sims:
    for i in range(3):
        with open("LJ_sigma.txt",'w') as Ham: # Change paramaters
                Ham.write(matrix_Ham[i])
        for j in range(3):
            with open("Visc.txt",'w') as Visc: # Change paramaters
                Visc.write(matrix_viscMult[j])
            # Run simulation
            run(float(matrix_Ham[i]),float(matrix_viscMult[j]))

# Now extract the last frame and stich into one PNG file in a matrix
if gen_matrix:
    import cv2 # image manipultaion stuff

    def last_frame_from_video(path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # get total frame count in video
        if total_frames < 1:
            raise RuntimeError(f"No frames in {path}") # had some issues with this running on incomplete files - these alerts the issue

        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Failed to read last frame from {path}") # extra failsafe

        return frame

    # create matrix image

    from defineParameters import params
    #from os.path import exists,abspath # debug

    frames = []

    for i in range(3):
        for j in range(3):
            fname = f"dendrite_growth_simulation-pulses-newcurrent-{params['num_e']}_electrons-visc-{matrix_viscMult[j]}_vdW-{matrix_Ham[i]}_Ly-{params["L_y"]}-{params['n']}particles.mp4"
            #fname = r"\\?\\" + abspath(fname) # the filenames are long so append the windows long filename prefix; preventative measure
            #print(exists(fname)) # debug
            frames.append(last_frame_from_video(fname)) # get the last frame of the current file - add to list
    # normalize the size of the frame - should already be the same size; but for quality assurence...
    h, w, _ = frames[0].shape
    frames = [cv2.resize(f, (w, h)) for f in frames]
    #build 3x3 grid
    row1 = hstack(frames[0:3])
    row2 = hstack(frames[3:6])
    row3 = hstack(frames[6:9])
    grid = vstack([row1, row2, row3])

    cv2.imwrite("Matrix-Image.png", grid) # save the new image

    print("Saved the matrix")