# Project 3:

Grayson Gilbert, Marcus Hurt, Erebus Oh

ENPM 661, Spring 2025

March 28, 2025

[Github Link](https://github.com/ereoh/ENPM661-Project-3)

## Run This Code
---

### Dependenices
Install the necessary dependencies:
```
pip install -r requirements.txt
```
Or manually install the dependencies:
```
pip install numpy matplotlib tqdm
```

The exploration and path animatio will be output as `AStar_animation_start_to_goal.mp4` if ffmpeg is available (otherwise will be saved as `AStar_animation_start_to_goal.gif`).
- ffmpeg is significantly faster than pillow, to install:
    - Linux: `sudo apt update && sudo apt install ffmpeg`
    - [GeeksForGeeks: How to Install FFmpeg in Linux?](https://www.geeksforgeeks.org/how-to-install-ffmpeg-in-linux/)
    - [WikiHow: Easily Download and Install FFmpeg on a Windows PC](https://www.wikihow.com/Install-FFmpeg-on-Windows)

### Run Solver
To run the solver, just run the `phase1.py` file:
```bash
python phase1.py
```

