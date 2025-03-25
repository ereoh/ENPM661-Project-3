# Project 3:

Grayson Gilbert, Marcus Hurt, Erebus Oh

ENPM 661, Spring 2025

March 28, 2025

[Github Link](https://github.com/ereoh/ENPM661-Project-3)

## Deliverables
- [ ] `Proj3_grayson_marcus_erebus.zip`
    - [ ] `README.md`
        - how run code and give inputs
        - mention libraries/dependencies used
        - team members + UID
        - Github repo link
    - [ ] `a_star_grayson_marcus_erebus.py`
        - Github repo link
    - [ ] animation video
        - exploration and optimal path
- [ ] `Proj3_grayson_marcus_erebus.pdf`
    - soure code from `a_star_grayson_marcus_erebus.py`
    - for plagarism checks
    - outside zip

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

## Requirements
- Givens
    - radius: 5mm?
    - clearance: 5mm?
    - Map: 250x600, same as project 2
        - use half planes, semi-algebriac models
    - Goal threshold: 1.5mm radius
    - State threshold: 0.5 xy and 30 degrees
    - Action Space
        - 
- User Input
    - Start: x y theta
    - Goal: x y theta
    - clearance?
    - robot radius?
    - step size of robot: magnitude of robot action movement
- Use A-Star Search
    - 
    - forward search
    - heuristic = Euclidean distance
    - search space tree for path
        - generate as 3D space
        - matrix to store visited nodes
            - 250/threshold x 600/threshold x (360/30) = 500x1200x12
            - 1 for visted
            - empty or zero otherwise
- Output animation video
    - exploration
    - optimal path