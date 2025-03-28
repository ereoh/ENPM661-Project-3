# Project 3:

Grayson Gilbert
UID: 115837461
Directory ID: ggilbert

Marcus Hurt
UID: 121361738
Directory ID: mhurt

Erebus Oh
UID: 117238105
Directory: eoh12

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
    - outside zip file

## Run This Code
---

### Dependenices
Dependencies used in this project include: numpy, pygame, heapq, math, and time.

Install the necessary dependencies:
```
pip install -r requirements.txt
```
Or manually install the dependencies:
```
pip install numpy pygame
```

### Run Solver
To run the solver, just run the `a_star_grayson_marcus_erebus.py` file:
```bash
python3 a_star_grayson_marcus_erebus.py
```
Then, enter in start position, start theta, goal position, and step amount following the format as prompted.

## Requirements
- Givens
    - radius: 5mm?
    - clearance: 5mm?
    - Map: 250x600, same as project 2
        - use half planes, semi-algebriac models
    - Goal threshold: robot center within 1.5 unit radius (any orientation)
    - State threshold: 0.5 xy and 30 degrees
    - Action Space
- User Input
    - Start: x y theta
    - Goal: x y theta
    - step size of robot: magnitude of robot action movement
## Use A-Star Search
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