This program is for the camera simulation using PPC.
The basic structure to get photon event is same with PPC.

- Structure
1. Photon event generator
    - Using PPC, sufficient number of photon simulation can be done
2. Project on the sensor of camera
    - Each event is occured in the DOM. This program will project photons to the camera

- Setup
1. Each event in the log is 32bytes
2. The maximum size of each log file is 256MB(pow(2, 23) event is written)
