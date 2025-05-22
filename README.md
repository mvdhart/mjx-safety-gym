# mjx-safety-gym
Open-source MJX implementation of OpenAI Safety Gym for accelerated safe reinforcement learning.

To run the interactive viewer on MacOS M1, run
```sudo mjpython scripts/interactive.py ```

Note that we need sudo privileges to get access to keypress information. 


## Madrona
To use vision-based observations, run 
```
chmod +x vision_setup.bash
./vision_setup.bash
```
This requires a linux machine with an NVidia GPU and can take up to 5 minutes to install.

