```
$ roslaunch asv_system drl_collision.launch
$ roslaunch waypoint_generator drl_wpt.launch
```

### setup python environment
- conda create -n luman python=3.7
- conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
- pip install git+https://github.com/thu-ml/tianshou.git@master --upgrade
- **pip install 2to3 catkin_pkg pyyaml empy rospkg joblib mpi4py tabulate**
- **pip install tensorflow-gpu==1.14.0**
- **pip install --upgrade tensorlayer==1.11.0**
- **pip uninstall tensorboard-plugin-wit**

### run vo

roslaunch asv_system head_on.launch

