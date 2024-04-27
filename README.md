# rigid_body_xl
Test project to get familiar with mujoco

## Requirements installation
```
pip install -r requirements.txt
```
In addition to the above, you may need to run the following to avoid [this error which would occur when creating cv2 videowriter](https://stackoverflow.com/questions/72540359/glibcxx-3-4-30-not-found-for-librosa-in-conda-virtual-environment-after-tryin)
```
conda install -c conda-forge gcc=12.1.0
```

Just memo:
```
parent's body id of:
      worldbody (body_id==0): 0,
          link1 (body_id==1): 0,
          link2 (body_id==2): 1,
          link3 (body_id==3): 2,
          link4 (body_id==4): 3,
          link5 (body_id==5): 4,
          link6 (body_id==6): 5,
 target/ = worldbody in
     object.xml (body_id==7): 6,
  target/object (body_id==8): 7,
```
