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

