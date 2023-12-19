# GPU Implementation of Scanline Stereo With Application to Stereoscopic Video Enhancement

There are several dependencies, including scikit-image, opencv (for rectification), Numpy, Matplotlib, imageio, and ffmpeg.

To compile, gcc and nvcc (only if you're using the GPU) are required. 

To run the demo application, go to the root of the project and use the command

```
make run
```

If you want to ensure that the gpu results line up exactly with the cpu results, use

```
make test
```

The gallery.py, performance.py, plots.py, roofline.py files are all used to generate charts for the final report.

