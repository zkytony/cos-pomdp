![img](https://user-images.githubusercontent.com/7720184/167312293-6a747b75-497b-406a-b4c3-231ed18d64e4.png)

# cos-pomdp

This is the repository for "[Towards Optimal Correlational Object Search](https://arxiv.org/pdf/2110.09991.pdf)" (ICRA 2022).

**Abstract:**
In realistic applications of **object search**, robots will need to locate target objects in complex environments while coping with **unreliable sensors**, especially for **small or hard-todetect objects**. In such settings, **correlational information** can be valuable for planning efficiently. Previous approaches that consider correlational information typically resort to ad-hoc, greedy search strategies. We introduce the **Correlational Object Search POMDP (COS-POMDP)**, which models correlations while preserving optimal solutions with a reduced state space. We propose **a hierarchical planning algorithm** to scale up COS-POMDPs for practical domains. Our evaluation, conducted with the AI2-THOR household simulator and the YOLOv5 object detector, shows that our method finds objects more successfully and efficiently compared to baselines, particularly for hard-to-detect objects such as srub brush and remote control.



![cospomdp-example-creditcard](https://user-images.githubusercontent.com/7720184/167313059-34f03724-f9b4-49ed-bd99-bfbf134abfb8.gif)


* Paper on arxiv: https://arxiv.org/pdf/2110.09991.pdf
* Presentation video on Youtube: https://www.youtube.com/watch?v=-eehMN6sod8
* More sim demos: https://www.youtube.com/watch?v=RneTq4o0a-A
* Robot demo (on Boston Dynamics Spot): https://www.youtube.com/watch?v=9tCnRTZa9C4&t=7s


## Organization
Contains two packages: [cospomdp](https://github.com/zkytony/cos-pomdp/tree/master/cospomdp) and [cospomdp_apps](https://github.com/zkytony/cos-pomdp/tree/master/cospomdp_apps).
The former defines the COS-POMDP (domain, models, agent)
and the latter applies COS-POMDP to specific applications
such as AI2-THOR, along with [agents](https://github.com/zkytony/cos-pomdp/tree/master/cospomdp_apps/thor/agent) that can perform object search
in the specific application domain.

## Citation
```
@inproceedings{zheng2022towards,
  title={Towards Optimal Correlational Object Search,
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  author={Zheng, Kaiyu and Chitnis, Rohan and Sung, Yoonchang and Konidaris, George and Tellex, Stefanie},
  year={2022}
}
```

## Installation and Setup

### Requirements
* Ubuntu 18.04+
* Python 3.8+

If your system does not meet these requirements, you can use Docker. Although we do not provide
currently a Dockerfile (help would be appreciated), you could build upon the [ubuntu:20.04](https://hub.docker.com/layers/ubuntu/library/ubuntu/20.04/images/sha256-7c9c7fed23def3653a0da5bc9ecb651efe155ebd5802c7ba5d585edaa6c89496?context=explore) image (recommended)
or [ai2thor-docker](https://github.com/allenai/ai2thor-docker) (I have not tried),
and then do the following instructions inside it. You can set up X11 forwarding to enable GUI inside docker.

_Note on dependencies:_ `requirements_snapshot.txt` contains a dump of the installed pip packages
at the time when the experiments were conducted. You don't have to install dependencies through
this file. Just follow the steps below. COS-POMDP should still work with
the latest versions of those packages that contain security vulnerability fixes (e.g. for [Pillow](https://snyk.io/vuln/pip:pillow)),
with minor changes when interfacing with those packages, as long as the functionalities of those packages are preserved.

Some dependencies:

* [pomdp_py](https://github.com/h2r/pomdp-py): A framework to build and solve POMDP problems.
* [thortils](https://github.com/zkytony/thortils): Utility functions when working with Ai2-THOR (v3.3.4)
* [sciex](https://github.com/zkytony/sciex): A framework for scientific experiments

### Setup Repo
Clone the repo:
```
git clone git@github.com:zkytony/cos-pomdp.git
```
After cloning do the following three commands separately.
```
source setup.bash
source setup.bash -I
source setup.bash -s
```

It is recommended to use a virtualenv with Python3.8+

To test if it is working: Do `pip install pytest` and then go to `tests/` folder and run
```
pytest
```
You are expected to see something like:
```
-- Docs: https://docs.pytest.org/en/stable/warnings.html
======= 14 passed, 2 skipped, 3 warnings in 43.71s ======
```

At this point, you should be able to run a basic object search domain with COS-POMDP.
```
python -m cospomdp_apps.basic.search
```
A pygame window will be displayed

### To Run in Ai2Thor
(Skip this if you are running on a computer connected to a display) If you are running offline, or on a server, make sure there is an x server running.
You can do this by:
1. Creating the [`xorg_conf`](https://www.x.org/releases/current/doc/man/man5/xorg.conf.5.xhtml) file. (Check [this out](https://github.com/allenai/ai2thor/issues/886))
2. Then run:
```
sudo Xorg -noreset +extension GLX +extension RANDR +extension RENDER -config xorg_conf :0
```
Note that depending on your local configuration you may want to use something other than `:0`. If there is already an x server running at `:0` but your `$DISPLAY` shows nothing, you should run it at another display number.

Now, download necessary models by running the following script. This will download three files: `yolov5-training-data.zip`, `yolov5-detectors.zip`, `corrs.zip`, place them in the desired location and decompress them.
```
# run at repository root
python download.py
```

Then, main test we will run is
```
cd tests/thor
python test_vision_detector_search.py
```
This will run a search trial for AlarmClock in a bedroom scene. When everything works, you may see something like this:

<img src="https://user-images.githubusercontent.com/7720184/155869506-d7d1b8df-cb2b-43b9-9ca2-8da31ce6d9eb.png" width="800px">

Note that the search process may vary due to random sampling during planning.



### Caveats
The external methods, e.g. SAVN, MJOLNIR, are placed under `cos-pomdp/external`.
However, for importability, a symbolic link to the `cos-pomdp/external/mjolnir`
directory is created under `cos-pomdp/cospomdp_apps/thor/mjolnir`. Please
make sure that link points to the right absolute path on your computer.
For example, you can directly create a new one by:
```
cd repo/cos-pomdp/cospomdp_apps/thor
ln -sf $HOME/repo/cos-pomdp/external/mjolnir/ mjolnir
```

**Note**: setup for SAVN, MJOLNIR etc. were attempted during the project; MJOLNIR can run but does not work well.


## Experiment Results
You can download experiment results (individual trials) from this Google Drive file link: [cospomdp-results-final.zip](https://drive.google.com/file/d/1DJFZLf2QTnQ-NE2IVbdClvQeP-jjwT3s) (45.1MB)
Place this file under the `results/` folder and decompress it.
You can gather statistics by running:
```
cd cospomdp-results-final
python gather_results.py
```
Refer to the [sciex](https://github.com/zkytony/sciex) package for more information on the experiment framework.

You can replay a trial by:
```
python -m cospomdp_apps.thor.replay <trial_name>
```
The `<trial_name>` is the name of a directory inside `cospomdp-results-final`, for example `bathroom-FloorPlan421-Candle_000_random#gt`.
Note that the dynamically generated topological graph was not saved therefore is not visualized.


## Appendix: AI2-Thor Constants Configuration

Compare with:
- [1] IQA (CVPR'18)
- [2] Visual Semantic Navigation Using Scene Priors (ICRL'19)
- [3] Learning hierarchical relationships for object-goal navigation (CoRL'20)
- [4] Hierarchical and Partially Observable Goal-driven Policy Learning with Goals
  Relational Graph (CVPR'21)


|                  | grid size | h_rotate | v_rotate | fov |
|------------------|-----------|----------|----------|-----|
| [1] IQA          | 0.25      | 90       | 30       | 60  |
| [2] Scene Priors | 0.25      | 45       | 30       | 90  |
| [3] MJONIR       | 0.25      | 45       | 30       | 100 |
| [4] HRL-GRG      | 0.25      | 90       | 30       | 90  |
| [5] ALFRED       | 0.25      | 90       | 15       | 90  |
| ours             | 0.25      | 45       | 30       | 90  |

Constants can be found in `cospomdp_apps/thor/constants.py`.
