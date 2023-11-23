import sys

sys.path.append("../")
from build.pybind_interface.ReID import ReID
import cv2
import time


if __name__ == '__main__':
    iter_ = 10
    m = ReID(0)
    m.build("../build/sbs_R50-ibn.engine")
    print("build done")
    
    frame = cv2.imread("../data/Market-1501-v15.09.15/calib_set/-1_c1s2_009916_03.jpg")
    m.infer(frame)
    t0 = time.time()

    for i in range(iter_):
        m.infer(frame)

    total = time.time() - t0
    print("CPP API fps is {:.1f}, avg infer time is {:.2f}ms".format(iter_ / total, total / iter_ * 1000))