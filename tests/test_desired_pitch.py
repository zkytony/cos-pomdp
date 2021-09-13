from cospomdp.models.sensors import pitch_facing
import random
from tqdm import tqdm

def _test():

    for i in tqdm(range(100)):
        r = (random.randint(0, 8), random.randint(0, 8), random.randint(0, 8))
        t = (random.randint(0, 8), random.randint(0, 8), random.randint(0, 8))
        pitch = pitch_facing(r, t)
        try:
            if t[2] < r[2]:
                assert pitch > 0
            else:
                assert pitch < 0
        except:
            import pdb; pdb.set_trace()


if __name__ == "__main__":
    _test()
