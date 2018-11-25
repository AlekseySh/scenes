from utils.common import Stopper


def test_stopper():
    values = [60, 63, 62, 64, 65, 64, 63, 63, 63, 63, 63]

    stopper = Stopper(n_observation=3, delta=2)
    for i, v in enumerate(values):
        stopper.update(v)

        if stopper.check_criterion():
            assert i == 8
            break


test_stopper()
