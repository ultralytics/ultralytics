import matplotlib.pyplot as plt

from ultralytics.utils import plt_settings


def test_backend_reset():
    @plt_settings(backend="Agg")
    def raise_error():
        raise RuntimeError

    backend = "ps"
    plt.switch_backend(backend)
    try:
        raise_error()
    except RuntimeError:
        assert plt.get_backend() == backend
    else:
        assert False, "Test function did not trigger correctly."
