from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING

try:
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS["tlc"] is True  # verify integration is enabled
    import tlc

    assert hasattr(tlc, "__version__")  # verify package is not directory

except (ImportError, AssertionError):
    tlc = None

