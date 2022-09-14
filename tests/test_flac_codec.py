from flac_numcodecs import Flac
import numpy as np
import zarr
import pytest

DEBUG = False

# dtypes = ["int8", "int16", "int32", "float32"]
dtypes = ["int16"]

def run_all_options(data):
    dtype = data.dtype
    for level in range(1, 9):
        for bs in [None, 100, 1000]:
            print(f"Dtype {dtype} - level {level} - blocksize {bs}")
            cod = Flac(level=level)
            enc = cod.encode(data)
            dec = cod.decode(enc)

            assert len(enc) < len(dec)
            print("CR", len(dec) / len(enc))
            data_dec = np.frombuffer(dec, dtype=dtype).reshape(data.shape)
            assert np.all(data_dec == data)
        

def make_noisy_sin_signals(shape=(30000,), sin_f=100, sin_amp=50, noise_amp=5,
                           sample_rate=30000, dtype="int16"):
    assert isinstance(shape, tuple)
    assert len(shape) <= 3
    if len(shape) == 1:
        y = np.sin(2 * np.pi * sin_f * np.arange(shape[0]) / sample_rate) * sin_amp
        y = y + np.random.randn(shape[0]) * noise_amp
        y = y.astype(dtype)
    elif len(shape) == 2:
        nsamples, nchannels = shape
        y = np.zeros(shape, dtype=dtype)
        for ch in range(nchannels):
            y[:, ch] = make_noisy_sin_signals((nsamples,), sin_f, sin_amp, noise_amp,
                                              sample_rate, dtype)
    else:
        nsamples, nchannels1, nchannels2 = shape
        y = np.zeros(shape, dtype=dtype)
        for ch1 in range(nchannels1):
            for ch2 in range(nchannels2):
                y[:, ch1, ch2] = make_noisy_sin_signals((nsamples,), sin_f, sin_amp, noise_amp,
                                                        sample_rate, dtype)
    return y


def generate_test_signals(dtype):
    test1d = make_noisy_sin_signals(shape=(3000,), dtype=dtype)
    test1d_long = make_noisy_sin_signals(shape=(200000,), dtype=dtype)
    test2d = make_noisy_sin_signals(shape=(3000, 10), dtype=dtype)
    test2d_long = make_noisy_sin_signals(shape=(200000, 20), dtype=dtype)
    test2d_extra = make_noisy_sin_signals(shape=(3000, 300), dtype=dtype)
    test3d = make_noisy_sin_signals(shape=(1000, 5, 5), dtype=dtype)

    return [test1d, test1d_long, test2d, test2d_long, test2d_extra, test3d]

@pytest.mark.numcodecs
def test_flac_numcodecs():
    for dtype in dtypes:
        print(f"\n\nNUMCODECS: testing dtype {dtype}\n\n")

        test_signals = generate_test_signals(dtype)

        for test_sig in test_signals:
            print(f"signal shape: {test_sig.shape}")
            run_all_options(test_sig)

@pytest.mark.zarr
def test_flac_zarr():
    for dtype in dtypes:
        print(f"\n\nZARR: testing dtype {dtype}\n\n")
        test_signals = generate_test_signals(dtype)

        compressor = Flac()

        for test_sig in test_signals:
            print(f"signal shape: {test_sig.shape}")
            if test_sig.ndim == 1:
                z = zarr.array(test_sig, chunks=None, compressor=compressor)
                assert z[:].shape == test_sig.shape
                assert z[:100].shape == test_sig[:100].shape
                assert z.nbytes > z.nbytes_stored

                z = zarr.array(test_sig, chunks=(1000), compressor=compressor)
                assert z[:].shape == test_sig.shape
                assert z[:100].shape == test_sig[:100].shape

            elif test_sig.ndim == 2:
                z = zarr.array(test_sig, chunks=None, compressor=compressor)
                assert z[:].shape == test_sig.shape
                assert z[:100, :10].shape == test_sig[:100, :10].shape
                assert z.nbytes > z.nbytes_stored

                z = zarr.array(test_sig, chunks=(1000, None), compressor=compressor)
                assert z[:].shape == test_sig.shape
                assert z[:100, :10].shape == test_sig[:100, :10].shape

                z = zarr.array(test_sig, chunks=(None, 10), compressor=compressor)
                assert z[:].shape == test_sig.shape
                assert z[:100, :10].shape == test_sig[:100, :10].shape

            else: # 3d
                z = zarr.array(test_sig, chunks=None, compressor=compressor)
                assert z[:].shape == test_sig.shape
                assert z[:100, :2, :2].shape == test_sig[:100, :2, :2].shape
                assert z.nbytes > z.nbytes_stored

                z = zarr.array(test_sig, chunks=(1000, 2, None), compressor=compressor)
                assert z[:].shape == test_sig.shape
                assert z[:100, :2, :2].shape == test_sig[:100, :2, :2].shape

                z = zarr.array(test_sig, chunks=(None, 2, 3), compressor=compressor)
                assert z[:].shape == test_sig.shape
                assert z[:100, :2, :2].shape == test_sig[:100, :2, :2].shape



if __name__ == '__main__':
    test_flac_numcodecs()
    test_flac_zarr()
