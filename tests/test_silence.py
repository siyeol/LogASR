import numpy as np

from log_asr.silence import package_silence


def test_long_silence_keeps_first_frame_only():
    frame = 1280
    speech = np.full(frame, 0.5, dtype=np.float32)
    silence = np.zeros(frame, dtype=np.float32)
    waveform = np.concatenate([speech, silence, silence, silence, speech])
    output, stats = package_silence(waveform)
    assert stats.frames_in == 5
    assert stats.frames_out == 3
    assert stats.silence_frames_in == 3
    assert stats.silence_frames_out == 1
    np.testing.assert_array_equal(output[frame : 2 * frame], silence)


def test_short_silence_is_not_collapsed():
    frame = 1280
    waveform = np.concatenate(
        [np.full(frame, 0.5, dtype=np.float32), np.zeros(frame, dtype=np.float32)]
    )
    output, stats = package_silence(waveform, min_silence_frames=2)
    assert stats.frames_out == 2
    assert output.size == waveform.size

