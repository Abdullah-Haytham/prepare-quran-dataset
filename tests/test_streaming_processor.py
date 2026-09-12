import numpy as np
import torch

from prepare_quran_dataset.modeling_fastconformer_cache_aware.processor import (
    FastConformerMelProcessor,
)


class StreamingProcessorMath:
    def __init__(
        self,
        hop=160,
        n_fft=512,
        pad=512,
        window=400,
        front_pad=120,
    ):
        """Configure the Streaming Processor parameters once; every method reuses them.

        Args:
            hop: STFT hop size in samples.
            n_fft: STFT FFT size.
            pad: zero-padding per chunk (must equal n_fft).
            window: Hann window size (center=True) in samples.
            front_pad: leading pad applied before chunking.
        """
        self.hop = hop
        self.n_fft = n_fft
        self.pad = pad
        self.window = window
        self.front_pad = front_pad
        self.check_front_pad()
        self.overlap = self._calc_overlap()
        self.drop_start = self._calc_drop_start()
        self.drop_end = self._calc_drop_end()

    def calc_samples_to_frames(self, L: int) -> int:
        """Number of frames given number of samples.

        Note: the moving window is n_fft not window, and window corresponds to
        center=True hann window of the n_fft window.
        """
        return (L + self.pad - self.n_fft) // self.hop + 1

    def check_front_pad(self):
        """Check front pad for center=True torch.stft."""
        assert self.pad == self.n_fft
        assert self.pad % 2 == 0
        assert (self.n_fft - self.window) % 2 == 0
        assert (
            self.pad // 2 + self.front_pad - (self.n_fft - self.window) // 2
        ) % self.hop == 0, "Invalid front_pad calculation"

    def check_chunk_size(self, chunk: int):
        """Check correct chunk and shift size.

        Args:
            chunk: chunk size in samples. Defaults to self.chunk.
        """
        assert self.pad == self.n_fft
        assert self.pad % 2 == 0
        assert self.window % 2 == 0
        assert (chunk - self.window // 2 + self.front_pad) % self.hop == 0, (
            "invalid chunk_size"
        )

    def _calc_overlap(self) -> int:
        """The overlap with the previous chunk"""
        return self.window - self.hop

    def _calc_drop_start(self) -> int:
        """Number of frames to deop from the begining of each processed chunk"""
        return (self.front_pad + self.window // 2) // self.hop

    def _calc_drop_end(self) -> int:
        """Number of frames to deop from the begining of each processed chunk"""
        return self.window // (self.hop * 2)

    def calc_chunk_from_frames(self, frames: int) -> int:
        """calculating chunks lenght in samples given the chunk in frames"""
        return (
            self.hop * (frames + self.drop_start + self.drop_end)
            - self.front_pad
            + self.window // 2
        )


def run_offline_processor(
    sp: StreamingProcessorMath,
    wave: torch.FloatTensor,
    processor: FastConformerMelProcessor,
):
    L = wave.shape[1] + sp.front_pad
    wave = torch.nn.functional.pad(wave, (sp.front_pad, 0, 0, 0))
    assert wave.shape[1] == L
    features, lens = processor(wave, torch.tensor([L] * wave.shape[0]))
    assert features.shape[2] == lens[0] == sp.calc_samples_to_frames(L), (
        "Can not predict len of offline processor"
    )
    wave = torch.nn.functional.pad(wave, (sp.pad // 2, sp.pad // 2, 0, 0))
    start = 0
    sample_windows_l = []
    for _ in range(sp.calc_samples_to_frames(L)):
        end = start + sp.n_fft
        sample_windows_l.append(wave[:, start:end])
        # print(wave[:, start:end].shape)
        start += sp.hop
    return features, torch.stack(sample_windows_l, dim=2)


def run_streaming_processor(
    sp: StreamingProcessorMath,
    wave: torch.FloatTensor,
    processor: FastConformerMelProcessor,
    chunk: int,
):
    sp.check_front_pad()
    sp.check_chunk_size(chunk)
    num_iter = (wave.shape[1] - chunk) // (chunk - sp.overlap) + 2
    feat_l = []
    windows_l = []
    start = 0
    end = 0
    for idx in range(num_iter):
        end = start + chunk
        input_wave = torch.nn.functional.pad(
            wave[:, start:end], (sp.front_pad, 0, 0, 0)
        )
        feats, _ = processor(
            input_wave, torch.tensor([chunk + sp.front_pad] * wave.shape[0])
        )

        debug_input_wave = torch.nn.functional.pad(
            input_wave, (sp.pad // 2, sp.pad // 2, 0, 0)
        )
        chunk_windows_l = []
        d_start = 0
        for _ in range(sp.calc_samples_to_frames(input_wave.shape[1])):
            d_end = d_start + sp.n_fft
            chunk_windows_l.append(debug_input_wave[:, d_start:d_end])
            d_start += sp.hop

        if idx == 0:
            # start (this is not meant for streaming but for comparison of the orignal offline processor
            feat_l.append(feats[:, :, : -sp.drop_end])
            chunk_windows_l = chunk_windows_l[: -sp.drop_end]
        elif idx == num_iter - 1:
            # end (not meant for streaming but to compare with the original offline processor)
            feat_l.append(feats[:, :, sp.drop_start :])
            chunk_windows_l = chunk_windows_l[sp.drop_start :]

        else:
            feat_l.append(feats[:, :, sp.drop_start : -sp.drop_end])
            chunk_windows_l = chunk_windows_l[sp.drop_start : -sp.drop_end]

        start = end - sp.overlap
        windows_l += chunk_windows_l
    return torch.cat(feat_l, dim=2), torch.stack(windows_l, dim=2)


def make_processor() -> FastConformerMelProcessor:
    """Build the processor without dither so results are deterministic.

    NeMo adds random noise (dither) per call while the module is in train mode,
    which makes offline vs streaming outputs differ by ~1e-4 everywhere.
    """
    # NOTE: we set dither to 0.0 for sake of coparing the two processor but leave as the default for actual run
    return FastConformerMelProcessor(dither=0.0, rng=np.random.default_rng(10))


if __name__ == "__main__":
    processor = make_processor()
    wave = torch.randn(1, 320000)
    sp = StreamingProcessorMath()
    print(f"drops: {sp.drop_start, sp.drop_end}")
    offline_feats, offline_windows = run_offline_processor(sp, wave, processor)
    chunk = sp.calc_chunk_from_frames(4 * 1)
    streaming_feats, streaming_windows = run_streaming_processor(
        sp,
        wave,
        processor,
        chunk=chunk,
    )
    print(f"offline sample windows: {offline_windows.shape}")
    print(f"offline sample windows: {streaming_windows.shape}")
    margin = 2001
    print(
        f"Both Samples windows match: {torch.allclose(offline_windows[:, 56:456, :margin], streaming_windows[:, 56:456, :margin])}"
    )
    print(f"Offline Len: {offline_feats.shape}")
    print(f"Streaming Len: {streaming_feats.shape}")
    print(f" Num of chunk frames: {sp.calc_samples_to_frames(sp.front_pad + chunk)}")
    print(
        f"Both processor are {torch.allclose(offline_feats[:, :, :margin], streaming_feats[:, :, :margin], atol=1e-6)}"
    )
    print(
        f"Both processor are {torch.allclose(offline_feats[:, :, :margin], streaming_feats[:, :, :margin])}"
    )
