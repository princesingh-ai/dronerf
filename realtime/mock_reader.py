import logging
import time
import numpy as np

logger = logging.getLogger(__name__)


class MockHackRFLiveReader:
    """
    Temporary SDR simulator.
    Provides the same interface as HackRFLiveReader (start, read_chunk, stop).
    Generates synthetic complex IQ samples at the configured sample rate.
    
    IMPORTANT: The generated signal is NOT a real drone RF signal. 
    It is only for testing the realtime software pipeline.
    """

    def __init__(
        self,
        center_freq: float = 2.437e9,
        sample_rate: float = 20e6,
        chunk_size: int = 500_000,
        signal_type: str = "noise",
        realtime: bool = True,
    ):
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.signal_type = signal_type
        self.realtime = realtime

        self._streaming = False
        self._sample_index = 0

    def start(self) -> None:
        """Start the simulated SDR."""
        logger.info("Starting MOCK HackRF...")
        logger.info(f"Frequency: {self.center_freq / 1e6:.3f} MHz")
        logger.info(f"Sample rate: {self.sample_rate / 1e6:.3f} MSps")
        logger.info(f"Chunk size: {self.chunk_size} samples")
        logger.info(f"Signal type: {self.signal_type}")

        self._streaming = True
        self._sample_index = 0
        logger.info("MOCK HackRF initialized.")

    def read_chunk(self) -> np.ndarray:
        """Generate one chunk of complex64 IQ data."""
        if not self._streaming:
            raise RuntimeError("Mock HackRF is not running.")

        start = self._sample_index
        end = start + self.chunk_size

        if self.signal_type == "noise":
            iq = self._generate_noise()
        elif self.signal_type == "tone":
            iq = self._generate_tone(start, end)
        elif self.signal_type == "burst":
            iq = self._generate_burst(start, end)
        else:
            raise ValueError(f"Unknown mock signal type: {self.signal_type}")

        self._sample_index = end

        # Simulate real-time hardware latency
        if self.realtime:
            time.sleep(self.chunk_size / self.sample_rate)

        return iq

    def _generate_noise(self) -> np.ndarray:
        """Generate complex Gaussian noise."""
        i = np.random.randn(self.chunk_size).astype(np.float32) * 0.1
        q = np.random.randn(self.chunk_size).astype(np.float32) * 0.1
        return (i + 1j * q).astype(np.complex64)

    def _generate_tone(self, start: int, end: int) -> np.ndarray:
        """Generate a synthetic complex RF tone."""
        tone_frequency = 1e6  # 1 MHz baseband tone
        indices = np.arange(start, end, dtype=np.float64)
        phase = 2 * np.pi * tone_frequency * indices / self.sample_rate
        
        signal = 0.5 * np.exp(1j * phase)
        noise = (np.random.randn(self.chunk_size) + 1j * np.random.randn(self.chunk_size)) * 0.02

        return (signal + noise).astype(np.complex64)

    def _generate_burst(self, start: int, end: int) -> np.ndarray:
        """Generate intermittent RF-like bursts."""
        indices = np.arange(start, end, dtype=np.float64)

        # Base noise
        iq = (np.random.randn(self.chunk_size) + 1j * np.random.randn(self.chunk_size)) * 0.03

        # 2 MHz synthetic signal
        tone_frequency = 2e6
        phase = 2 * np.pi * tone_frequency * indices / self.sample_rate
        signal = 0.5 * np.exp(1j * phase)

        # 5 ms burst period, 1 ms burst length
        period = int(self.sample_rate * 0.005)
        burst_length = int(self.sample_rate * 0.001)

        positions = indices.astype(np.int64) % period
        active = positions < burst_length

        iq[active] += signal[active]
        return iq.astype(np.complex64)

    def stop(self) -> None:
        """Stop the simulated SDR."""
        if self._streaming:
            logger.info("Stopping MOCK HackRF...")
            self._streaming = False