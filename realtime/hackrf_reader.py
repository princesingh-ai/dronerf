import logging
import numpy as np
import SoapySDR
from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX

logger = logging.getLogger(__name__)

class HackRFLiveReader:
    """
    Live IQ data provider for HackRF One.
    Optimized to read directly into pre-allocated memory.
    """

    def __init__(
        self,
        center_freq: float = 2.437e9,
        sample_rate: float = 20e6,
        chunk_size: int = 500_000,
        lna_gain: int = 32,
        vga_gain: int = 30,
        amp_enable: bool = True,
    ):
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.lna_gain = lna_gain
        self.vga_gain = vga_gain
        self.amp_enable = amp_enable

        self.sdr = None
        self.rx_stream = None
        self._streaming = False
        
        # Optimization: Pre-allocate memory once to avoid slowing down the read loop
        self._chunk_buffer = np.empty(self.chunk_size, dtype=np.complex64)

    def start(self) -> None:
        """Connect to HackRF and start the RX stream."""
        logger.info("Connecting to HackRF One...")

        try:
            self.sdr = SoapySDR.Device({"driver": "hackrf"})
            self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.sample_rate)
            self.sdr.setFrequency(SOAPY_SDR_RX, 0, self.center_freq)
            
            self.configure_gain()

            self.rx_stream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
            self.sdr.activateStream(self.rx_stream)
            self._streaming = True

            logger.info("HackRF initialized successfully.")
            logger.info(f"Frequency: {self.center_freq / 1e6:.3f} MHz | Rate: {self.sample_rate / 1e6:.3f} MSps")
            
        except Exception as exc:
            self.stop()
            raise RuntimeError(f"Failed to initialize HackRF: {exc}") from exc

    def configure_gain(self) -> None:
        """Configure HackRF amplifier, LNA, and VGA gains."""
        gains = [
            ("AMP", 14 if self.amp_enable else 0),
            ("LNA", self.lna_gain),
            ("VGA", self.vga_gain)
        ]
        
        for name, value in gains:
            try:
                self.sdr.setGain(SOAPY_SDR_RX, 0, name, value)
            except Exception as exc:
                logger.warning(f"Could not configure {name} gain: {exc}")

    def read_chunk(self) -> np.ndarray:
        """Efficiently read samples directly into a pre-allocated buffer."""
        if not self._streaming:
            raise RuntimeError("HackRF stream is not active.")

        samples_received = 0

        while samples_received < self.chunk_size:
            samples_remaining = self.chunk_size - samples_received
            
            # Optimization: Create a view of the remaining space. 
            # SoapySDR writes directly here, eliminating intermediate buffer copies.
            target_slice = self._chunk_buffer[samples_received:self.chunk_size]
            
            result = self.sdr.readStream(
                self.rx_stream, [target_slice], samples_remaining, timeoutUs=100_000
            )

            if result.ret > 0:
                samples_received += result.ret
            elif result.ret == SoapySDR.SOAPY_SDR_TIMEOUT:
                continue
            elif result.ret == SoapySDR.SOAPY_SDR_OVERFLOW:
                logger.warning("HackRF RX overflow. Samples were dropped.")
            else:
                raise RuntimeError(f"SoapySDR readStream error: {result.ret}")

        # Return a copy so the downstream consumer (like an ML model) can process 
        # it safely without it being overwritten by the next read_chunk() call.
        return self._chunk_buffer.copy()

    def stream(self):
        """Continuously yield fixed-size IQ chunks."""
        if not self._streaming:
            self.start()

        try:
            while self._streaming:
                yield self.read_chunk()
        finally:
            self.stop()

    def stop(self) -> None:
        """Safely stop the HackRF stream."""
        if self.sdr is None:
            return

        logger.info("Stopping HackRF...")
        self._streaming = False

        if self.rx_stream is not None:
            try:
                self.sdr.deactivateStream(self.rx_stream)
                self.sdr.closeStream(self.rx_stream)
            except Exception:
                pass

        self.rx_stream = None
        self.sdr = None
        logger.info("HackRF stopped.")