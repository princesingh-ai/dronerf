import logging

from realtime.hackrf_reader import HackRFLiveReader
from realtime.mock_reader import MockHackRFLiveReader
from realtime.inference import DroneInference

logger = logging.getLogger(__name__)


class RealtimeDetector:
    """
    Connects the live HackRF (or Mock) IQ stream to the DroneCNN inference pipeline.
    """

    def __init__(
        self,
        center_freq: float = 2.437e9,
        sample_rate: float = 20e6,
        chunk_size: int = 500_000,
        lna_gain: int = 32,
        vga_gain: int = 30,
        amp_enable: bool = True,
        inference_batch_size: int = 128,
        use_mock: bool = False,
        mock_signal_type: str = "noise",
        model_path: str = None,
    ):
        if use_mock:
            logger.info("Using Mock SDR Reader.")
            self.reader = MockHackRFLiveReader(
                center_freq=center_freq,
                sample_rate=sample_rate,
                chunk_size=chunk_size,
                signal_type=mock_signal_type,
            )
        else:
            self.reader = HackRFLiveReader(
                center_freq=center_freq,
                sample_rate=sample_rate,
                chunk_size=chunk_size,
                lna_gain=lna_gain,
                vga_gain=vga_gain,
                amp_enable=amp_enable,
            )

        if model_path:
            self.inference = DroneInference(model_path=model_path, batch_size=inference_batch_size)
        else:
            self.inference = DroneInference(batch_size=inference_batch_size)

    def run(self) -> None:
        """Start realtime detection."""
        logger.info("Starting realtime detector...")
        chunk_index = 0

        try:
            self.reader.start()

            while True:
                iq_chunk = self.reader.read_chunk()
                result = self.inference.predict(iq_chunk)

                logger.info(
                    "Chunk %06d | Windows: %3d | Agreement: %.2f%% | Detection: %s",
                    chunk_index,
                    result["windows"],
                    result["agreement_ratio"] * 100,
                    result["prediction_str"],
                )

                chunk_index += 1

        except KeyboardInterrupt:
            logger.info("Realtime detection stopped by user.")

        finally:
            self.reader.stop()