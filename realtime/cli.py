import logging

from realtime.detector import RealtimeDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def get_input(prompt: str, default_val, cast_type):
    """Helper function to get user input with a default fallback."""
    user_input = input(f"{prompt} [{default_val}]: ").strip()
    if not user_input:
        return default_val
    return cast_type(user_input)

def main():
    print("=== Realtime DroneRF Detector CLI ===")
    print("Press Enter to use the default values.\n")

    # Ask about using the Mock SDR first
    use_mock_input = get_input("Use Mock SDR for testing? (y/n)", "n", str)
    use_mock = use_mock_input.lower() == 'y'
    
    mock_signal_type = "noise"
    if use_mock:
        mock_signal_type = get_input("Mock signal type (noise/tone/burst)", "noise", str)

    # Standard settings
    frequency = get_input("Center frequency in Hz", 2.437e9, float)
    sample_rate = get_input("Sample rate in Hz", 20e6, float)
    chunk_size = get_input("Chunk size", 500_000, int)
    
    # Gain settings (only relevant if NOT using mock, but we'll collect them anyway)
    lna_gain = get_input("LNA gain (dB)", 32, int)
    vga_gain = get_input("VGA gain (dB)", 30, int)
    amp_input = get_input("Enable RF amplifier? (y/n)", "y", str)
    amp_enable = amp_input.lower() == 'y'
    
    batch_size = get_input("Inference batch size", 128, int)
    
    # Prompt for custom model checkpoint path
    from utils.config import config
    model_path = get_input("Model Checkpoint Path", config["training"]["model_path"], str)

    print("\nInitializing detector with chosen settings...\n")

    detector = RealtimeDetector(
        center_freq=frequency,
        sample_rate=sample_rate,
        chunk_size=chunk_size,
        lna_gain=lna_gain,
        vga_gain=vga_gain,
        amp_enable=amp_enable,
        inference_batch_size=batch_size,
        use_mock=use_mock,
        mock_signal_type=mock_signal_type,
        model_path=model_path,
    )

    detector.run()

if __name__ == "__main__":
    main()