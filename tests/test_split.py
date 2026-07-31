from data_processing.split import split_dataset


def test_split_dataset():

    drone_files = [
        "drone_rf/DJI_inspire_2_2G.bin",
        "drone_rf/DJI_inspire_2_5G_1of2.bin",
        "drone_rf/DJI_inspire_2_5G_2of2.bin",
        "drone_rf/DJI_matrice_100_2G.bin",
        "drone_rf/DJI_matrice_210_2G.bin",
        "drone_rf/DJI_matrice_210_5G_1of2.bin",
        "drone_rf/DJI_matrice_210_5G_2of2.bin",
        "drone_rf/DJI_mavic_mini_2G.bin",
        "drone_rf/DJI_mavic_pro_2G.bin",
        "drone_rf/DJI_phantom_4_2G.bin",
        "drone_rf/DJI_phantom_4_pro_plus_2G.bin",
        "drone_rf/DJI_phantom_4_pro_plus_5G_1of2.bin",
        "drone_rf/DJI_phantom_4_pro_plus_5G_2of2.bin",
    ]

    non_drone_files = [
        "random_rf/5G-NR/32274CF-dell-latitude-D20211201T191100M024240.data",
        "random_rf/5G-NR/32274CF-dell-latitude-D20211201T191200M056171.data",
        "random_rf/5G-NR/32274CF-dell-latitude-D20211201T191300M048183.data",

        "random_rf/LTE/30 mbps/32274CF-dell-latitude-D20211125T150800M004475.data",
        "random_rf/LTE/30 mbps/32274CF-dell-latitude-D20211125T150900M059227.data",
        "random_rf/LTE/30 mbps/32274CF-dell-latitude-D20211125T151000M056507.data",

        "random_rf/LTE/50 mbps/32274CF-dell-latitude-D20211125T152000M052549.data",
        "random_rf/LTE/50 mbps/32274CF-dell-latitude-D20211125T152100M004125.data",
        "random_rf/LTE/50 mbps/32274CF-dell-latitude-D20211125T152400M046037.data",

        "random_rf/Wi-Fi/802_11ax_mcs6_30mbps/32274CA-015-D20211123T132300M011437.data",
        "random_rf/Wi-Fi/802_11ax_mcs6_30mbps/32274CA-015-D20211123T132400M058863.data",
        "random_rf/Wi-Fi/802_11ax_mcs6_30mbps/32274CA-015-D20211123T133000M028048.data",

        "random_rf/Wi-Fi/802_11ax_mcs7_50mbps/32274CA-015-D20211123T131300M012278.data",
        "random_rf/Wi-Fi/802_11ax_mcs7_50mbps/32274CA-015-D20211123T131400M050226.data",
        "random_rf/Wi-Fi/802_11ax_mcs7_50mbps/32274CA-015-D20211123T131500M053479.data",
    ]

    (
        train_drone,
        val_drone,
        test_drone,
        train_non_drone,
        val_non_drone,
        test_non_drone,
    ) = split_dataset(drone_files, non_drone_files)

    assert len(train_drone) + len(val_drone) + len(test_drone) == len(drone_files)

    assert (
        len(train_non_drone)
        + len(val_non_drone)
        + len(test_non_drone)
        == len(non_drone_files)
    )

    assert set(train_drone).isdisjoint(val_drone)
    assert set(train_drone).isdisjoint(test_drone)
    assert set(val_drone).isdisjoint(test_drone)

    assert set(train_non_drone).isdisjoint(val_non_drone)
    assert set(train_non_drone).isdisjoint(test_non_drone)
    assert set(val_non_drone).isdisjoint(test_non_drone)

    assert (
        set(train_drone)
        | set(val_drone)
        | set(test_drone)
    ) == set(drone_files)

    assert (
        set(train_non_drone)
        | set(val_non_drone)
        | set(test_non_drone)
    ) == set(non_drone_files)