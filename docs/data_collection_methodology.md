# DroneDetect V2 - Data Collection Methodology

**Dataset**: DroneDetect V2 (Swinney & Woods, 2021)
**Paper**: [The Effect of Real-World Interference on CNN Feature Extraction and Machine Learning Classification of Unmanned Aerial Systems](https://doi.org/10.3390/aerospace8070179)
**Dataset**: [IEEE DataPort - DroneDetect V2](https://dx.doi.org/10.21227/6w92-0x42)

Based on: Swinney & Woods (2021), *Aerospace* 2021, 8, 179. [DOI: 10.3390/aerospace8070179](https://doi.org/10.3390/aerospace8070179)

---

## Acquisition Hardware and Software

| Parameter | Value |
|-----------|--------|
| SDR | Nuand BladeRF (47 MHz - 6 GHz, cost: $480) |
| Antenna | Palm Tree Vivaldi (800 MHz - 6 GHz, cost: $18.99) |
| Software | GNURadio (open-source signal processing toolkit) |
| Center frequency | 2.4375 GHz |
| Bandwidth | 28 MHz |
| Sample rate | 60 Mbit/s |
| Recording duration | 2 seconds per file (1.2 x 10^8 complex samples) |
| Sample length for processing | 20 ms (1.2 x 10^6 samples) - extracted from 2s files |
| Samples per class | 500 (as per original paper, for V1 dataset) |
| Format | Raw IQ (interleaved floats, .dat files) |
| SDR Gain | Not documented (known limitation) |

**Note**: Raw .dat files contain 2-second recordings. These are split into 20 ms samples for signal processing and ML training (see [IEEE DataPort dataset description](https://dx.doi.org/10.21227/6w92-0x42)).

---

## Signal Processing Parameters

| Parameter | Value |
|-----------|--------|
| FFT size | 1024 |
| Window function | Hanning |
| Overlap | 120 samples |
| PSD resolution | 224x224 pixels |
| Spectrogram resolution | 224x224 pixels |

**Key finding**: PSD (frequency domain) features are significantly more robust to interference than spectrogram (time domain) features. See [Swinney & Woods 2021, Section 4.2](https://doi.org/10.3390/aerospace8070179).

---

## Flight Conditions During Recording

### Switched ON mode
- UAS and controller placed 4 m from antenna on ground

### Hovering mode
- Altitude: 20 m
- Position: Stationary above antenna

### Flying mode
- Altitude: 20 m
- Radius: 40 m around antenna
- Pilot/controller distance: ~4 m from detection system

---

## Interference Configuration

### Bluetooth interference
- Source: JBL Charge Bluetooth Speaker
- Distance: 2 m from antenna
- Activity: Music playback via phone placed next to antenna

### Wi-Fi interference
- Source: Apple MacBook
- Distance: 2 m from antenna
- Activity: YouTube video streaming via phone hotspot placed next to antenna

**Both interference sources active concurrently during all recordings**

---

## Drones and Transmission Protocols

| Drone | Protocol | EIRP (2.4 GHz) | Band |
|-------|-----------|----------------|-------|
| DJI Air 2S | OcuSync 3.0 | 26 dBm | 2.400-2.4835 GHz |
| DJI Mavic Pro 2 | OcuSync 2.0 | 25.5 dBm | 2.400-2.4835 GHz |
| DJI Inspire 2 | Lightbridge 2.0 | 20 dBm (100 mW) | 2.400-2.483 GHz |
| DJI Mavic Mini | Wi-Fi | 19 dBm | 2.400-2.4835 GHz |
| Parrot Disco | Wi-Fi (SkyController 2) | 19 dBm | 2.400-2.4835 GHz |
| DJI Phantom 4 | Lightbridge 2.0 | 20 dBm (100 mW) | 2.4 GHz |
| DJI Mavic Pro | OcuSync 1.0 | 26 dBm | 2.4 GHz |

### Transmission system characteristics

- **Wi-Fi**: Mavic Mini, Parrot Disco - standard 802.11 protocols
- **Lightbridge 2.0**: Inspire 2, Phantom 4 - proprietary DJI, max range 5 km, EIRP 100 mW
- **OcuSync 1.0**: Mavic Pro - adaptive channel selection, max range 7 km
- **OcuSync 2.0**: Mavic Pro 2 - dual-band 2.4/5.8 GHz diversity, max range 7 km
- **OcuSync 3.0**: Air 2S - enhanced version, max range 12 km

---

## Physical Drone Specifications

| Code | Drone | Weight (g) | Folded (mm) | Unfolded (mm) | Max Speed (km/h) |
|------|-------|-----------|-------------------|---------------------|-------------------|
| AIR | DJI Air 2S | 595 | 180x97x77 | 183x253x77 | 68.4 |
| MA1 | DJI Mavic 2 Pro | 907 | 214x91x84 | - | 72 |
| INS | DJI Inspire 2 | 3440-4250 | - | 427x425x317 | 94 |
| MIN | DJI Mavic Mini | 249 | 140x82x57 | 160x202x55 | 46.8 |
| DIS | Parrot Disco | 750 | - | 1150x580x119 (wingspan) | 80 |
| PHA | DJI Phantom 4 | 1380 | - | D350 (diagonal) | 72 |
| MAV | DJI Mavic Pro | 734-743 | 198x83x83 | - | 65 |

**Sources**: Official DJI specifications ([Air 2S](https://www.dji.com/air-2s/specs), [Mavic 2 Pro](https://www.dji.com/mavic-2/info), [Inspire 2](https://www.dji.com/inspire-2/specs), [Mavic Mini](https://www.dji.com/mavic-mini/specs), [Mavic Pro](https://www.dji.com/mavic/info)), Parrot Disco (multiple sources: Amazon, IEEE Spectrum)

**Notes**:
- Weight: takeoff mass with battery
- Speeds: Sport/S-Mode
- Parrot Disco: fixed-wing drone, dimensions differ from quadcopters
- DJI Air 2S: official commercial name (not "Mavic 2 Air S")

---

## Dataset Classification Tasks

### Detection (2 classes)
1. No UAS detected
2. UAS detected

### Type (8 classes)
1. No UAS detected
2-8. 7 drone types listed above

### Flight Mode (21 classes)
- 7 drones x 3 modes (switched on, hovering, flying) + "No UAS"
- Some combinations missing (see missing data analysis in exploration notebooks)

---

## Known Dataset Limitations

- **No ambient/noise class**: No recordings without drone (problematic for real-world detection)
- **SDR gain not documented**: Cannot replicate exact signal levels
- **2.4 GHz only**: Drones can switch to 5.8 GHz, not captured
- **Controlled environment**: Real-world conditions may differ significantly
- **Known class confusions**: OcuSync 1.0 vs 2.0 systems show similar signatures in 2.4 GHz band

---

## DroneDetect V1 vs V2

The original paper ([Swinney & Woods, 2021](https://doi.org/10.3390/aerospace8070179)) describes DroneDetect V1 with 500 samples per class. DroneDetect V2, publicly available on [IEEE DataPort](https://dx.doi.org/10.21227/6w92-0x42), contains a different subset of recordings with CLEAN and BOTH interference conditions only (no separate WIFI or BLUETOOTH categories).

---

## References

1. Swinney, C.J.; Woods, J.C. **The Effect of Real-World Interference on CNN Feature Extraction and Machine Learning Classification of Unmanned Aerial Systems**. *Aerospace* 2021, 8, 179. [DOI: 10.3390/aerospace8070179](https://doi.org/10.3390/aerospace8070179)

2. Swinney, C.J.; Woods, J.C. **DroneDetect Dataset V2**. IEEE DataPort, 2021. [DOI: 10.21227/6w92-0x42](https://dx.doi.org/10.21227/6w92-0x42)
