# Audio Signal Processing & Feature Extraction
**Author: Mohamed MELLOUK**

### 1.1 Dataset Selection and Literature Context
The foundational step of our pipeline relies on robust audio data. Drawing from literature in bioacoustics and environmental sound classification, we utilized a combination of public datasets: ESC-50, a curated subset of AudioSet, the ShivaRao dataset for canine vocalizations, and the CatMeows dataset (Pirrone et al.) for feline vocalizations. 

A critical observation from prior work is the discrepancy in how these sounds are labeled. The canine datasets primarily focus on *acoustic typing* (e.g., bark, growl, whine), relying on the physical properties of the sound. Conversely, feline datasets often label sounds based on *situational context* (e.g., isolation, waiting for food, brushing). This semantic gap poses a significant challenge for uniform classification, requiring our pipeline to map acoustic signals to behavioral intents carefully.

### 1.2 Addressing Real-World Noise and VAD
A major limitation of previous "Pet Translation" attempts is their reliance on pristine, studio-quality audio. To address this and test the hypothesis that our system can function in a real-world, noisy household environment, we engineered a rigorous preprocessing pipeline. We applied spectral gating for ambient noise reduction (using the `noisereduce` library) to filter out stationary background noises like laptop fans or air conditioning. 

Furthermore, to isolate the exact moment of vocalization from continuous recordings, we integrated a Voice Activity Detection (VAD) module based on WebRTC. This ensures the classifier only evaluates the animal's voice, drastically reducing false positives caused by human speech or environmental transients.

### 1.3 Feature Engineering and Dimensionality Standardization
Machine learning models, particularly CNNs adapted for audio, require highly standardized inputs. We transformed the 1D audio waveforms into 2D Mel-spectrograms using `librosa`. We chose Mel-spectrograms over standard STFTs or raw waveforms because the Mel scale applies logarithmic scaling to frequencies, which closely mimics the non-linear auditory perception of biological hearing systems.

During our data exploration phase, we encountered a significant engineering anomaly: while our assumptions held that pet vocalizations are brief (typically 1-3 seconds), statistical analysis revealed extreme outliers—most notably a single audio file lasting exactly 1 minute and 7 seconds. This outlier severely skewed our batch processing and normalization matrices. To solve this, we enforced a strict standardization protocol, cropping or padding audio to exactly 4 seconds for dogs and 2 seconds for cats. This careful engineering choice ensured uniform 96x96 feature matrices, providing mathematical stability for the downstream neural networks.
