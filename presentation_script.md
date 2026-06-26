# Pet Translator - Final Presentation Script (10 Minutes)

*Note: This script is designed for a ~10-minute video. Each section is roughly 2.5 minutes (approx. 300-350 words per person).*

---

## Part 1: Amine (Introduction & Full Architecture)
**[Slide 1: Title & Team]**
**Amine:** "Hello everyone! We are Amine, Mohamed, Anas, and Abir, and today we are thrilled to present our Machine Learning project: The Pet Translator. Have you ever wondered what your cat or dog is actually trying to say when they meow or bark at you? Our goal was to design a conceptual wearable device, running on a smartphone, that captures pet sounds, analyzes them using AI, and translates them into a human sentence with a real personality.

**[Slide 2: The Global Pipeline]**
**Amine:** To achieve this, we built a complex, end-to-end AI pipeline. It starts right in the browser. Using the MediaRecorder API, our React frontend captures the audio. But an raw audio file isn't enough. We need to classify it, and we need to run this on the 'Edge'—meaning directly on the user's phone for privacy and low latency. Once we classify the intent of the sound, we send that intent to a remote server where a Large Language Model transforms it into a funny, sarcastic, or loving English sentence. We've built an entire system using React, FastAPI, TensorFlow.js, and Llama 3.2. I'll pass the floor to Mohamed to explain the very first step: how we handle the raw audio."

---

## Part 2: Mohamed (Audio Signal Processing)
**[Slide 3: Datasets & Prior Studies]**
**Mohamed:** "Thank you, Amine. Before we could classify anything, we needed data. We worked with three main datasets: ESC-50, a subset of AudioSet, and specifically the ShivaRao dataset for dogs, and the CatMeows dataset from Pirrone et al. for cats. One major challenge we faced was that the dog dataset labels sounds by their acoustic *type*—like a bark, a growl, or a grunt. But the cat dataset labels sounds by the *context* or situation—like isolation, brushing, or waiting for food. 

**[Slide 4: Audio Cleaning & VAD]**
**Mohamed:** In a real-world scenario, a pet's environment is noisy. A microphone will pick up laptop fans, street noise, and human voices. My role was to build a robust audio processing pipeline in Python. First, I implemented ambient noise reduction using the `noisereduce` library, and then isolated the actual animal vocalization using Voice Activity Detection (WebrtcVAD). 

**[Slide 5: Feature Extraction (Mel-Spectrograms)]**
**Mohamed:** But machine learning models don't read audio waves; they read images. Using `librosa`, I converted the cleaned audio into Mel-spectrograms. To do this, we had to standardize the audio lengths. Fun fact: we noticed that while almost all the audio files were just a few seconds long, there was one massive outlier that lasted for exactly 1 minute and 7 seconds! That single file completely messed up some of our initial calculations before we realized what was happening. We eventually fixed it by forcing a strict fixed length—4 seconds for dogs, 2 seconds for cats—and applying logarithmic scaling to the Mel frequencies. These spectrograms are essentially visual fingerprints of the sounds. Now, to explain how the neural network reads those fingerprints, here is Anas."

---

## Part 3: Anas (Classification Models & Edge AI)
**[Slide 6: The AI Brain - MobileNetV2]**
**Anas:** "Thanks Mohamed. My job was to build the analytical brain. With small datasets—only about 113 dog clips and 440 cat clips—training a CNN from scratch leads to massive overfitting. I proved this by building a logistic regression baseline that memorized the training set perfectly but failed on validation. So, we used Transfer Learning. We took a frozen MobileNetV2 model, pre-trained on ImageNet, and treated Mohamed's spectrograms as images. We added a tiny dense head on top to classify the intents.

**[Slide 7: Edge Deployment (TFJS)]**
**Anas:** Because we wanted this to run locally on the phone, I converted the trained Keras model into TensorFlow.js format with INT8 quantization. It now runs entirely in the browser with less than 50 milliseconds of latency! 

**[Slide 8: The "Data Ceiling" Conclusion]**
**Anas:** But we wanted to be scientifically rigorous. Our dog model performs great, hitting around 85% accuracy. However, our cat model, specifically for the 'food' class, plateaued around 35% F1-score. I ran exhaustive experiments: tuning hyperparameters, applying heavy data augmentation, trying different classifiers like SVMs, switching to an Audio Spectrogram Transformer (AST), and even using SMOTE to handle class imbalance. Nothing broke the ceiling. Our scientific conclusion is that the bottleneck isn't the model; it's the data. The visual features of a 'food' meow in MobileNetV2 simply overlap too much with other meows. It was a fascinating lesson in the limits of small datasets. Now, Abir will explain how we turn these intents into actual human words."

---

## Part 4: Abir (LLM, LoRA, and RAG Integration)
**[Slide 9: The Linguistic Brain - Llama 3.2]**
**Abir:** "Thank you, Anas. Knowing that a cat is hungry or a dog is excited is great, but we wanted to give the pets a voice. I built a generative AI module using Llama 3.2. First, I generated a synthetic dataset of over 5000 examples mapping intents and personalities to natural language sentences. Then, using a Google Colab GPU and Unsloth, I fine-tuned the Llama model using LoRA. 

**[Slide 10: RAG - Scientific Contextualization]**
**Abir:** I converted the model to a lightweight GGUF format so it could be served on our FastAPI backend. But to make the translations truly realistic, I implemented a RAG system—Retrieval-Augmented Generation. I built a behavioral knowledge base and embedded it into a ChromaDB database using Langchain and Sentence-Transformers. Now, when the model generates a sentence for a 'growl', it automatically queries the scientific definition of a growl to ensure the response is grounded in actual veterinary science. We also implemented custom personalities, like a 'Haughty Cat' or a 'Shy Dog', controlling the creativity with a temperature of 0.7. 

**[Slide 11: Demonstration]**
**Amine:** And now, it’s time to see it in action! Here is a live demonstration of our React application... *(Show video demo of the app, recording a sound, showing the TFJS local prediction, and the LLM response)*. 

**[Slide 12: Conclusion]**
**Amine:** To conclude, this project allowed us to bridge the gap between Edge AI and heavy generative models. We learned how to handle real-world audio, the harsh realities of small datasets, and how to orchestrate complex AI agents. Thank you for listening!"
