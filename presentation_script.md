# Pet Translator - Final Presentation Script (10 Minutes)

*Note: This script is designed for a ~10-minute video. Each section is roughly 2.5 minutes (approx. 300-350 words per person).*

---

## Part 1: Amine (Introduction & Full Architecture)
**[Slide 1: Pet Translator]**
**Amine:** "Hello everyone! We are Amine, Mohamed, Anas, and Abir, and today we are thrilled to present our Machine Learning project: The Pet Translator. Have you ever wondered what your cat or dog is actually trying to say when they meow or bark at you? Our goal was to design a conceptual wearable device, running on a smartphone, that captures pet sounds, analyzes them using AI, and translates them into a human sentence with a real personality.

**[Slide 2: Architecture]**
**Amine:** To achieve this, we built a complex, end-to-end AI pipeline. It starts right in the browser. Using the MediaRecorder API, our React frontend captures the audio. We run a classification model directly on the 'Edge'—meaning on the user's phone for privacy and low latency. Once we classify the intent of the sound, we send that intent to a remote server where a Large Language Model transforms it into a funny, sarcastic, or loving English sentence. We've built an entire system using React, FastAPI, TensorFlow.js, and Llama 3.2. I'll pass the floor to Mohamed to explain the very first step: how we handle the raw audio."

---

## Part 2: Mohamed (Audio Signal Processing)
**[Slide 3: Audio Signal Processing]**
**Mohamed:** "Thank you, Amine. Before we could classify anything, we needed data. We worked with three main datasets: ESC-50, a subset of AudioSet, and specifically the ShivaRao dataset for dogs and CatMeows dataset for cats. In a real-world scenario, a pet's environment is noisy. A microphone will pick up street noise and human voices. My role was to build a robust audio processing pipeline in Python. First, I implemented ambient noise reduction using the `noisereduce` library, and then isolated the actual animal vocalization using Voice Activity Detection. 

**[Slide 4: Feature Extraction]**
**Mohamed:** But machine learning models don't read audio waves; they read images. Using `librosa`, I converted the cleaned audio into Mel-spectrograms. To do this, we had to standardize the audio lengths. We noticed one massive outlier that lasted for exactly 1 minute and 7 seconds, which messed up our initial calculations! We eventually fixed it by forcing a strict fixed length—4 seconds for dogs, 2 seconds for cats—and applying logarithmic scaling to the Mel frequencies. These spectrograms are essentially visual fingerprints of the sounds. Now, to explain how the neural network reads those fingerprints, here is Anas."

---

## Part 3: Anas (Classification Models)
**[Slide 5: Classification Models]**
**Anas:** "Thanks Mohamed. My job was to build the analytical brain. With small datasets—only about 113 dog clips and 440 cat clips—training a CNN from scratch leads to massive overfitting. I proved this by building a logistic regression baseline that memorized the training set perfectly but failed on validation. So, we used Transfer Learning. We took a frozen MobileNetV2 model, pre-trained on ImageNet, and treated Mohamed's spectrograms as images. We added a tiny dense head on top to classify the intents. We also implemented a rigorous group-aware cross-validation to ensure zero data leakage.

**[Slide 6: The Data Ceiling]**
**Anas:** But we wanted to be scientifically rigorous. Our dog model performs great, hitting around 82-85% F1-score. However, our cat model, specifically for the 'food' class, plateaued around 30-37% F1-score. I ran exhaustive experiments: tuning the head, applying data augmentation, trying an Audio Spectrogram Transformer (AST), and using Focal Loss and SMOTE. Nothing broke the ceiling. Our scientific conclusion is that the bottleneck isn't the model; it's the dataset size and label overlap. It was a fascinating lesson in the limits of small datasets. Now, Abir will explain how we turn these intents into human words."

---

## Part 4: Abir (LLM, LoRA, and RAG Integration)
**[Slide 7: Linguistic Brain]**
**Abir:** "Thank you, Anas. Knowing that a cat is hungry or a dog is excited is great, but we wanted to give the pets a voice. I built a generative AI module using Llama 3.2. First, I generated a synthetic dataset of over 5000 examples mapping intents and personalities to natural language sentences. Then, using a Google Colab GPU and Unsloth, I fine-tuned the Llama model using LoRA. 

**[Slide 8: Grounding the AI]**
**Abir:** I converted the model to a lightweight GGUF format so it could be served on our FastAPI backend. But to make the translations truly realistic, I implemented a RAG system—Retrieval-Augmented Generation. I built a behavioral knowledge base and embedded it into a ChromaDB database. Now, when the model generates a sentence for a 'growl', it automatically queries the scientific definition of a growl to ensure the response is grounded in actual veterinary science. We also implemented custom personalities, like a 'Haughty Cat' or a 'Excited Dog'. I will now pass it back to Amine for the Frontend Integration."

---

## Part 5: Amine (Frontend, UX, and Demonstration)
**[Slide 9: Frontend Integration]**
**Amine:** "Thank you Abir. For the user interface, we built a modern React application with a dark theme and chat bubbles for translations. The most critical part here is the real-time feedback. When you record audio, our TFJS classifier runs in the browser and displays class probabilities in real-time, right alongside the LLM translation.

**[Slide 10: Edge Deployment & UX]**
**Amine:** Deploying to the Edge had its own challenges. We converted the Keras model to a TF.js GraphModel, and we actually had to fix 4 severe preprocessing bugs—like mel scaling and filter norms—to ensure the browser results mathematically matched Python exactly (diff=0). We implemented a strict WebGL backend load to prevent silent crashes on macOS, and added an energy-based VAD (RMS < 0.015) so the app immediately rejects silent audio without wasting computation. Finally, UX research showed us we needed a 3-second minimum perception delay to make the AI feel 'thoughtful'.

**[Slide 11: Demonstration]**
**Amine:** And now, it’s time to see it in action! Here is a live demonstration of our React application... *(Show video demo of the app, recording a sound, showing the TFJS local prediction, and the LLM response)*. 

**[Slide 12: Conclusion]**
**Amine:** To conclude, this project allowed us to successfully build a fully functional WebApp simulating a mobile application, bridging audio signal processing, Edge classification, and NLP. We learned how to handle real-world audio, the harsh realities of small datasets, and how to orchestrate complex AI architectures. Thank you for listening!"
