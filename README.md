
### **Day 14 - Image Captioning with CNN + LSTM**  
As part of my **#100DaysOfAI** challenge, on **Day 14**, I implemented an **Image Captioning System** that generates natural language descriptions for images by combining **Convolutional Neural Networks (CNN)** for image understanding and **Recurrent Neural Networks (LSTM)** for sequence generation.

---

### **Goal**  
Automatically generate human-like **captions for images** by extracting visual features and translating them into descriptive sentences using deep learning.

---

### **Technologies Used**

| Tool/Library     | Purpose                                               |
|------------------|--------------------------------------------------------|
| Python           | Core programming language                             |
| TensorFlow/Keras | Building CNN + LSTM model for image captioning        |
| OpenCV           | Image preprocessing (if needed)                       |
| NumPy            | Numerical operations                                  |
| Matplotlib       | Visualizing results and data                          |
| tqdm             | Progress bar for loops                                |
| InceptionV3      | Pre-trained CNN for image feature extraction          |
| Flickr8k Dataset | Dataset for training and testing the caption model    |

---

### **How It Works**

1. **Load and Preprocess Captions**
   - Loaded `Flickr8k.token.txt` file and cleaned the text data.
   - Added start (`startseq`) and end (`endseq`) tokens to each caption.

2. **Image Feature Extraction**
   - Used a pre-trained **InceptionV3** model (excluding the top softmax layer) to extract 2048-dimension features from each image.

3. **Tokenization and Vocabulary**
   - Tokenized all captions using Keras' `Tokenizer`.
   - Computed vocabulary size and maximum caption length.

4. **Sequence Creation**
   - Created input-output sequences: `(image_feature + partial_caption_input)` → `next_word_output`.

5. **Model Architecture**
   - Combined CNN (image features) and RNN (text sequence) using:
     - Dense + Dropout layers for image features.
     - Embedding + LSTM for text processing.
     - Merged the two with a decoder to predict the next word.

6. **Model Training**
   - Trained the combined model using categorical crossentropy loss.
   - Used a batch size of 64 over 20 epochs.

7. **Caption Generation**
   - Started with `'startseq'` and predicted one word at a time until `'endseq'` or max length is reached.

---

### **Highlights**

- Integrated **Computer Vision + NLP** into a unified pipeline.
- Used **Transfer Learning (InceptionV3)** for efficient visual representation.
- Built a real-world AI system capable of interpreting and describing images.
- Reinforced concepts of **sequence modeling**, **embedding**, and **attention to preprocessing**.
- Understood the importance of **pairing visual context with language models**.

---

### **What I Learned**

- How to preprocess and clean caption datasets for NLP tasks.
- Extracting rich visual features from CNNs (InceptionV3) for downstream tasks.
- Building hybrid models (CNN + RNN) for sequence generation.
- Implementing sequential prediction with teacher forcing.
- Real-world applications of image captioning in **assistive tech**, **search engines**, and **autonomous systems**.

---

