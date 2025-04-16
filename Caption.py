import numpy as np
import os
import re
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

# Step 1: Load and Preprocess Captions
def load_captions(filepath):
    with open(filepath, 'r') as file:
        captions = file.readlines()
    desc = {}
    for line in captions:
        tokens = line.strip().split('\t')
        img_id, caption = tokens[0].split('#')[0], tokens[1]
        if img_id not in desc:
            desc[img_id] = []
        desc[img_id].append('startseq ' + caption.lower() + ' endseq')
    return desc

# Clean captions
def clean_caption(caption):
    caption = re.sub(r'[^a-zA-Z ]', '', caption)
    caption = caption.lower().split()
    caption = [word for word in caption if len(word) > 1]
    return ' '.join(caption)

# Step 2: Image Feature Extraction using InceptionV3
def extract_features(directory):
    model = InceptionV3(weights='imagenet')
    model_new = Model(model.input, model.layers[-2].output)  # Remove last softmax layer

    features = {}
    for img_name in tqdm(os.listdir(directory)):
        img_path = directory + '/' + img_name
        img = load_img(img_path, target_size=(299, 299))
        img = img_to_array(img)
        img = preprocess_input(np.expand_dims(img, axis=0))
        feature = model_new.predict(img, verbose=0)
        features[img_name] = feature
    return features

# Step 3: Tokenization
def tokenize_descriptions(descriptions):
    all_captions = []
    for key in descriptions:
        all_captions.extend(descriptions[key])
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, vocab_size

# Step 4: Create Sequences
def create_sequences(tokenizer, max_length, desc_list, photo):
    X1, X2, y = [], [], []
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            X1.append(photo)
            X2.append(in_seq)
            y.append(to_categorical([out_seq], num_classes=vocab_size)[0])
    return np.array(X1), np.array(X2), np.array(y)

# Step 5: Build CNN + LSTM Model
def build_model(vocab_size, max_length):
    # Image feature extractor (CNN)
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Sequence model (RNN)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Merge both models and decoder
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Step 6: Train Model
def train_model(model, X1, X2, y, epochs=20, batch_size=64):
    model.fit([X1, X2], y, epochs=epochs, batch_size=batch_size, verbose=1)

# Step 7: Generate Captions
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text

# Main Code
if __name__ == "__main__":
    # Load captions and clean them
    captions_path = 'Flickr8k_text/Flickr8k.token.txt'
    descriptions = load_captions(captions_path)
    for img_id in descriptions:
        descriptions[img_id] = [clean_caption(c) for c in descriptions[img_id]]

    # Extract features from images
    image_dir = 'Flickr8k_Dataset'
    features = extract_features(image_dir)

    # Tokenize captions
    tokenizer, vocab_size = tokenize_descriptions(descriptions)
    max_length = max([len(caption.split()) for captions in descriptions.values() for caption in captions])

    # Create sequences for training
    X1, X2, y = [], [], []
    for img_id, captions in descriptions.items():
        photo = features[img_id]
        seqX1, seqX2, seqy = create_sequences(tokenizer, max_length, captions, photo)
        X1.append(seqX1)
        X2.append(seqX2)
        y.append(seqy)

    X1 = np.concatenate(X1)
    X2 = np.concatenate(X2)
    y = np.concatenate(y)

    # Build and train the model
    model = build_model(vocab_size, max_length)
    train_model(model, X1, X2, y, epochs=20)

    # Save the trained model
    model.save('image_caption_model.h5')

    img_name = 'example.jpg'  
    photo = features[img_name]  
    caption = generate_caption(model, tokenizer, photo, max_length)
    print("Generated Caption:", caption)
