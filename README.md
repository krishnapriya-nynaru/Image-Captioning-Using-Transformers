# Image-Captioning-Using-Transformers
## Introduction
This repository implements an Image Captioning System using the BLIP (Bootstrapping Language-Image Pre-training) model, a cutting-edge transformer-based model specifically designed for vision-language tasks. BLIP combines visual and text data to generate highly accurate and context-aware captions for images, video frames, or webcam feeds. This system is capable of automatically producing appropriate descriptions for input images, making it an excellent tool for applications such as content creation, multimedia analysis, and assistive technology.

### Model Architecture
The core of the BLIP model consists of:

**Vision Encoder:** A deep neural network (often based on pre-trained models like ViT - Vision Transformer) that processes the image and extracts high-level visual features. This encoder represents the image as a series of embeddings that encapsulate the key components of the image.

**Language Model (Transformer):** A transformer-based architecture that takes in the visual embeddings produced by the Vision Encoder and conditions the language generation on these inputs. The transformer uses attention mechanisms to align parts of the image with the words in the caption.

**Cross-Modality Attention:** BLIP’s architecture integrates cross-modal attention, meaning it effectively models the relationship between image features and text, helping the model generate more accurate and descriptive captions. This attention mechanism enables the model to align the textual output with specific image regions, ensuring the caption reflects the image content correctly.

![Alt Text](https://github.com/krishnapriya-nynaru/Image-Captioning-Using-Transformers/blob/main/Image_Captioning_Using_Transformers/Model/model.gif)

### Model Overview: BLIP (Bootstrapping Language-Image Pre-training)
BLIP is a vision-language model developed by Salesforce Research, designed to handle a variety of vision and language tasks, including image captioning, visual question answering (VQA), and image-text retrieval. It leverages a large-scale pre-training paradigm to align visual and linguistic modalities, allowing it to understand and generate language from visual input. The architecture is built on top of the popular transformer model, widely recognized for its ability to handle complex, multi-modal data.

How BLIP Works?

BLIP operates by training on large datasets containing paired images and texts, allowing it to learn:

Object Recognition: Identifying objects and elements within an image.

Semantic Understanding: Understanding relationships between objects and contextual information.

Text Generation: Generating grammatically correct and contextually appropriate captions based on the visual input.

By doing so, BLIP captures both the visual aspects of images and how they relate to natural language, generating fluent and relevant image descriptions.

### Training Details
**Model Type:** Vision-Language model using Transformer architecture (BLIP).

**Input Resolution:** Varies based on the input image (resize before processing).

**Checkpoint:** Pretrained on large-scale image-text datasets such as ImageNet and MS COCO.

**Number of Parameters:** Approximately 110 million parameters for the base model.

**Model Size:** Around 480 MB for the base BLIP model.

**Training Tasks:**
- **Masked Language Modeling (MLM)** for understanding the structure of language with masked text tokens.
- **Image-Text Matching (ITM)** for aligning visual features with corresponding text descriptions.
- **Pretraining:** Large-scale pretraining on diverse image-text datasets, including ImageNet and MS COCO.
- **Cross-Modality Attention:** Utilizes cross-attention to align image regions with text tokens.
- **Fine-tuning:** Task-specific fine-tuning on datasets like MS COCO and Flickr30k for image captioning.

**Performance:** Achieves state-of-the-art results on image captioning tasks, excelling in caption accuracy and fluency.

**Model Variants:** BLIP-Base, BLIP-Large, and BLIP-Huge, with differing number of layers and model capacity for handling more complex tasks.

## Usage
1. For creating the Conda environment, follow steps 1-3 from the [***Usage Instructions in this link***](https://github.com/krishnapriya-nynaru/Dlib-Face-Recognition?tab=readme-ov-file#usage).
2. Clone the repository: 
   ```bash
   git clone https://github.com/krishnapriya-nynaru/Image-Captioning-Using-Transformers.git
3. Unzip the downloaded file: 
   ```bash
   unzip Image-Captioning-Using-Transformers.git
4. Install the required packages: 
   ```bash
   pip install -r requirements.txt 
5. Navigate to the project directory: 
   ```bash
   cd Image-Captioning-Using-Transformers
## Inference Instructions
You can run the script with specified paths or use default paths for image and video inputs. Below are the commands for both options:
### Inference on image
#### 1. Image Folder
- ***Specify Path***:
   ```bash
   python image_caption.py --method image --path /path/to/your/image_folder
- ***Default Path***(uses test_images folder)
    ```bash
    python image_caption.py --method image
### Inference on Video/webcam
#### 2. Video File
- ***Specify Path***:
   ```bash
   python image_caption.py --method video --path /path/to/your/video_file
- ***Default Path***(uses test_videos folder)
    ```bash
    python image_caption.py --method video
#### 3. Webcam
-  ***Start webcam***
    ```bash
    python image_caption.py --method webcam
### Results
***Below are some results on test Images:-***

![alt text](https://github.com/krishnapriya-nynaru/Image-Captioning-Using-Transformers/blob/main/Image_Captioning_Using_Transformers/output/output_image_1.jpg)

***Below are some results on test Videos:-***

![alt text](https://github.com/krishnapriya-nynaru/Image-Captioning-Using-Transformers/blob/main/Image_Captioning_Using_Transformers/output/output_video_4.jpg)

![alt text](https://github.com/krishnapriya-nynaru/Image-Captioning-Using-Transformers/blob/main/Image_Captioning_Using_Transformers/output/output_video_11.jpg)


### References
- https://huggingface.co/Salesforce/blip-image-captioning-base