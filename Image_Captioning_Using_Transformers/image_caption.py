import cv2
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate a caption for a single image
def generate_caption(img):
    inputs = processor(img, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Function to handle image folder captioning
def image_folder_captioning(folder_path):
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            img_cv2 = cv2.imread(img_path)
            if img_cv2 is not None:
                img = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
                caption = generate_caption(img)
            
                print(f"Image: {img_name}, Caption: {caption}")
            else:
                print(f"Failed to load image {img_name}")
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

# Function to handle video or webcam captioning
def video_or_webcam_captioning(source=0, is_video=False):
    if is_video:
        cap = cv2.VideoCapture(source)  # Open video file
    else:
        cap = cv2.VideoCapture(0)  # Open the webcam
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from video/webcam or end of video.")
            break
        
        if frame_count % 30 == 0:  # Process every 30th frame
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            caption = generate_caption(img)
            print(f"Frame {frame_count}: Caption: {caption}")

        cv2.putText(frame, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        frame_count += 1
        cv2.imshow("Video/Webcam Feed", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Main function to get user input and process accordingly
def main():
    print("Select Input Method:")
    print("1. Image Folder")
    print("2. Video")
    print("3. Webcam")
    
    choice = input("Enter the number for your choice (1/2/3): ")
    
    if choice == '1':
        folder_path = input("Enter the path to the image folder: ")
        if os.path.exists(folder_path):
            image_folder_captioning(folder_path)
        else:
            print("Invalid folder path.")
    
    elif choice == '2':
        video_path = input("Enter the path to the video file: ")
        if os.path.exists(video_path):
            video_or_webcam_captioning(video_path, is_video=True)
        else:
            print("Invalid video path.")
    
    elif choice == '3':
        video_or_webcam_captioning()  # Webcam is the default
    
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
