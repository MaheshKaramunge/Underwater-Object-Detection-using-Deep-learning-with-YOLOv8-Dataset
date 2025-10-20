# Underwater-Object-Detection-using-Deep-learning-with-YOLOv8-Dataset

This project detects underwater objects and waste materials such as plastic bottles, metal cans, nets, and other debris using a YOLOv8 deep learning model integrated with a Streamlit web application.

It aims to support marine environment conservation by automatically identifying and classifying objects present in underwater images.

🚀 Project Overview
Objective: Detect and classify different types of underwater objects using computer vision.
Model Used: YOLOv8 (You Only Look Once – Version 8)
Frameworks: Streamlit (for the web interface) + Ultralytics YOLO (for training/inference)
Output: Detected objects are highlighted with bounding boxes and confidence levels.

🧠 Key Features
✅ Real-time underwater object detection
✅ Supports multiple object classes (bottles, cans, nets, paper, bags, etc.)
✅ Easy-to-use and interactive Streamlit interface
✅ Trained on a custom underwater dataset
✅ Downloadable annotated detection results

📂 Project Structure
underwater_object_detection/
│
├── data/
│   ├── images/           # Training, validation & test images
│   └── labels/           # YOLO format labels
│
├── runs/
│   └── detect/
│       └── train2/       # Folder containing best.pt (trained model)
│
├── src/
│   └── app.py            # Streamlit web application
│
├── dataset.yaml          # Dataset configuration file
├── requirements.txt      # Required dependencies
├── README.md             # Project documentation
└── .gitignore

🧩 Dataset Details
Dataset Type: Custom YOLO-formatted underwater dataset
Classes:
plastic_bag
metal_can
plastic_bottle
paper
fishing_net
star_fish

Annotation Format:
Each .txt file follows YOLO format:
class_id  x_center  y_center  width  height

⚙️ System Requirements
Hardware:
GPU Recommended (e.g., NVIDIA RTX 3050 or higher)
Minimum 8GB RAM
Software:
Python 3.8+
CUDA-enabled PyTorch
Streamlit

🧩 Installation & Setup
1️⃣ Clone this Repository
git clone https://github.com/<your-username>/underwater_object_detection.git
cd underwater_object_detection

2️⃣ Create Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate  # For Windows
# source .venv/bin/activate   # For Mac/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Web Application
streamlit run src/app.py

5️⃣ Upload an Underwater Image
The app will display the uploaded image.
It will then show the detection result with labeled objects.
You can download the processed image.
🧮 Training the Model (Optional)
If you want to train your own model:
yolo task=detect mode=train data=dataset.yaml model=yolov8s.pt epochs=50 imgsz=640

After training, the model weights will be saved at:

runs/detect/train*/weights/best.pt


Use this file in the Streamlit app as:
MODEL_PATH = "runs/detect/train2/weights/best.pt"

🖼️ Sample Results
Input Image	Detection Output

	
🧰 Technologies Used
Component	Technology
Model	YOLOv8 (Ultralytics)
Language	Python
Web Framework	Streamlit
Libraries	OpenCV, PIL, PyTorch, Ultralytics
IDE	Visual Studio Code
💡 Future Enhancements

Deploy on Streamlit Cloud / Hugging Face Spaces

Add real-time underwater video detection

Improve accuracy using transfer learning

Include object count and analytics reports

🙌 Acknowledgements

Ultralytics YOLOv8

Streamlit

Open-source underwater datasets

Researchers working towards ocean cleanup and marine conservation 🌎
