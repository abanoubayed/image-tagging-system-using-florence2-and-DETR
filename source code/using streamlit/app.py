import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, DetrImageProcessor, DetrForObjectDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set the page layout for the Streamlit app
st.set_page_config(
    page_title="Automated Image Tagging System",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="auto",
)

# Define the header and description for the Streamlit app
st.title("Automated Image Tagging System")
st.markdown("""
This application allows you to upload images and 
automatically generate captions or detect objects within the image using powerful AI models.
Simply upload an image below and choose your desired task from the sidebar to get started!
""")

# Sidebar for task selection and submit button
task = st.sidebar.selectbox("Select a Task", ["Caption", "Object Detection"])
submit = st.sidebar.button("Submit")

# Function to load Florence-2 for captioning and DETR for object detection
@st.cache_resource
def load_models():
    florence_model_id = 'microsoft/Florence-2-large'
    florence_model = AutoModelForCausalLM.from_pretrained(florence_model_id, trust_remote_code=True).eval()
    florence_processor = AutoProcessor.from_pretrained(florence_model_id, trust_remote_code=True)

    detr_model_id = 'facebook/detr-resnet-50'
    detr_model = DetrForObjectDetection.from_pretrained(detr_model_id).eval()
    detr_processor = DetrImageProcessor.from_pretrained(detr_model_id)

    return florence_model, florence_processor, detr_model, detr_processor

# Load models and processors (cached for efficiency)
florence_model, florence_processor, detr_model, detr_processor = load_models()

# Function to process image and generate captions with Florence-2
def process_image(uploaded_image):
    task_prompt = '<CAPTION>'
    inputs = florence_processor(text=task_prompt, images=uploaded_image, return_tensors="pt")
    
    with torch.no_grad():
        generated_ids = florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=True,
            num_beams=3,
        )
    
    generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# Function for object detection with bounding boxes using DETR
def object_detection_with_bbox(uploaded_image):
    inputs = detr_processor(images=uploaded_image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = detr_model(**inputs)

    # Process DETR outputs to get boxes and labels
    target_sizes = torch.tensor([uploaded_image.size[::-1]])
    results = detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detected_objects = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        if score > 0.9:  # Confidence threshold
            x1, y1, x2, y2 = box.tolist()
            label_name = detr_model.config.id2label[label.item()]
            detected_objects.append((label_name, (x1, y1, x2, y2)))
    
    return detected_objects

# Function to plot bounding boxes for detected objects
def plot_bbox(uploaded_image, detected_objects):
    fig, ax = plt.subplots()
    ax.imshow(uploaded_image)
    
    # Draw bounding boxes and labels
    for label, (x1, y1, x2, y2) in detected_objects:
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))
    
    ax.axis('off')  # Hide the axes
    st.pyplot(fig)

# Uploading image section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Check if an image is uploaded and the submit button is clicked
if uploaded_file is not None and submit:
    # Display the uploaded image
    uploaded_image = Image.open(uploaded_file).convert("RGB")
    st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)

    # Run the selected task
    if task == "Caption":
        st.write("Generating caption...")
        generated_caption = process_image(uploaded_image)
        st.write("**Generated Tags**:")
        st.success(generated_caption)
    elif task == "Object Detection":
        st.write("Performing object detection...")
        detected_objects = object_detection_with_bbox(uploaded_image)
        st.write("**Detected Objects with Bounding Boxes:**")
        for label, bbox in detected_objects:
            st.write(f"Label: {label}, BBox: {bbox}")
        plot_bbox(uploaded_image, detected_objects)
else:
    if not uploaded_file:
        st.warning("Please upload an image to continue.")
    elif not submit:
        st.info("Select a task and click Submit to start processing.")

# Footer information
st.markdown("""
    ---
    **Project created by DEPI AWS_ML_GradTeam**  
    Powered by [Hugging Face](https://huggingface.co/)
""")

