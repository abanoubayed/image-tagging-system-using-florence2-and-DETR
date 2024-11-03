# Automated Image Tagging System

An advanced image tagging system capable of analyzing images to detect objects, generate accurate labels, and produce descriptive captions, ideal for educational and professional uses.
## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Highlights](#project-Highlights)
- [Contributing](#contributing)
- [License](#license)
- ## Features
- Leveraging HuggingFace models (Florence-2, DETR) for effective zero-shot caption generation and object detection.
- Streamlit UI for easy interaction
- Integration with AWS services for deployment
- Real-time labeling and tagging of images
## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/abanoubayed/image-tagging-system-using-florence2-and-DETR/tree/mainautomated-image-tagging-system.git
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
## Usage
To start the application, run:
```bash
streamlit run app.py
```
## Project Highlights
System Design:
Leveraged AWS services for data storage, model hosting, and scalability, creating a robust cloud-based infrastructure.

![design](https://github.com/user-attachments/assets/a2776f24-67b3-4623-800b-f35d964e443d)

System Development:
Utilized the Florence-2 model for accurate caption generation and the DETR model for effective zero-shot object detection.

System Deployment:
Deployed the application on Streamlit, providing a user-friendly interface for real-time image tagging and captioning accessible through a web-based platform.
![Screenshot 2024-11-03 173758](https://github.com/user-attachments/assets/e337740c-5120-43c8-8c85-0b203cf7b3f7)
![Screenshot 2024-11-03 173906](https://github.com/user-attachments/assets/717b658d-2ddb-4448-b581-0c69e3279a8b)
![Screenshot 2024-11-03 173950](https://github.com/user-attachments/assets/c002e1e0-c6df-48f2-b41c-2b7a50f65b25)

## Contributing
Contributions are welcome! Please fork this repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
