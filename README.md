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
•System Design: Leveraged AWS services for data storage, model hosting, and scalability, creating arobust cloud-based infrastructure.

•System Development: Utilized the Florence-2 model for accurate caption generation and the DETRmodel for effective zero-shot object detection.

•System Deployment: Deployed the application on Streamlit, providing a user-friendly interface forreal-time image tagging and captioning accessible through a web-based platform.
## Contributing
Contributions are welcome! Please fork this repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
