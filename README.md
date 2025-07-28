# Deep Learning-Damage-Car-Detection
This deep learning project detects and classifies car damage from images using Convolutional Neural Networks (CNN) and transfer learning with pre-trained models like ResNet. The model is trained to identify 6 types of damage that are Rear Normal, Rear Crushed, Rear Breakage, Front Normal, Front Crushed and Front Breakage. It leverages transfer learning to improve accuracy and reduce training time, making it suitable for real-world applications like insurance claim automation or vehicle inspection systems.
- Achieved an accuracy of 80% on the validation/test dataset.
- Deployed the solution using Streamlit for an interactive, user-friendly web interface.

## Techstack
- **Python**
- **Streamlit for frontend development**
- **Pre-trained ResNet Model**
- **Convolutional Neural Network**


## Project Structure

- **streamlitapp/app.py and streamlitapp/model_helper.py**: Contains Streamlit and model prediction code.
- **notebooks folder**: Contains the jupyter notebook. 
- **requirements.txt**: Lists the required Python packages.
- **streamlitapp/model folder**: Contains the exported model as joblib file


## Setup Instructions

1. **Clone the repository**:
   ```bash
     git clone https://github.com/Himanshu-b20/DL-Damage-Car-Detection.git
   ```
2. **Install dependencies:**:   
   ```commandline
    pip install -r requirements.txt
   ```
3. **Run the Streamlit app:**:   
   ```commandline
    streamlit run streamlitapp/app.py

## Streamlit UI

<img width="1206" height="1276" alt="image" src="https://github.com/user-attachments/assets/4b9d448e-e075-4dc7-8f44-60f4e33bd061" />


   ```

