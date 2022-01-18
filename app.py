from io import BytesIO
import streamlit as st
import pandas as pd
import torch
from torchvision import transforms
import PIL

STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""

IMG_SIZE = 224

def load_model(model_name):
    model = torch.load(model_name, map_location=torch.device("cpu"))
    return model


def predict(model, image):
    transforms_test = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ]
    )
    img_input = transforms_test(PIL.Image.open(image).convert("RGB"))
    img_input = torch.unsqueeze(img_input, 0)
    output = model(img_input)
    probs = torch.max(torch.softmax(output, 1, torch.float)).item() * 100
    preds = torch.argmax(output, dim=1).item()
    
    return {"probs": probs, "preds": preds}
    


class FileReader(object):

    def __init__(self, learner, file_types, labels):
        self.learner = learner
        self.file_types = file_types
        self.labels = labels

    def run(self):
        """
            UPLOAD file on streamlit
        """
        file = st.file_uploader("Upload file", type=self.file_types)
        show_file = st.empty()

        if not file:
            show_file.info(f"Please Upload a file: {', '.join(self.file_types)}")
            return

        content = file.getvalue()

        if isinstance(file, BytesIO):
            show_file.image(file)
            output = predict(self.learner, file)
            st.text(f"Predicted {self.labels[output['preds']]} with probability of {output['probs']:.2f}")

        file.close()

if __name__ == "__main__":
    learner = load_model("model.pth")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.title("Mask Detection")
    file_reader = FileReader(learner, ["png", "jpg"], ["Mask worn properly", "Mask not worn properly: nose out", "Mask not worn properly: chin and nose out", "Didn't wear mask"])
    file_reader.run()
