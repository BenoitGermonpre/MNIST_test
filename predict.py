from googleapiclient import discovery
import os
import numpy as np
import pandas as pd
from trainer.config import PROJECT_ID, WIDTH, HEIGHT
import matplotlib.pyplot as plt

## Set local path to folder that contains test data
DATA_DIR = "C:/Users/Benoit Germonpre/Google Drive/4C Documents/PycharmProjects/MNIST_test/data/"

import pandas as pd

url = "https://github.com/BenoitGermonpre/MNIST_test/blob/master/data/test.csv"
data = pd.read_csv(url)

def read_image(index, file='test.csv'):
    """Read an example image from a test file
    Args:
        index (int): index of the image to read
        file (str): file name that contains test images
    """
    data = pd.read_csv(os.path.join(DATA_DIR, file))
    image_pixels = np.array(data.iloc[index]) / 255

    return image_pixels


def get_predictions(project, model, instances, version=None):
    """Send json data to a deployed model for prediction.
    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    service = discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']



if __name__ == "__main__":

    # Read a test image and plot it
    image_pixels = read_image(index=1)
    plt.imshow(np.array(image_pixels).reshape(WIDTH, HEIGHT), cmap='gray')

    # Cast values to float
    images = [np.float(i) for i in image_pixels.tolist()]

    # Get predictions
    predictions = get_predictions(
        project=PROJECT_ID,
        model="mnist_model",
        instances=[
            {
                'image': images,
            }]
    )
    print(predictions)