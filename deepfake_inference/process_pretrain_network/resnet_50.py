from keras.applications import ResNet50 # type: ignore
from keras.applications.resnet import preprocess_input # type: ignore
from tensorflow import constant, expand_dims

def resnet_50(ndarray):
    rnet = ResNet50(include_top = False, weights = 'imagenet')
    output = preprocess_input(ndarray)
    output = constant(rnet.predict(output, verbose = False))
    output = expand_dims(output,0)
    return output



