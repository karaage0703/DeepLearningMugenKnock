import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from copy import copy

# config 
max_dim = 128
channel = 3

style_weight=5e-1
content_weight=1e4
total_variation_weight=30

epochs = 10
steps_per_epoch = 100


def load_img(path):
    # image read
    
    img = cv2.imread(path)
    long_dim = max(img.shape[:2])
    scale = max_dim / long_dim
    img = cv2.resize(img, None, fx=scale, fy=scale)
    
    img = np.expand_dims(img, axis=0)
    img = img[..., ::-1]
    img = img.astype(np.float32)
    img /= 255.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    """
    return img


def vgg_layers(layer_names):
    # get imagenet pretrained model
    vgg = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg16.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {content_name : value for content_name, value in zip(self.content_layers, content_outputs)}

        style_dict = {style_name : value for style_name, value in zip(self.style_layers, style_outputs)}
        
        return {'content' : content_dict, 'style' : style_dict}


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)


# train
def train():
    # image
    content_image = load_img('../Dataset/train/images/madara/madara_0007.jpg')
    style_image = load_img('../Dataset/train/images/akahara/akahara_0008.jpg')
    
    # Content layer where will pull our feature maps
    content_layers = ['block5_conv2'] 

    # Style layer of interest
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

    content_layers_N = len(content_layers)
    style_layers_N = len(style_layers)

    # extract model
    extractor = StyleContentModel(style_layers, content_layers)

    # original style and content matrix
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    # output image to be optimized
    image = tf.Variable(content_image)

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
    
    @tf.function()
    def train_step(image):
        with tf.GradientTape() as tape:
            # feed forward
            outputs = extractor(image)
            
            # style loss
            style_outputs = outputs['style']
            style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2) for name in style_outputs.keys()])
            style_loss *= style_weight / style_layers_N

            # content loss
            content_outputs = outputs['content']
            content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2) for name in content_outputs.keys()])
            content_loss *= content_weight / content_layers_N
            loss = style_loss + content_loss
            
            # total valiation loss
            x_deltas = image[:, :, 1:] - image[:, :, :-1]
            y_deltas = image[:, 1:] - image[:, :-1]
            loss += total_variation_weight * (tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas)))

        # get and apply gradient
        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        # clip image to [0, 1]
        image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))


    step = 0
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            train_step(image)
        
        _image = np.array(image * 255).astype(np.uint8)[0]
        plt.imshow(_image)
        plt.show()
        #display.display(tensor_to_image(image))
        print("Epoch : {}".format(epoch + 1))


    

# test
def test():
    print('not implemented')
        
    

def arg_parse():
    parser = argparse.ArgumentParser(description='CNN implemented with Keras')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    args = parser.parse_args()
    return args

# main
if __name__ == '__main__':
    args = arg_parse()

    if args.train:
        train()
    if args.test:
        test()

    if not (args.train or args.test):
        print("please select train or test flag")
        print("train: python main.py --train")
        print("test:  python main.py --test")
        print("both:  python main.py --train --test")
