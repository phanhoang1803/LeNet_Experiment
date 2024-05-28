import tensorflow as tf



def preprocess(image, label):
    """
    Preprocesses an image and its corresponding label.
    
    Args:
        image: A numpy array representing the image.
        label: An integer representing the label of the image.
        
    Returns:
        image: A float tensor representing the normalized image.
        label: An integer tensor representing the label.
    """
    
    # Cast the image to float32 and normalize the pixel values to be between 0 and 1.
    image = tf.cast(image, tf.float32) / 255.0
    
    return image, label

def adapt_input(image, dst_shape):
    """
    Adapts the input image to match the destination shape.
    
    Args:
        image: A numpy array representing the input image.
        dst_shape: A tuple representing the destination shape.
        
    Returns:
        image: A float tensor representing the normalized and resized image.
    """
    
    # If the image has 3 channels and the destination shape has 1 channel,
    # convert the image to grayscale.
    if image.shape[-1] == 3 and dst_shape[-1] == 1:
        image = tf.image.rgb_to_grayscale(image)
    elif image.shape[-1] == 1 and dst_shape[-1] == 3:
        image = tf.image.grayscale_to_rgb(image)

    # Resize the image to match the destination shape.
    if image.shape[:-1] != dst_shape[:-1]:
        image = tf.image.resize(image, dst_shape[:-1])
    
    # Cast the image to float32 and normalize the pixel values to be between 0 and 1.
    image = tf.cast(image, tf.float32) / 255.0
    
    return image
