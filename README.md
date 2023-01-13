# object_locators
For study object locator using TensorFlow layer without grids loop


## object_locators ##

It is simply matching boxes we remarks after removed original image it is the ```picture + our remark values``` .

1. Read target image file and decode it into binary, image or array of numbers format.
2. Draws gride boxes for the area.
3. Convolution the image for normalized pixel color value within the kernel size ```Conv layers```, specific the initialize value and bias to 0.
4. Pass the previous step result into ```Dense ```, target result is image with in rages of value standards ( test it if input has value it return some value contrast ).
5. Find contrast X and contrast Y and resizes for summation.
6. Presents.

```
def object_locators( FILE="F:\\Pictures\\actor-Ploy\\001.jpg" ):
    image = tf.io.read_file( FILE )
    image = image_original = tf.io.decode_jpeg( image )
    image = tf.keras.utils.img_to_array( image )
    print( "image shape: " + str( image.shape ) )

    result_image = draw_bounding_box( image, n = 0 )
    result_image = draw_bounding_box( result_image, n = 1 )
    result_image = draw_bounding_box( result_image, n = 2 )
    result_image = draw_bounding_box( result_image, n = 3 )
    result_image = draw_bounding_box( result_image, n = 4 )
    result_image = draw_bounding_box( result_image, n = 5 )
    result_image = draw_bounding_box( result_image, n = 6 )
    result_image = draw_bounding_box( result_image, n = 7 )
    result_image = draw_bounding_box( result_image, n = 8 )
    result_image = draw_bounding_box( result_image, n = 9 )

    result_image = target_conv_image( result_image )
    original_conv_image = target_conv_image( image )


    Dense_layer = tf.keras.layers.Dense(10, kernel_initializer=tf.random_normal_initializer(mean=1., stddev=2.), 
                    bias_initializer=tf.constant_initializer( value=0.0 ), activation='relu')
    temp = Dense_layer( tf.keras.utils.img_to_array( tf.keras.utils.img_to_array( original_conv_image ) - 
                    tf.keras.utils.img_to_array( result_image ) ) )


    temp = tf.math.argmax( temp, axis=2)
    temp_y = tf.math.argmax( temp, axis=1)
    temp_x = tf.math.argmax( temp, axis=0)


    temp_x = tf.expand_dims(temp_x, axis=0)
    temp_y = tf.expand_dims(temp_y, axis=1)
    temp = tf.math.multiply(temp_x, temp_y)
    temp = tf.expand_dims(temp, axis=2)


    image = tf.image.resize(temp, [23, 18])
    image = image[11:16,:,:]

    image = tf.reduce_sum( image, axis=0 )
    temp = tf.keras.activations.softmax( image, axis=0 )
    y, idx = tf.unique( tf.squeeze( temp ).numpy() )

    return image_original, result_image, idx
```

## Files and Directory ##

| File name | Description |
| --- | --- |
| sample.py | sample codes |
| Figure_1.png | result 1 |
| Figure_2.png | result 2 |
| Figure_3.png | result 3 |
| README.md | readme file |

## Result ##

![result 1](https://github.com/jkaewprateep/object_locators/blob/main/Figure_1.png "result 1")


![result 2](https://github.com/jkaewprateep/object_locators/blob/main/Figure_2.png "result 2")


![result 3](https://github.com/jkaewprateep/object_locators/blob/main/Figure_3.png "result 3")
