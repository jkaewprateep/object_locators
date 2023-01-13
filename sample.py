

import tensorflow as tf

import matplotlib.pyplot as plt

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def draw_bounding_box( image, n = 0 ) :

	image = tf.squeeze( image )

	if n == 0 :
		## box 0 ##
		target_top = 0.3
		target_left = 0.0
		target_button = 0.4
		target_right = 0.1
		
		target_top = 1 - target_top
		target_button = abs(target_button - 1)
		## end box 0 ##

	elif n == 1 :
		## box 1 ##
		target_top = 0.4
		target_left = 0.1
		target_button = 0.5
		target_right = 0.2
		
		target_top = 1 - target_top
		target_button = abs(target_button - 1)
		## end box 1 ##
	
	elif n == 2 :
		## box 2 ##
		target_top = 0.3
		target_left = 0.2
		target_button = 0.4
		target_right = 0.3
		
		target_top = 1 - target_top
		target_button = abs(target_button - 1)
		## end box 2 ##
	
	elif n == 3 :
		## box 3 ##
		target_top = 0.4
		target_left = 0.3
		target_button = 0.5
		target_right = 0.4
		
		target_top = 1 - target_top
		target_button = abs(target_button - 1)
		## end box 3 ##
			
	elif n == 4 :
		## box 4 ##
		target_top = 0.3
		target_left = 0.4
		target_button = 0.4
		target_right = 0.5
		
		target_top = 1 - target_top
		target_button = abs(target_button - 1)
		## end box 4 ##
	
	elif n == 5 :
		## box 5 ##
		target_top = 0.4
		target_left = 0.5
		target_button = 0.5
		target_right = 0.6
		
		target_top = 1 - target_top
		target_button = abs(target_button - 1)
		## end box 5 ##
	
	elif n == 6 :
		## box 6 ##
		target_top = 0.3
		target_left = 0.6
		target_button = 0.4
		target_right = 0.7
		
		target_top = 1 - target_top
		target_button = abs(target_button - 1)
		## end box 6 ##	
	
	elif n == 7 :
		## box 7 ##
		target_top = 0.4
		target_left = 0.7
		target_button = 0.5
		target_right = 0.8
		
		target_top = 1 - target_top
		target_button = abs(target_button - 1)
		## end box 7 ##
	
	elif n == 8 :
		## box 6 ##
		target_top = 0.3
		target_left = 0.8
		target_button = 0.4
		target_right = 0.9
		
		target_top = 1 - target_top
		target_button = abs(target_button - 1)
		## end box 8 ##	

	elif n == 9 :
		## box 9 ##
		target_top = 0.4
		target_left = 0.9
		target_button = 0.5
		target_right = 1.0
		
		target_top = 1 - target_top
		target_button = abs(target_button - 1)
		## end box 9 ##

	else :
		return image
		
	colors = tf.constant([[1.0, 0.0, 0.0]], shape=(1, 3))
	boxes_custom = tf.constant( [ target_button, target_left, target_top, target_right ], shape=(1, 1, 4)).numpy()
	result_image = tf.image.draw_bounding_boxes( tf.expand_dims(image, axis=0), boxes_custom, colors )
	
	return result_image
	
def target_conv_image( image ):

	image = tf.keras.utils.img_to_array( tf.squeeze( image ) )

	image = tf.keras.layers.Conv2D(6, (3, 3), kernel_initializer=tf.ones_initializer(), bias_initializer=tf.ones_initializer(), activation='relu')( tf.expand_dims(image, axis=0) )
	image = tf.keras.layers.MaxPooling2D((2, 2))( image )
	image = tf.keras.layers.Conv2D(6, (3, 3), kernel_initializer=tf.ones_initializer(), bias_initializer=tf.ones_initializer(), activation='relu')( image )
	image = tf.keras.layers.MaxPooling2D((2, 2))( image )
	image = tf.squeeze( image )
	# result = ( image[:,:,0:1] + image[:,:,1:2] + image[:,:,2:3] + image[:,:,3:4] + image[:,:,4:5] + image[:,:,5:6] ) / 6
	
	result = tf.reduce_sum( image, axis=2 ) / 6
	result = tf.keras.utils.array_to_img( tf.expand_dims(result, axis=2) )

	return result

def object_locators( FILE="F:\\Pictures\\actor-Ploy\\001.jpg" ):
	# FILE = "F:\\Pictures\\actor-Ploy\\001.jpg"
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


	Dense_layer = tf.keras.layers.Dense(10, kernel_initializer=tf.random_normal_initializer(mean=1., stddev=2.), bias_initializer=tf.constant_initializer( value=0.0 ), activation='relu')
	temp = Dense_layer( tf.keras.utils.img_to_array( tf.keras.utils.img_to_array( original_conv_image ) - tf.keras.utils.img_to_array( result_image ) ) )


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

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Tasks
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

image_original, result_image, idx = object_locators( "F:\\Pictures\\actor-Ploy\\001.jpg" )

plt.figure(figsize=(2, 2))
plt.suptitle("Object locations")

plt.subplot(2, 2, 1)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow( tf.keras.utils.array_to_img( image_original ) )
plt.xlabel( "Original")

plt.subplot(2, 2, 2)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow( result_image )
plt.xlabel( str( idx.numpy() ) )

image_original, result_image, idx = object_locators( "F:\\Pictures\\actor-Ploy\\004.jpg" )	

plt.subplot(2, 2, 3)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow( tf.keras.utils.array_to_img( image_original ) )
plt.xlabel( "Original")

plt.subplot(2, 2, 4)
plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.imshow( result_image )
plt.xlabel( str( idx.numpy() ) )

plt.show()
