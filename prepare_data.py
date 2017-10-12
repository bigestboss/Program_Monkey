import numpy as np
import PIL
from sklearn.feature_extraction import image

def prepare_data(file_images, blocks_per_image, block_size):

	with open(file_images) as f:

                images_names = f.readlines()
                images_names = [x.strip() for x in images_names]
                array_images=[]
                array_classes = []
                
	        for line in images_names:

                        print('Reading Image ' + line)
			im=PIL.Image.open(line)
			class_number=define_class(line)	        
			im=np.asarray(im).astype(np.float32)

			#im=centeredCrop(im,1000,1000)
			
			patches_r=patchify(im[:,:,0], block_size)
			patches_g=patchify(im[:,:,1], block_size)
			patches_b=patchify(im[:,:,2], block_size)

			total_blocks_per_image=0

					
			for x in range(0, patches_r.shape[0]):
				for y in range(0, patches_r.shape[1]):

					if(total_blocks_per_image<blocks_per_image):
						new_image=np.zeros(shape=((block_size[0],block_size[1],3)))

					#print(new_image.shape)
					#print(patches_r[x,y].shape)

						new_image[:,:,0]=patches_r[x,y]
						new_image[:,:,1]=patches_g[x,y]
						new_image[:,:,2]=patches_b[x,y]
							
						array_images.append(new_image)
						array_classes.append(class_number)
						total_blocks_per_image+=1

	array_images=np.array(array_images)
	array_classes=np.array(array_classes)
	return(array_images, array_classes) 


def centeredCrop(img, new_height, new_width):

   width =  np.size(img,1)
   height =  np.size(img,0)

   left = np.ceil((width - new_width)/2.)
   top = np.ceil((height - new_height)/2.)
   right = np.floor((width + new_width)/2.)
   bottom = np.floor((height + new_height)/2.)
   cImg= img[top:bottom, left:right,:]
   return cImg

def patchify(img, patch_shape):

    img = np.ascontiguousarray(img)  
    X, Y = img.shape
    x, y = patch_shape
    shape = ((X-x+1), (Y-y+1), x, y) 
    strides = img.itemsize*np.array([Y, 1, Y, 1])
    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)

def define_class(img_patchname):
	if img_patchname.find('HTC-1-M7')!=-1:
		class_image=0
		
	if img_patchname.find('iPhone-4s')!=-1:
		class_image=1
		
	if img_patchname.find('iPhone-6')!=-1:
		class_image=2
		
	if img_patchname.find('LG-Nexus-5x')!=-1:
		class_image=3
		
	if img_patchname.find('Motorola-Droid-Maxx')!=-1:
		class_image=4
		
	if img_patchname.find('Motorola-Nexus-6')!=-1:
		class_image=5
		
	if img_patchname.find('Motorola-X')!=-1:
		class_image=6
		
	if img_patchname.find('Samsung-Galaxy-Note3')!=-1:
		class_image=7
		
	if img_patchname.find('Samsung-Galaxy-S4')!=-1:
		class_image=8
		
	if img_patchname.find('Sony-NEX-7')!=-1: 
		class_image=9    
		
	return class_image
