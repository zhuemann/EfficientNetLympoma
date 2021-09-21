"Module containing code for custom data generators"
import cv2
import tensorflow
import numpy as np
from PIL import Image
from skimage.exposure import equalize_adapthist
import nibabel as nib
import keras

#pip install -U git+https://github.com/albu/albumentations
from albumentations.augmentations import transforms

try:
    import scipy
    from scipy import linalg
    from scipy import ndimage
except ImportError:
    scipy = None


class PngDataGenerator_albumen(tensorflow.keras.utils.Sequence):   #!! version problem -- tf2 this is tensorflow.keras.utils.Sequence
    """
    Image Data Generator with augmentation
    to be used for providing batches of
    images read by PIL to a model
    """

    def __init__(self,
                 file_list,
                 labels,
                 batch_size=32,
                 dim=(256, 256),
                 n_channels=1,
                 shuffle=True,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 zoom_range=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,    
                 preprocessing_function=None,#'CLAHE', or number to divide by, or
              
                 downscale = 0.,
                 gauss_noise = 0.,
                 gauss_blur = 0.,  #must be odd   np.ceil(f) // 2 * 2 + 1
                 elastic_transform = False,
                 elastic_transform_params = (100,10,10), #alpha, sigma, alpha_affine               
                 dtype='float32'):
        
        """initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = file_list
        self.n_channels = n_channels
        self.shuffle = shuffle

        # augmentation parameters
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.dtype = dtype
        
        self.downscale = downscale
        self.gauss_noise = gauss_noise
        self.gauss_blur = gauss_blur  #must be odd
        self.elastic_transform = elastic_transform,
        self.elastic_transform_params = elastic_transform_params #alpha, sigma, alpha_affine

        # designate axes
        self.channel_axis = 3
        self.row_axis = 1
        self.col_axis = 2

        # parse blur parameter -- must be odd
        if self.gauss_blur > 0:
            self.gauss_blur = np.int(np.ceil( self.gauss_blur) // 2 * 2 + 1)
  
        self.on_epoch_end()
        super().__init__()

    def on_epoch_end(self):
        'updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_files_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        Y = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        # Generate data
        for i, f in enumerate(list_files_temp):
            # load and resize image
            im = np.array(Image.open(f))
            if len(im.shape) > 2:
                im = im[..., 0]
            if im.shape[:2] != self.dim:
                im = cv2.resize(im, self.dim)
            im = im.astype(np.float)

            # normalize to [0,1]
            if self.rescale:
                im /= np.float(self.rescale)
                

            # apply CLAHE, if selected
            if self.preprocessing_function == 'CLAHE':
                im = equalize_adapthist(im)

            im = np.expand_dims(im,-1)

            # load mask
            mask = np.array(Image.open(self.labels[f]))
            # convert to binary
            mask = (mask > 0).astype(np.float)
            # resize if needed
            if mask.shape[:2] != self.dim:
                mask = cv2.resize(mask, self.dim)
            mask = mask[..., np.newaxis].astype(np.float)

            # apply random transformation
            x,y = self.apply_random_transform_image_and_mask(im, mask)

            # store image sample
            X[i, ] = x

            # store mask
            Y[i, ] = y

        return X, Y

    def __len__(self):
        'Denotes the number of batches per epoch'
        # edit this later after augmentation is implemented
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)
        return X, Y


    def apply_random_transform_image_and_mask(self, x, y):
        """Applies a transformation to an image according to given parameters.
        # Arguments
            x: 3D tensor, single image.
            transform_parameters: Dictionary with string - parameter pairs
                describing the transformation.
                Currently, the following parameters
                from the dictionary are used:
                - `'theta'`: Float. Rotation angle in degrees.
                - `'tx'`: Float. Shift in the x direction.
                - `'ty'`: Float. Shift in the y direction.
                - `'shear'`: Float. Shear angle in degrees.
                - `'zx'`: Float. Zoom in the x direction.
                - `'zy'`: Float. Zoom in the y direction.
                - `'flip_horizontal'`: Boolean. Horizontal flip.
                - `'flip_vertical'`: Boolean. Vertical flip.
                - `'channel_shift_intencity'`: Float. Channel shift intensity.
        # Returns
            A transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        
        
        prob_to_appy=0.25
        
        if self.downscale > 0. and self.downscale < 1.:
            aug = transforms.Downscale(scale_min=self.downscale, scale_max=0.95, interpolation=0, always_apply=False, p=prob_to_appy)
            augmented = aug(image=x, mask=y)
            x = augmented['image']
            y = augmented['mask']
        
        if self.gauss_noise:
            aug = transforms.GaussNoise(var_limit=(0,self.gauss_noise), mean=0.0, always_apply=False, p=prob_to_appy)
            augmented = aug(image=x, mask=y)
            x = augmented['image']
            y = augmented['mask']
        
        if self.gauss_blur:
            aug = transforms.GaussianBlur(blur_limit=self.gauss_blur, always_apply=False, p=prob_to_appy)  
            augmented = aug(image=x, mask=y)
            x = augmented['image']
            y = augmented['mask']    
            
        if self.horizontal_flip:
            aug = transforms.HorizontalFlip(always_apply=False, p=prob_to_appy)
            augmented = aug(image=x, mask=y)
            x = augmented['image']
            y = augmented['mask']    
            
               
        if self.elastic_transform:
            aug = transforms.ElasticTransform(alpha=self.elastic_transform_params[0], sigma=self.elastic_transform_params[1], alpha_affine=self.elastic_transform_params[2], interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=None, always_apply=False, p=prob_to_appy, approximate=False)
            augmented = aug(image=x, mask=y)
            x = augmented['image']
            y = augmented['mask']    
            
            
        if self.rotation_range or self.width_shift_range or self.height_shift_range or self.zoom_range:
            aug = transforms.ShiftScaleRotate(shift_limit=(-self.width_shift_range,self.height_shift_range), scale_limit=(-self.zoom_range,self.zoom_range), rotate_limit=(-self.rotation_range,self.rotation_range), interpolation=1, border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=False, p=prob_to_appy)
            augmented = aug(image=x, mask=y)
            x = augmented['image']
            y = augmented['mask'] 
    

        return x, y

    def apply_random_transform_image(self, x, seed=None):
        """Applies a random transformation to an image.
        # Arguments
            x: 3D tensor, single image.
            seed: Random seed.
        # Returns
            A randomly transformed version of the input (same shape).
        """
        prob_to_appy=0.25
        
        if self.downscale > 0 and self.downscale < 1:
            aug = transforms.Downscale(scale_min=self.downscale, scale_max=0.95, interpolation=0, always_apply=False, p=prob_to_appy)
            augmented = aug(image=x)
            x = augmented['image']
        
        if self.gauss_noise:
            aug = transforms.GaussNoise(var_limit=(0,self.gauss_noise), mean=0.0, always_apply=False, p=prob_to_appy)
            augmented = aug(image=x)
            x = augmented['image']
        
        if self.gauss_blur:
            aug = transforms.GaussianBlur(blur_limit=self.gauss_blur, always_apply=False, p=prob_to_appy)  
            augmented = aug(image=x)
            x = augmented['image'] 
            
        if self.horizontal_flip:
            aug = transforms.HorizontalFlip(always_apply=False, p=prob_to_appy)
            augmented = aug(image=x)
            x = augmented['image']
            
               
        if self.elastic_transform:
            aug = transforms.ElasticTransform(alpha=self.elastic_transform_params[0], sigma=self.elastic_transform_params[1], alpha_affine=self.elastic_transform_params[2], interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=None, always_apply=False, p=prob_to_appy, approximate=False)
            augmented = aug(image=x)
            x = augmented['image']  
            
            
        if self.rotation_range or self.width_shift_range or self.height_shift_range or self.zoom_range:
            aug = transforms.ShiftScaleRotate(shift_limit=(-self.width_shift_range,self.height_shift_range), scale_limit=(-self.zoom_range,self.zoom_range), rotate_limit=(-self.rotation_range,self.rotation_range), interpolation=1, border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=False, p=prob_to_appy)
            augmented = aug(image=x)
            x = augmented['image']
    

        return x


class PngClassDataGenerator_albumen(PngDataGenerator_albumen):
    def __init__(self,
                 file_list,
                 labels,
                 batch_size=32,
                 dim=(256, 256),
                 n_channels=1,
                 shuffle=True,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 zoom_range=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,    
                 preprocessing_function=None,#'CLAHE', or number to divide by, or
              
                 downscale = 0.,
                 gauss_noise = 0.,
                 gauss_blur = 0.,  #must be odd   np.ceil(f) // 2 * 2 + 1
                 elastic_transform = False,
                 elastic_transform_params = (100,10,10), #alpha, sigma, alpha_affine               
                 dtype='float32'):
        """initialization"""
        """initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = file_list
        self.n_channels = n_channels
        self.shuffle = shuffle

        # augmentation parameters
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.dtype = dtype
        
        self.downscale = downscale
        self.gauss_noise = gauss_noise
        self.gauss_blur = gauss_blur  #must be odd
        self.elastic_transform = elastic_transform,
        self.elastic_transform_params = elastic_transform_params #alpha, sigma, alpha_affine

        # designate axes
        self.channel_axis = 3
        self.row_axis = 1
        self.col_axis = 2

         # parse blur parameter -- must be odd
        if self.gauss_blur > 0:
            self.gauss_blur = np.int(np.ceil( self.gauss_blur) // 2 * 2 + 1)

        self.on_epoch_end()

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)
        return X, Y

    def __data_generation(self, list_files_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        Y = np.empty((self.batch_size,))
        # Generate data
        for i, f in enumerate(list_files_temp):
            # load and resize image
            im = np.array(Image.open(f))
            if len(im.shape) > 2:
                im = im[..., 0]
            if im.shape[:2] != self.dim:
                im = cv2.resize(im, self.dim)
            
            im = im.astype(np.float)
            if self.n_channels > 1:
                im = np.repeat(im,self.n_channels,axis=-1)
            
             # normalize to [0,1]
            if self.rescale:
                im /= np.float(self.rescale)

            # apply CLAHE, if selected
            if self.preprocessing_function == 'CLAHE':
                im = equalize_adapthist(im)

            im = np.expand_dims(im,-1)
            
            # apply random transformation
            x = self.apply_random_transform_image(im)

            # store image sample
            X[i, ] = x

            # store mask
            Y[i, ] = self.labels[f]

        return X, Y
    
    



















class NiftiDataGenerator_albumen(tensorflow.keras.utils.Sequence):
    """
    Image Data Generator with augmentation
    to be used for providing batches of
    images read by PIL to a model
    """

    def __init__(self,
                 file_list,
                 labels,
                 batch_size=6,
                 dim=(192,192,192),
                 n_channels=1,
                 shuffle=True,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 zoom_range=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,    
                 preprocessing_function=None,#'CLAHE', or number to divide by, or
              
                 downscale = 0.,
                 gauss_noise = 0.,
                 gauss_blur = 0.,  #must be odd   np.ceil(f) // 2 * 2 + 1
                 elastic_transform = 0,
                 elastic_transform_params = (100,10,10), #alpha, sigma, alpha_affine               
                 dtype='float32'):
        
        """initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = file_list
        self.n_channels = n_channels
        self.shuffle = shuffle

        # augmentation parameters
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.dtype = dtype
        
        self.downscale = downscale
        self.gauss_noise = gauss_noise
        self.gauss_blur = gauss_blur  #must be odd
        self.elastic_transform = elastic_transform,
        self.elastic_transform_params = elastic_transform_params #alpha, sigma, alpha_affine

        # designate axes
        self.channel_axis = 4
        self.row_axis = 1
        self.col_axis = 2
        self.z_axis = 3


        # parse blur parameter -- must be odd
        if self.gauss_blur > 0:
            self.gauss_blur = np.int(np.ceil( self.gauss_blur) // 2 * 2 + 1)
  
        self.on_epoch_end()
        super().__init__()

    def on_epoch_end(self):
        'updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_files_temp):
        'Generates data containing batch_size samples'
       # Initialization
        X = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        Y = np.empty((self.batch_size,))
        # Generate data
        for i, f in enumerate(list_files_temp):
            # load and resize image
            
            if self.n_channels == 2:
                im = np.stack((nib.load(f[0]).get_fdata(dtype=np.float32), nib.load(f[1]).get_fdata(dtype=np.float32)),axis=3)
            else:
                im = nib.load(list_files_temp[0]).get_fdata(dtype=np.float32)
                im = im[:,:,:,np.newaxis]
            
            # if len(im.shape) > 2:
            #     im = im[..., 0]
            if im.shape[:3] != self.dim:
                zoom_factor = tuple([i/j for i,j in zip(self.dim, im.shape[:3])])  #how much to zoom for each dimension
                im = ndimage.interpolation.zoom(im, zoom=(zoom_factor + (1,)))
     
#            if self.n_channels > 1:
#                im = np.repeat(im,self.n_channels,axis=-1)
            
             # normalize to [0,1]
            if self.rescale:
                im /= np.float(self.rescale)

            # apply CLAHE, if selected
#            if self.preprocessing_function == 'CLAHE':
#                im = equalize_adapthist(im)

#            im = np.expand_dims(im,-1
            # load mask
            mask = nib.load(self.labels[f]).get_fdata(dtype=np.float32)
            # convert to binary
            mask = (mask > 0).astype(np.float)
            # resize if needed
            if mask.shape[:3] != self.dim:
                zoom_factor = tuple(
                    [i / j for i, j in zip(self.dim, mask.shape[:3])])  # how much to zoom for each dimension
                mask = cv2.resize(mask, self.dim)
            mask = ndimage.interpolation.zoom(mask, zoom=(zoom_factor + (1,)))

            # apply random transformation
            x,y = self.apply_random_transform_image_and_mask(im, mask)

            # store image sample
            X[i, ] = x

            # store mask
            Y[i, ] = y

        return X, Y

    def __len__(self):
        'Denotes the number of batches per epoch'
        # edit this later after augmentation is implemented
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)
        return X, Y


    def apply_random_transform_image_and_mask(self, x, y):
        """Applies a transformation to an image according to given parameters.
        # Arguments
            x: 4D tensor, single image.
            transform_parameters: Dictionary with string - parameter pairs
                describing the transformation.
                Currently, the following parameters
                from the dictionary are used:
                - `'theta'`: Float. Rotation angle in degrees.
                - `'tx'`: Float. Shift in the x direction.
                - `'ty'`: Float. Shift in the y direction.
                - `'shear'`: Float. Shear angle in degrees.
                - `'zx'`: Float. Zoom in the x direction.
                - `'zy'`: Float. Zoom in the y direction.
                - `'flip_horizontal'`: Boolean. Horizontal flip.
                - `'flip_vertical'`: Boolean. Vertical flip.
                - `'channel_shift_intencity'`: Float. Channel shift intensity.
        # Returns
            A transformed version of the input (same shape).
        """
        # x is a single image, so it doesn't have image number at index 0
        
        
        prob_to_appy=0.25
        if self.downscale > 0. and self.downscale < 1.:
            aug_downscale = transforms.Downscale(scale_min=self.downscale, scale_max=0.95, interpolation=0, always_apply=False, p=prob_to_appy)
        if self.gauss_noise:
            aug_gauss_noise = transforms.GaussNoise(var_limit=(0,self.gauss_noise), mean=0.0, always_apply=False, p=prob_to_appy)

        if self.gauss_blur:
            aug_gauss_blur = transforms.GaussianBlur(blur_limit=self.gauss_blur, always_apply=False, p=prob_to_appy)

        if self.horizontal_flip:
            aug_horizontal_flip = transforms.HorizontalFlip(always_apply=False, p=prob_to_appy)

        if self.elastic_transform:
            aug_elastic = transforms.ElasticTransform(alpha=self.elastic_transform_params[0],
                                              sigma=self.elastic_transform_params[1],
                                              alpha_affine=self.elastic_transform_params[2],
                                              interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=0,
                                              mask_value=None, always_apply=False, p=prob_to_appy, approximate=False)

        if self.rotation_range or self.width_shift_range or self.height_shift_range or self.zoom_range:
            aug_transforms = transforms.ShiftScaleRotate(shift_limit=(-self.width_shift_range, self.height_shift_range),
                                              scale_limit=(-self.zoom_range, self.zoom_range),
                                              rotate_limit=(-self.rotation_range, self.rotation_range), interpolation=1,
                                              border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=False,
                                              p=prob_to_appy)


        for i in range(x.shape[2]):
            slice_i = x[:, :, i]
            mask_i = y[:, :, i]

            slice_i = 'fix for channels'
            if self.downscale > 0. and self.downscale < 1.:
                augmented = aug_downscale(image=slice_i, mask=mask_i)
                slice_i = augmented['image']
                mask_i = augmented['mask']


            if self.gauss_noise:
                augmented = aug_gauss_noise(image=slice_i, mask=mask_i)
                slice_i = augmented['image']
                mask_i = augmented['mask']

            if self.gauss_blur:
                augmented = aug_gauss_blur(image=slice_i, mask=mask_i)
                slice_i = augmented['image']
                mask_i = augmented['mask']

            if self.horizontal_flip:
                augmented = aug_horizontal_flip(image=slice_i, mask=mask_i)
                slice_i = augmented['image']
                mask_i = augmented['mask']


            if self.elastic_transform:
                augmented = aug_elastic(image=slice_i, mask=mask_i)
                slice_i = augmented['image']
                mask_i = augmented['mask']


            if self.rotation_range or self.width_shift_range or self.height_shift_range or self.zoom_range:
                augmented = aug_transforms(image=slice_i, mask=mask_i)
                slice_i = augmented['image']
                mask_i = augmented['mask']

            x[:, :, i] = slice_i
            y[:, :, i] = mask_i
    

        return x, y

    def apply_random_transform_image(self, x, seed=None):
        """Applies a random transformation to an image.
        # Arguments
            x: 3D tensor, single image.
            seed: Random seed.
        # Returns
            A randomly transformed version of the input (same shape).
        """
        prob_to_appy=0.25

        if self.downscale > 0. and self.downscale < 1.:
            aug_downscale = transforms.Downscale(scale_min=self.downscale, scale_max=0.95, interpolation=0,
                                                 always_apply=False, p=prob_to_appy)
        if self.gauss_noise:
            aug_gauss_noise = transforms.GaussNoise(var_limit=(0, self.gauss_noise), mean=0.0, always_apply=False,
                                                    p=prob_to_appy)

        if self.gauss_blur:
            aug_gauss_blur = transforms.GaussianBlur(blur_limit=self.gauss_blur, always_apply=False, p=prob_to_appy)

        if self.horizontal_flip:
            aug_horizontal_flip = transforms.HorizontalFlip(always_apply=False, p=prob_to_appy)

        if self.elastic_transform:
            aug_elastic = transforms.ElasticTransform(alpha=self.elastic_transform_params[0],
                                                      sigma=self.elastic_transform_params[1],
                                                      alpha_affine=self.elastic_transform_params[2],
                                                      interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                                                      value=0,
                                                      mask_value=None, always_apply=False, p=prob_to_appy,
                                                      approximate=False)

        if self.rotation_range or self.width_shift_range or self.height_shift_range or self.zoom_range:
            aug_transforms = transforms.ShiftScaleRotate(shift_limit=(-self.width_shift_range, self.height_shift_range),
                                                         scale_limit=(-self.zoom_range, self.zoom_range),
                                                         rotate_limit=(-self.rotation_range, self.rotation_range),
                                                         interpolation=1,
                                                         border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=False,
                                                         p=prob_to_appy)
        # loop through each z slice
        for i in range(x.shape[2]):

            slice_i = x[:, :, i, :]

            if self.downscale > 0. and self.downscale < 1.:
                #loop through each channel, apply same transform to all channels
                for j in range(x.shape[3]):
                    augmented = aug_downscale(image=slice_i[:,:,j])
                    slice_i[:,:,j] = augmented['image']

            if self.gauss_noise:
                for j in range(x.shape[3]):
                    augmented = aug_gauss_noise(image=slice_i[:,:,j])
                    slice_i[:,:,j] = augmented['image']

            if self.gauss_blur:
                for j in range(x.shape[3]):
                    augmented = aug_gauss_blur(image=slice_i[:, :, j])
                    slice_i[:, :, j] = augmented['image']

            if self.horizontal_flip:
                for j in range(x.shape[3]):
                    augmented = aug_horizontal_flip(image=slice_i[:, :, j])
                    slice_i[:, :, j] = augmented['image']

            if self.elastic_transform:
                for j in range(x.shape[3]):
                    augmented = aug_elastic(image=slice_i[:, :, j])
                    slice_i[:, :, j] = augmented['image']

            if self.rotation_range or self.width_shift_range or self.height_shift_range or self.zoom_range:
                for j in range(x.shape[3]):
                    augmented = aug_transforms(image=slice_i[:, :, j])
                    slice_i[:, :, j] = augmented['image']

            x[:, :, i, :] = slice_i

        return x










class NiftiClassDataGenerator_albumen(NiftiDataGenerator_albumen):
    def __init__(self,
                 file_list,
                 labels,
                 batch_size=32,
                 dim=(192,192,192),
                 n_channels=1,
                 shuffle=True,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 zoom_range=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,    
                 preprocessing_function=None,#'CLAHE', or number to divide by, or
              
                 downscale = 0.,
                 gauss_noise = 0.,
                 gauss_blur = 0.,  #must be odd   np.ceil(f) // 2 * 2 + 1
                 elastic_transform = 0,
                 elastic_transform_params = (100,10,10), #alpha, sigma, alpha_affine               
                 dtype='float32'):
        """initialization"""
        """initialization"""
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = file_list
        self.n_channels = n_channels
        self.shuffle = shuffle

        # augmentation parameters
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.dtype = dtype
        
        self.downscale = downscale
        self.gauss_noise = gauss_noise
        self.gauss_blur = gauss_blur  #must be odd
        self.elastic_transform = elastic_transform,
        self.elastic_transform_params = elastic_transform_params #alpha, sigma, alpha_affine

        # designate axes
        self.channel_axis = 4
        self.row_axis = 1
        self.col_axis = 2
        self.z_axis = 3

         # parse blur parameter -- must be odd
        if self.gauss_blur > 0:
            self.gauss_blur = np.int(np.ceil( self.gauss_blur) // 2 * 2 + 1)

        self.on_epoch_end()

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)
        return X, Y

    def __data_generation(self, list_files_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        Y = np.empty((self.batch_size,))
        # Generate data
        for i, f in enumerate(list_files_temp):
            # load and resize image
            
            if self.n_channels == 2:
                im = np.stack((nib.load(f[0]).get_fdata(dtype=np.float32), nib.load(f[1]).get_fdata(dtype=np.float32)),axis=3)
            else:
                im = nib.load(list_files_temp[0]).get_fdata(dtype=np.float32)
                im = im[:,:,:,np.newaxis]
            
            # if len(im.shape) > 2:
            #     im = im[..., 0]
            if im.shape[:3] != self.dim:
                zoom_factor = tuple([i/j for i,j in zip(self.dim, im.shape[:3])])  #how much to zoom for each dimension
                im = ndimage.interpolation.zoom(im, zoom=(zoom_factor + (1,)))
     
#            if self.n_channels > 1:
#                im = np.repeat(im,self.n_channels,axis=-1)
            
             # normalize to [0,1]
            if self.rescale:
                im /= np.float(self.rescale)

            # apply CLAHE, if selected
#            if self.preprocessing_function == 'CLAHE':
#                im = equalize_adapthist(im)

#            im = np.expand_dims(im,-1)
            
            # apply random transformation
            x = self.apply_random_transform_image(im)

            # store image sample
            X[i, ] = x

            # store mask
            Y[i, ] = self.labels[f]

        return X, Y
        
    
