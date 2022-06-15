
# Batch size
batch_size = 1	
real_batch_size = 22	

# Image size
image_size = (128, 256)	

# Total size of the noise
random_z = 8 * 16 * 2

# Number of training epochs
num_epochs = 750

# Learning rate for optimizers
lr_d = 1e-5				
lr_g = 1e-4				

# Unrolled parameters
unrolled_steps = 0
d_steps = 1				 
g_steps = 2	

# Weighting losses
lambda_BCE = 1.00		
lambda_DTW = 0.10		

# Checkpointing folder
ckpt_folder = 'checkpoints/checkpoint_'

# Model folder
model_folder = 'models/model_'

# Paths
images_path = 'data/img/'
gaze_path = 'data/gazes/'
