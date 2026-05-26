- can i test the pipeline with one single image and dashcam video coupled? 
- generated using the 4dgs ?
- I can for my puposes eliminate the Image VAE decoder at the end. Can I also eliminate the LiDaR VAE decoder as well?
- Do we need 2 diffusion U-Net models? If we aren't using the image decoder? Or does the cross attn model need this? 
- Can I recreate a simpler smaller architetcure which emulates this or does what this pipeline intends to do? Everything within it should be scaled to fit within my 12GB GPU.
- Can I create a pipeline and test it with a single pass of a sensor value and dashcam for training? 
- Which variant of (3DGS) can we consider for 4DGS pipeline? It should support dynamic rigid (vehicles) and deformable (pedestrian) objects. 
- which loss terms to consider for our case- Lidar only
- U-Net architecture (default is 2x). Can we reduce it to 1x, since we do not need video and for lidar only?
- Say we have the 4dgs generated coupled dashcam and AV logs coupled. What model params are needed next? 
- Explore alternative more efficient algorithms 
- Say I dont want to add the dashcam video, but first test pipeline with the 8AV and lidar logs. What should be the changes to the pipeline then? 

Video Generation
- Do not consider autoregressive video generation
- Dagger algorithm mentioned in the paper does this. 


Explain 
1. cross-view attn
the model replaces the 2D attention modules in the original LDM to
3D (1D cross views and 2D in spatial) and computes attentions on all images

2. cross-sensor attn: 
- Generate consistent images and LiDar within each block of the U-Net -promote continous information exchange. 
- Can we replace one with another? 
- allow features from both sensors to interact directly. I guess. this is what causes the lidar to be consistent 360 degree rebuilding?

Dataset for testing
- 3 sec long paired Fixed camera to AV log sequences
- Training steps
Step 1: 80 k
Step 2: 40 k
Step 3:
Step 4: 20 k

Total no. of params: 250M as per paper. Why will this not fit in my 12GB GPU then? 

Comparison pipeline
- X Drive 
- Can i make this pipeline optimised/lean so that it can run the XDrive dataset for training while fitting on my 12GB 3060 GPU?
- compare with XDrive output?

Summary Ideal Pipeline requirements
- no autoregressive component
- no video VAE head the end for generation
- no support for dashcam video


