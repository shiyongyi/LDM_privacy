Synthesizing Big Data of High Quality without Privacy Leakage 
–Competitive Performance of Deep CT Denoising Networks Trained on Diffusion Model-generated Data


Install

* To install all available Python packages, please use the following command:
  pip install -r requirements.txt

* Please install our CTLIB library for CT reconstruction:
* Go to [5_SOTA_LDCT_denoising/CTLIB-main/setup.py] and run: python setup.py install;
For linux installation, remove library_dirs and extra_link_args in setup.py first.

Get Started 

* Before executing the files, please navigate to the respective files path.
* The pre-trained checkpoints for the main models have been shared at:
https://drive.google.com/drive/folders/1hp1J41b8HduP1V4TtBdvVUF069WvrO9I
  
1. Data Preparation [1]

* Go to [1_data_preparation/prep.m] to prepare the data;
* Revise Line 12 ('..','MayoDataset') with your own path to the Mayo Clinic Dataset;
* [Optional] Revise Line 13 ('..','data','ori_dataset') with your own path to save the original dataset.

The 2016 NIH-AAPM-Mayo Clinic Low Dose CT Grand Challenge organized by Mayo Clinic   
(I am authorized to share this dataset but you can ask Mayo Clinical directly for permission 
at the URL: https://www.aapm.org/GrandChallenge/LowDoseCT/, which is for public data sharing).

The data path should look like the following:

    data_path
    ├── L067
    │   ├── quarter_3mm
    │   │       ├── L067_QD_3_1.CT.0004.0001 ~ .IMA
    │   │       ├── L067_QD_3_1.CT.0004.0002 ~ .IMA
    │   │       └── ...
    │   └── full_3mm
    │           ├── L067_FD_3_1.CT.0004.0001 ~ .IMA
    │           ├── L067_FD_3_1.CT.0004.0002 ~ .IMA
    │           └── ...
    ├── L096
    │   ├── quarter_3mm
    │   │       └── ...
    │   └── full_3mm
    │           └── ...      
    ...
    │
    └── L506 (for test)
        ├── quarter_3mm
        │       └── ...
        └── full_3mm
                └── ...     

2. Latent Diffusion Model (LDM) [2,3]

2.1. Variational Autoencoder (VAE) Training

* Go to [2_latent_diffusion_model/scripts/train_latent_embedder.py] to train the VAE; 
* [Optional] Revise Line 36 ('..', '..', 'data', 'ori_dataset') with your own path to the original dataset;
* The checkpoint will be saved in [2_latent_diffusion_model/scripts/runs/VAE/last.ckpt] (VAE checkpoint).

2.2. VAE Evaluation

* Go to [2_latent_diffusion_model/scripts/evaluate_latent_embedder.py] to evaluate the performance of the VAE;
* [Optional] Revise Line 43 ('..', '..', 'data', 'ori_dataset') with your own path to the original dataset.
* [Optional] Revise Line 54 ('..', '..', 'weights', 'VAE', 'last.ckpt') with your own path to the VAE checkpoint.

2.3. Diffusion Model Training

* Go to [2_latent_diffusion_model/scripts/train_diffusion.py] to train the diffusion model;
* [Optional] Revise Line 31 ('..', '..', 'data', 'ori_dataset') with your own path to the original dataset;
* The checkpoint will be saved in [2_latent_diffusion_model/scripts/runs/LDM/last.ckpt] (LDM checkpoint).

2.4. Dataset Synthesis

* Go to [2_latent_diffusion_model/scripts/sample_dataset.py] to construct a synthetic dataset;
* Revise Line 30 (n_samples) with your specified number of samples; 
* [Optional] Revise Line 20 ('..', '..', 'weights', 'LDM', 'last.ckpt') with your own path to the LDM checkpoint.
* [Optional] Revise Line 25 ('..', '..', 'data', 'syn_dataset') with your own path to save the synthetic dataset.

2.5. Diffusion Model Evaluation

* Go to [2_latent_diffusion_model/scripts/evaluate_images.py] to calculate the FID of the samples;
* [Optional] Revise Line 39 ('..', '..', 'data', 'ori_dataset') with your own path to the original dataset;
* [Optional] Revise Line 46 ('..', '..', 'data', 'syn_dataset') with your own path to the synthetic dataset.

3. Privacy Enhancement 

3.1. Dictionary Construction [4]

* Go to [3_privacy_enhancement/generate_dictionary/main.py] to construct the dictionary;
* Choose the original or synthetic dataset in Line 14;
* The corresponding image path will be saved in [3_privacy_enhancement\generate_dictionary\dictionary\ori.txt or syn.txt];
* The dictionary will be saved in [3_privacy_enhancement\generate_dictionary\dictionary\ori_dictionary.npy or syn_dictionary.npy].

3.2. Privacy Enhancement

* Go to [3_privacy_enhancement/privacy_enhancement.m] to construct the privacy enhanced dataset;
* Revise Line 30 ('..', 'data', 'en_dataset') with your own path to save the privacy enhanced dataset;
* [Optional] Revise Line 16 (thres) to set the threshold.

3.3. Security Evaluation

* Go to [3_privacy_enhancement/privacy_evaluation.m] to evaluate the privacy protection;
* Revise Line 10 ('..', 'data', 'ori_dataset') with your own path to the original dataset;
* Revise Line 11 ('..', 'data', 'syn_dataset') with your synthetic dataset or privacy enhanced dataset.

4. LDCT Simulation 

* Compile [4_LDCT_simulation/ForwardProjection.cu] using "mexcuda ForwardProjection.cu" 
  and [4_LDCT_simulation/BackProjection.cu] using "mexcuda BackProjection.cu" in MATLAB;
* Go to [4_LDCT_simulation/ldct_simulation.m] to simulate low-dose sinograms and images;
* Choose the original or synthetic dataset in Line 14;
* [Optional] Revise Line 16 ('..', 'data', [dataset,'_ldct']) with your own path to save low-dose CT images;
* [Optional] Revise Line 17 ('..', 'data', [dataset,'_sino']) with your own path to save low-dose CT sinograms;
* [Optional] Revise Lines 18 ('..', 'data', 'test_ldct') and 19 ('..', 'data', 'test_sino') with your path of test images and sinograms.

5. SOTA LDCT Denoising [5] 

5.1. RED-CNN [6]

* Go to [5_SOTA_LDCT_denoising/RED-CNN/main.py] to train and test RED-CNN;
* Revise Line 47 (default='train') to choose the model for training or testing;
* Revise Line 50 ('..', '..', 'data', 'ori_ldct') to your dataset;
* Revise Line 71 ('..', '..', 'weights', 'RED-CNN', 'ori', 'last.ckpt') with your own checkpoint for testing;
* The checkpoint will be saved in [5_SOTA_LDCT_denoising/RED-CNN/save/REDCNN_####iter.ckpt] (RED-CNN checkpoint).
 
5.2. FBPConvNet [7]

* Go to [5_SOTA_LDCT_denoising/FBPConvNet/main.py] to train and test FBPConvNet;
* Revise Line 47 (default='train') to choose the model for training or testing;
* Revise Line 50 ('..', '..', 'data', 'ori_ldct') to your dataset;
* Revise Line 71 ('..', '..', 'weights', 'FBPConvNet', 'ori', 'last.ckpt') with your own checkpoint for testing;
* The checkpoint will be saved in [5_SOTA_LDCT_denoising/FBPConvNet/save/FBPConvNet_####iter.ckpt] (FBPConvNet checkpoint).

5.3. AdaptiveNet [8], HDNet [9], LPD [10], FISTANet [11]

* Go to [5_SOTA_LDCT_denoising/demos/config.json] to configure the parameters (set "dose" to choose the dataset);
* As a exmpale, go to [5_SOTA_LDCT_denoising/demos/LPD.py]:
  Use trainer.fit() for training;
  Use trainer.test() for testing.
* Revise the last Line ('..','..','weights','AdaptiveNet','ori','last.ckpt') with your own checkpoint for testing;

5.4. SUNet [12]

* Go to [5_SOTA_LDCT_denoising/SUNet/training.yaml] to configure the parameters; 

* Go to [5_SOTA_LDCT_denoising/SUNet/train.py] to train SUNet;
* Revise Line 50 ('..', '..', 'data', 'ori_ldct') with your dataset path;
* The checkpoint will saved in [5_SOTA_LDCT_denoising/SUNet/checkpoint/SUNet_###.pth];

* Go to [5_SOTA_LDCT_denoising/SUNet/test.py] to test SUNet;
* Revise Line 24 ('..', '..', 'weights', 'SUNet', 'syn', 'last.pth') with your saved checkpoint;
* [Optional] Revise Line 20 ('..', '..', 'data', 'test_ldct') with your test dataset path.


References

[1] 	[Online] Available: https://www.aapm.org/grandchallenge/lowdosect/
[2] 	R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer, “High-resolution image synthesis with latent diffusion models,” 
		In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10684-10695, 2022.
[3] 	G. Müller-Franzes, J. M. Niehues, F. Khader, S. T. Arasteh, C. Haarburger, C. Kuhl et al., “Diffusion Probabilistic Models 
		beat GANs on Medical Images,” arXiv preprint arXiv:2212.07501. 2022.
[4] 	K. Simonyan, A. Zisserman, "Very deep convolutional networks for large-scale image recognition," arXiv preprint arXiv:1409.1556. 2014
[5] 	W. Xia, H. Shan, G. Wang, Y. Zhang, "Physics-/model-based and data-driven methods for low-dose computed tomography: A survey," 
		IEEE Signal Processing Magazine, 40(2), 89-100.	2023.
[6] 	H. Chen, Y. Zhang, M. K. Kalra, F. Lin, Y. Chen, P. Liao et al., “Low-dose CT with a residual encoder-decoder convolutional neural network,” 
		IEEE Transactions on Medical Imaging, vol. 36, no. 12, pp. 2524-35, 2017.
[7] 	K. H. Jin, M. T. McCann, E. Froustey, M. Unser, “Deep convolutional neural network for inverse problems in imaging,” 
		IEEE Transactions on Image Processing, vol. 26, no. 9, pp. 4509-4522, 2017.
[8] 	Y. Ge, T. Su, J. Zhu, X. Deng, Q. Zhang, J. Chen, et al., “ADAPTIVE-NET: deep computed tomography reconstruction network with 
		analytical domain transformation knowledge,” Quantitative Imaging in Medicine and Surgery, vol. 10, no. 2, pp. 415-427, 2020.
[9] 	D. Hu, J. Liu, T. Lv, Q. Zhao, Y. Zhang, G. Quan et al., “Hybrid-domain neural network processing for sparse-view CT reconstruction,” 
		IEEE Transactions on Radiation and Plasma Medical Sciences, vol. 5, no. 1, pp. 88-98, 2020.
[10] 	J. Adler, O. Öktem, “Learned primal-dual reconstruction,” IEEE Transactions on Medical Imaging, vol. 37, no. 6, pp. 1322-1332, 2018.
[11] 	J. Xiang, Y. Dong, Y. Yang, “FISTA-net: Learning a fast iterative shrinkage thresholding network for inverse problems in imaging,” 
		IEEE Transactions on Medical Imaging, vol. 40, no. 5, pp. 1329-1339, 2021.
[12] 	C. Fan, T. Liu, K. Liu, “SUNet: swin transformer UNet for image denoising,” 
		In IEEE International Symposium on Circuits and Systems (ISCAS), pp. 2333-2337, 2022.
