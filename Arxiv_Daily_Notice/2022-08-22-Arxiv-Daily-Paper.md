# New submissions for Mon, 22 Aug 22
## Keyword: SLAM
There is no result 
## Keyword: odometry
### ArUco Maker based localization and Node graph approach to mapping
 - **Authors:** Abhijith Sampathkrishna
 - **Subjects:** Robotics (cs.RO)
 - **Arxiv link:** https://arxiv.org/abs/2208.09355
 - **Pdf link:** https://arxiv.org/pdf/2208.09355
 - **Abstract**
 This paper explores a method of localization and navigation of indoor mobile robots using a node graph of landmarks that are based on fiducial markers. The use of ArUco markers and their 2D orientation with respect to the camera of the robot and the distance to the markers from the camera is used to calculate the relative position of the robot as well as the relative positions of other markers. The proposed method combines aspects of beacon-based navigation and Simultaneous Localization and Mapping based navigation. The implementation of this method uses a depth camera to obtain the distance to the marker. After calculating the required orientation of the marker, it relies on odometry calculations for tracking the position after localization with respect to the marker. Using the odometry and the relative position of one marker, the robot is then localized with respect to another marker. The relative positions and orientation of the two markers are then calculated. The markers are represented as nodes and the relative distances and orientations are represented as edges connecting the nodes and a node graph can be generated that represents a map for the robot. The method was tested on a wheeled humanoid robot with the objective of having it autonomously navigate to a charging station inside a room. This objective was successfully achieved and the limitations and future improvements are briefly discussed.
## Keyword: livox
There is no result 
## Keyword: loam
There is no result 
## Keyword: lidar
There is no result 
## Keyword: loop detection
There is no result 
## Keyword: nerf
There is no result 
## Keyword: mapping
### ArUco Maker based localization and Node graph approach to mapping
 - **Authors:** Abhijith Sampathkrishna
 - **Subjects:** Robotics (cs.RO)
 - **Arxiv link:** https://arxiv.org/abs/2208.09355
 - **Pdf link:** https://arxiv.org/pdf/2208.09355
 - **Abstract**
 This paper explores a method of localization and navigation of indoor mobile robots using a node graph of landmarks that are based on fiducial markers. The use of ArUco markers and their 2D orientation with respect to the camera of the robot and the distance to the markers from the camera is used to calculate the relative position of the robot as well as the relative positions of other markers. The proposed method combines aspects of beacon-based navigation and Simultaneous Localization and Mapping based navigation. The implementation of this method uses a depth camera to obtain the distance to the marker. After calculating the required orientation of the marker, it relies on odometry calculations for tracking the position after localization with respect to the marker. Using the odometry and the relative position of one marker, the robot is then localized with respect to another marker. The relative positions and orientation of the two markers are then calculated. The markers are represented as nodes and the relative distances and orientations are represented as edges connecting the nodes and a node graph can be generated that represents a map for the robot. The method was tested on a wheeled humanoid robot with the objective of having it autonomously navigate to a charging station inside a room. This objective was successfully achieved and the limitations and future improvements are briefly discussed.
## Keyword: localization
### ArUco Maker based localization and Node graph approach to mapping
 - **Authors:** Abhijith Sampathkrishna
 - **Subjects:** Robotics (cs.RO)
 - **Arxiv link:** https://arxiv.org/abs/2208.09355
 - **Pdf link:** https://arxiv.org/pdf/2208.09355
 - **Abstract**
 This paper explores a method of localization and navigation of indoor mobile robots using a node graph of landmarks that are based on fiducial markers. The use of ArUco markers and their 2D orientation with respect to the camera of the robot and the distance to the markers from the camera is used to calculate the relative position of the robot as well as the relative positions of other markers. The proposed method combines aspects of beacon-based navigation and Simultaneous Localization and Mapping based navigation. The implementation of this method uses a depth camera to obtain the distance to the marker. After calculating the required orientation of the marker, it relies on odometry calculations for tracking the position after localization with respect to the marker. Using the odometry and the relative position of one marker, the robot is then localized with respect to another marker. The relative positions and orientation of the two markers are then calculated. The markers are represented as nodes and the relative distances and orientations are represented as edges connecting the nodes and a node graph can be generated that represents a map for the robot. The method was tested on a wheeled humanoid robot with the objective of having it autonomously navigate to a charging station inside a room. This objective was successfully achieved and the limitations and future improvements are briefly discussed.
## Keyword: transformer
### Treeformer: Dense Gradient Trees for Efficient Attention Computation
 - **Authors:** Lovish Madaan, Srinadh Bhojanapalli, Himanshu Jain, Prateek Jain
 - **Subjects:** Computation and Language (cs.CL); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2208.09015
 - **Pdf link:** https://arxiv.org/pdf/2208.09015
 - **Abstract**
 Standard inference and training with transformer based architectures scale quadratically with input sequence length. This is prohibitively large for a variety of applications especially in web-page translation, query-answering etc. Consequently, several approaches have been developed recently to speedup attention computation by enforcing different attention structures such as sparsity, low-rank, approximating attention using kernels. In this work, we view attention computation as that of nearest neighbor retrieval, and use decision tree based hierarchical navigation to reduce the retrieval cost per query token from linear in sequence length to nearly logarithmic. Based on such hierarchical navigation, we design Treeformer which can use one of two efficient attention layers -- TF-Attention and TC-Attention. TF-Attention computes the attention in a fine-grained style, while TC-Attention is a coarse attention layer which also ensures that the gradients are "dense". To optimize such challenging discrete layers, we propose a two-level bootstrapped training method. Using extensive experiments on standard NLP benchmarks, especially for long-sequences, we demonstrate that our Treeformer architecture can be almost as accurate as baseline Transformer while using 30x lesser FLOPs in the attention layer. Compared to Linformer, the accuracy can be as much as 12% higher while using similar FLOPs in the attention layer.
### VAuLT: Augmenting the Vision-and-Language Transformer with the  Propagation of Deep Language Representations
 - **Authors:** Georgios Chochlakis, Tejas Srinivasan, Jesse Thomason, Shrikanth Narayanan (University of Southern California)
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2208.09021
 - **Pdf link:** https://arxiv.org/pdf/2208.09021
 - **Abstract**
 We propose the Vision-and-Augmented-Language Transformer (VAuLT). VAuLT is an extension of the popular Vision-and-Language Transformer (ViLT), and improves performance on vision-and-language tasks that involve more complex text inputs than image captions while having minimal impact on training and inference efficiency. ViLT, importantly, enables efficient training and inference in vision-and-language tasks, achieved by using a shallow image encoder. However, it is pretrained on captioning and similar datasets, where the language input is simple, literal, and descriptive, therefore lacking linguistic diversity. So, when working with multimedia data in the wild, such as multimodal social media data (in our work, Twitter), there is a notable shift from captioning language data, as well as diversity of tasks, and we indeed find evidence that the language capacity of ViLT is lacking instead. The key insight of VAuLT is to propagate the output representations of a large language model like BERT to the language input of ViLT. We show that such a strategy significantly improves over ViLT on vision-and-language tasks involving richer language inputs and affective constructs, such as TWITTER-2015, TWITTER-2017, MVSA-Single and MVSA-Multiple, but lags behind pure reasoning tasks such as the Bloomberg Twitter Text-Image Relationship dataset. We have released the code for all our experiments at https://github.com/gchochla/VAuLT.
### Improved Image Classification with Token Fusion
 - **Authors:** Keong Hun Choi, Jin Woo Kim, Yao Wang, Jong Eun Ha
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2208.09183
 - **Pdf link:** https://arxiv.org/pdf/2208.09183
 - **Abstract**
 In this paper, we propose a method using the fusion of CNN and transformer structure to improve image classification performance. In the case of CNN, information about a local area on an image can be extracted well, but there is a limit to the extraction of global information. On the other hand, the transformer has an advantage in relatively global extraction, but has a disadvantage in that it requires a lot of memory for local feature value extraction. In the case of an image, it is converted into a feature map through CNN, and each feature map's pixel is considered a token. At the same time, the image is divided into patch areas and then fused with the transformer method that views them as tokens. For the fusion of tokens with two different characteristics, we propose three methods: (1) late token fusion with parallel structure, (2) early token fusion, (3) token fusion in a layer by layer. In an experiment using ImageNet 1k, the proposed method shows the best classification performance.
### SoMoFormer: Social-Aware Motion Transformer for Multi-Person Motion  Prediction
 - **Authors:** Xiaogang Peng, Yaodi Shen, Haoran Wang, Binling Nie, Yigang Wang, Zizhao Wu
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2208.09224
 - **Pdf link:** https://arxiv.org/pdf/2208.09224
 - **Abstract**
 Multi-person motion prediction remains a challenging problem, especially in the joint representation learning of individual motion and social interactions. Most prior methods only involve learning local pose dynamics for individual motion (without global body trajectory) and also struggle to capture complex interaction dependencies for social interactions. In this paper, we propose a novel Social-Aware Motion Transformer (SoMoFormer) to effectively model individual motion and social interactions in a joint manner. Specifically, SoMoFormer extracts motion features from sub-sequences in displacement trajectory space to effectively learn both local and global pose dynamics for each individual. In addition, we devise a novel social-aware motion attention mechanism in SoMoFormer to further optimize dynamics representations and capture interaction dependencies simultaneously via motion similarity calculation across time and social dimensions. On both short- and long-term horizons, we empirically evaluate our framework on multi-person motion datasets and demonstrate that our method greatly outperforms state-of-the-art methods of single- and multi-person motion prediction. Code will be made publicly available upon acceptance.
### Pseudo-Labels Are All You Need
 - **Authors:** Bogdan Kostić, Mathis Lucka, Julian Risch
 - **Subjects:** Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2208.09243
 - **Pdf link:** https://arxiv.org/pdf/2208.09243
 - **Abstract**
 Automatically estimating the complexity of texts for readers has a variety of applications, such as recommending texts with an appropriate complexity level to language learners or supporting the evaluation of text simplification approaches. In this paper, we present our submission to the Text Complexity DE Challenge 2022, a regression task where the goal is to predict the complexity of a German sentence for German learners at level B. Our approach relies on more than 220,000 pseudo-labels created from the German Wikipedia and other corpora to train Transformer-based models, and refrains from any feature engineering or any additional, labeled data. We find that the pseudo-label-based approach gives impressive results yet requires little to no adjustment to the specific task and therefore could be easily adapted to other domains and tasks.
### Diverse Video Captioning by Adaptive Spatio-temporal Attention
 - **Authors:** Zohreh Ghaderi, Leonard Salewski, Hendrik P. A. Lensch
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2208.09266
 - **Pdf link:** https://arxiv.org/pdf/2208.09266
 - **Abstract**
 To generate proper captions for videos, the inference needs to identify relevant concepts and pay attention to the spatial relationships between them as well as to the temporal development in the clip. Our end-to-end encoder-decoder video captioning framework incorporates two transformer-based architectures, an adapted transformer for a single joint spatio-temporal video analysis as well as a self-attention-based decoder for advanced text generation. Furthermore, we introduce an adaptive frame selection scheme to reduce the number of required incoming frames while maintaining the relevant content when training both transformers. Additionally, we estimate semantic concepts relevant for video captioning by aggregating all ground truth captions of each sample. Our approach achieves state-of-the-art results on the MSVD, as well as on the large-scale MSR-VTT and the VATEX benchmark datasets considering multiple Natural Language Generation (NLG) metrics. Additional evaluations on diversity scores highlight the expressiveness and diversity in the structure of our generated captions.
### Expressing Multivariate Time Series as Graphs with Time Series Attention  Transformer
 - **Authors:** William T. Ng, K. Siu, Albert C. Cheung, Michael K. Ng
 - **Subjects:** Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Dynamical Systems (math.DS); Representation Theory (math.RT)
 - **Arxiv link:** https://arxiv.org/abs/2208.09300
 - **Pdf link:** https://arxiv.org/pdf/2208.09300
 - **Abstract**
 A reliable and efficient representation of multivariate time series is crucial in various downstream machine learning tasks. In multivariate time series forecasting, each variable depends on its historical values and there are inter-dependencies among variables as well. Models have to be designed to capture both intra- and inter-relationships among the time series. To move towards this goal, we propose the Time Series Attention Transformer (TSAT) for multivariate time series representation learning. Using TSAT, we represent both temporal information and inter-dependencies of multivariate time series in terms of edge-enhanced dynamic graphs. The intra-series correlations are represented by nodes in a dynamic graph; a self-attention mechanism is modified to capture the inter-series correlations by using the super-empirical mode decomposition (SMD) module. We applied the embedded dynamic graphs to times series forecasting problems, including two real-world datasets and two benchmark datasets. Extensive experiments show that TSAT clearly outerperforms six state-of-the-art baseline methods in various forecasting horizons. We further visualize the embedded dynamic graphs to illustrate the graph representation power of TSAT. We share our code at https://github.com/RadiantResearch/TSAT.
### Dance Style Transfer with Cross-modal Transformer
 - **Authors:** Wenjie Yin, Hang Yin, Kim Baraka, Danica Kragic, Mårten Björkman
 - **Subjects:** Machine Learning (cs.LG); Multimedia (cs.MM)
 - **Arxiv link:** https://arxiv.org/abs/2208.09406
 - **Pdf link:** https://arxiv.org/pdf/2208.09406
 - **Abstract**
 We present CycleDance, a dance style transfer system to transform an existing motion clip in one dance style to a motion clip in another dance style while attempting to preserve motion context of the dance. Our method extends an existing CycleGAN architecture for modeling audio sequences and integrates multimodal transformer encoders to account for music context. We adopt sequence length-based curriculum learning to stabilize training. Our approach captures rich and long-term intra-relations between motion frames, which is a common challenge in motion transfer and synthesis work. We further introduce new metrics for gauging transfer strength and content preservation in the context of dance movements. We perform an extensive ablation study as well as a human study including 30 participants with 5 or more years of dance experience. The results demonstrate that CycleDance generates realistic movements with the target style, significantly outperforming the baseline CycleGAN on naturalness, transfer strength, and content preservation.
## Keyword: autonomous driving
### Single-Stage Open-world Instance Segmentation with Cross-task  Consistency Regularization
 - **Authors:** Xizhe Xue, Dongdong Yu, Lingqiao Liu, Yu Liu, Ying Li, Zehuan Yuan, Ping Song, Mike Zheng Shou
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2208.09023
 - **Pdf link:** https://arxiv.org/pdf/2208.09023
 - **Abstract**
 Open-world instance segmentation (OWIS) aims to segment class-agnostic instances from images, which has a wide range of real-world applications such as autonomous driving. Most existing approaches follow a two-stage pipeline: performing class-agnostic detection first and then class-specific mask segmentation. In contrast, this paper proposes a single-stage framework to produce a mask for each instance directly. Also, instance mask annotations could be noisy in the existing datasets; to overcome this issue, we introduce a new regularization loss. Specifically, we first train an extra branch to perform an auxiliary task of predicting foreground regions (i.e. regions belonging to any object instance), and then encourage the prediction from the auxiliary branch to be consistent with the predictions of the instance masks. The key insight is that such a cross-task consistency loss could act as an error-correcting mechanism to combat the errors in annotations. Further, we discover that the proposed cross-task consistency loss can be applied to images without any annotation, lending itself to a semi-supervised learning method. Through extensive experiments, we demonstrate that the proposed method can achieve impressive results in both fully-supervised and semi-supervised settings. Compared to SOTA methods, the proposed method significantly improves the $AP_{100}$ score by 4.75\% in UVO$\rightarrow$UVO setting and 4.05\% in COCO$\rightarrow$UVO setting. In the case of semi-supervised learning, our model learned with only 30\% labeled data, even outperforms its fully-supervised counterpart with 50\% labeled data. The code will be released soon.
### Real-Time Robust Video Object Detection System Against Physical-World  Adversarial Attacks
 - **Authors:** Husheng Han, Xing Hu, Kaidi Xu, Pucheng Dang, Ying Wang, Yongwei Zhao, Zidong Du, Qi Guo, Yanzhi Yang, Tianshi Chen
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV); Hardware Architecture (cs.AR); Cryptography and Security (cs.CR); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2208.09195
 - **Pdf link:** https://arxiv.org/pdf/2208.09195
 - **Abstract**
 DNN-based video object detection (VOD) powers autonomous driving and video surveillance industries with rising importance and promising opportunities. However, adversarial patch attack yields huge concern in live vision tasks because of its practicality, feasibility, and powerful attack effectiveness. This work proposes Themis, a software/hardware system to defend against adversarial patches for real-time robust video object detection. We observe that adversarial patches exhibit extremely localized superficial feature importance in a small region with non-robust predictions, and thus propose the adversarial region detection algorithm for adversarial effect elimination. Themis also proposes a systematic design to efficiently support the algorithm by eliminating redundant computations and memory traffics. Experimental results show that the proposed methodology can effectively recover the system from the adversarial attack with negligible hardware overhead.
### PersDet: Monocular 3D Detection in Perspective Bird's-Eye-View
 - **Authors:** Hongyu Zhou, Zheng Ge, Weixin Mao, Zeming Li
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2208.09394
 - **Pdf link:** https://arxiv.org/pdf/2208.09394
 - **Abstract**
 Currently, detecting 3D objects in Bird's-Eye-View (BEV) is superior to other 3D detectors for autonomous driving and robotics. However, transforming image features into BEV necessitates special operators to conduct feature sampling. These operators are not supported on many edge devices, bringing extra obstacles when deploying detectors. To address this problem, we revisit the generation of BEV representation and propose detecting objects in perspective BEV -- a new BEV representation that does not require feature sampling. We demonstrate that perspective BEV features can likewise enjoy the benefits of the BEV paradigm. Moreover, the perspective BEV improves detection performance by addressing issues caused by feature sampling. We propose PersDet for high-performance object detection in perspective BEV space based on this discovery. While implementing a simple and memory-efficient structure, PersDet outperforms existing state-of-the-art monocular methods on the nuScenes benchmark, reaching 34.6% mAP and 40.8% NDS when using ResNet-50 as the backbone.
### MonoPCNS: Monocular 3D Object Detection via Point Cloud Network  Simulation
 - **Authors:** Han Sun, Zhaoxin Fan, Zhenbo Song, Zhicheng Wang, Kejian Wu, Jianfeng Lu
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2208.09446
 - **Pdf link:** https://arxiv.org/pdf/2208.09446
 - **Abstract**
 Monocular 3D object detection is a fundamental but very important task to many applications including autonomous driving, robotic grasping and augmented reality. Existing leading methods tend to estimate the depth of the input image first, and detect the 3D object based on point cloud. This routine suffers from the inherent gap between depth estimation and object detection. Besides, the prediction error accumulation would also affect the performance. In this paper, a novel method named MonoPCNS is proposed. The insight behind introducing MonoPCNS is that we propose to simulate the feature learning behavior of a point cloud based detector for monocular detector during the training period. Hence, during inference period, the learned features and prediction would be similar to the point cloud based detector as possible. To achieve it, we propose one scene-level simulation module, one RoI-level simulation module and one response-level simulation module, which are progressively used for the detector's full feature learning and prediction pipeline. We apply our method to the famous M3D-RPN detector and CaDDN detector, conducting extensive experiments on KITTI and Waymo Open dataset. Results show that our method consistently improves the performance of different monocular detectors for a large margin without changing their network architectures. Our method finally achieves state-of-the-art performance.
### Neural Light Field Estimation for Street Scenes with Differentiable  Virtual Object Insertion
 - **Authors:** Zian Wang, Wenzheng Chen, David Acuna, Jan Kautz, Sanja Fidler
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2208.09480
 - **Pdf link:** https://arxiv.org/pdf/2208.09480
 - **Abstract**
 We consider the challenging problem of outdoor lighting estimation for the goal of photorealistic virtual object insertion into photographs. Existing works on outdoor lighting estimation typically simplify the scene lighting into an environment map which cannot capture the spatially-varying lighting effects in outdoor scenes. In this work, we propose a neural approach that estimates the 5D HDR light field from a single image, and a differentiable object insertion formulation that enables end-to-end training with image-based losses that encourage realism. Specifically, we design a hybrid lighting representation tailored to outdoor scenes, which contains an HDR sky dome that handles the extreme intensity of the sun, and a volumetric lighting representation that models the spatially-varying appearance of the surrounding scene. With the estimated lighting, our shadow-aware object insertion is fully differentiable, which enables adversarial training over the composited image to provide additional supervisory signal to the lighting prediction. We experimentally demonstrate that our hybrid lighting representation is more performant than existing outdoor lighting estimation methods. We further show the benefits of our AR object insertion in an autonomous driving application, where we obtain performance gains for a 3D object detector when trained on our augmented data.