# New submissions for Thu,  3 Nov 22
## Keyword: SLAM
### Ambiguity-Aware Multi-Object Pose Optimization for Visually-Assisted  Robot Manipulation
 - **Authors:** Myung-Hwan Jeon, Jeongyun Kim, Jee-Hwan Ryu, Ayoung Kim
 - **Subjects:** Robotics (cs.RO); Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2211.00960
 - **Pdf link:** https://arxiv.org/pdf/2211.00960
 - **Abstract**
 6D object pose estimation aims to infer the relative pose between the object and the camera using a single image or multiple images. Most works have focused on predicting the object pose without associated uncertainty under occlusion and structural ambiguity (symmetricity). However, these works demand prior information about shape attributes, and this condition is hardly satisfied in reality; even asymmetric objects may be symmetric under the viewpoint change. In addition, acquiring and fusing diverse sensor data is challenging when extending them to robotics applications. Tackling these limitations, we present an ambiguity-aware 6D object pose estimation network, PrimA6D++, as a generic uncertainty prediction method. The major challenges in pose estimation, such as occlusion and symmetry, can be handled in a generic manner based on the measured ambiguity of the prediction. Specifically, we devise a network to reconstruct the three rotation axis primitive images of a target object and predict the underlying uncertainty along each primitive axis. Leveraging the estimated uncertainty, we then optimize multi-object poses using visual measurements and camera poses by treating it as an object SLAM problem. The proposed method shows a significant performance improvement in T-LESS and YCB-Video datasets. We further demonstrate real-time scene recognition capability for visually-assisted robot manipulation. Our code and supplementary materials are available at https://github.com/rpmsnu/PrimA6D.
### Semantic SuperPoint: A Deep Semantic Descriptor
 - **Authors:** Gabriel S. Gama, Nícolas S. Rosa, Valdir Grassi Jr
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2211.01098
 - **Pdf link:** https://arxiv.org/pdf/2211.01098
 - **Abstract**
 Several SLAM methods benefit from the use of semantic information. Most integrate photometric methods with high-level semantics such as object detection and semantic segmentation. We propose that adding a semantic segmentation decoder in a shared encoder architecture would help the descriptor decoder learn semantic information, improving the feature extractor. This would be a more robust approach than only using high-level semantic information since it would be intrinsically learned in the descriptor and would not depend on the final quality of the semantic prediction. To add this information, we take advantage of multi-task learning methods to improve accuracy and balance the performance of each task. The proposed models are evaluated according to detection and matching metrics on the HPatches dataset. The results show that the Semantic SuperPoint model performs better than the baseline one.
## Keyword: odometry
There is no result 
## Keyword: livox
There is no result 
## Keyword: loam
There is no result 
## Keyword: lidar
### 3DMODT: Attention-Guided Affinities for Joint Detection & Tracking in 3D  Point Clouds
 - **Authors:** Jyoti Kini, Ajmal Mian, Mubarak Shah
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2211.00746
 - **Pdf link:** https://arxiv.org/pdf/2211.00746
 - **Abstract**
 We propose a method for joint detection and tracking of multiple objects in 3D point clouds, a task conventionally treated as a two-step process comprising object detection followed by data association. Our method embeds both steps into a single end-to-end trainable network eliminating the dependency on external object detectors. Our model exploits temporal information employing multiple frames to detect objects and track them in a single network, thereby making it a utilitarian formulation for real-world scenarios. Computing affinity matrix by employing features similarity across consecutive point cloud scans forms an integral part of visual tracking. We propose an attention-based refinement module to refine the affinity matrix by suppressing erroneous correspondences. The module is designed to capture the global context in affinity matrix by employing self-attention within each affinity matrix and cross-attention across a pair of affinity matrices. Unlike competing approaches, our network does not require complex post-processing algorithms, and processes raw LiDAR frames to directly output tracking results. We demonstrate the effectiveness of our method on the three tracking benchmarks: JRDB, Waymo, and KITTI. Experimental evaluations indicate the ability of our model to generalize well across datasets.
### Road Markings Segmentation from LIDAR Point Clouds using Reflectivity  Information
 - **Authors:** Novel Certad, Walter Morales-Alvarez, Cristina Olaverri-Monreal
 - **Subjects:** Robotics (cs.RO)
 - **Arxiv link:** https://arxiv.org/abs/2211.01105
 - **Pdf link:** https://arxiv.org/pdf/2211.01105
 - **Abstract**
 Lane detection algorithms are crucial for the development of autonomous vehicles technologies. The more extended approach is to use cameras as sensors. However, LIDAR sensors can cope with weather and light conditions that cameras can not. In this paper, we introduce a method to extract road markings from the reflectivity data of a 64-layers LIDAR sensor. First, a plane segmentation method along with region grow clustering was used to extract the road plane. Then we applied an adaptive thresholding based on Otsu s method and finally, we fitted line models to filter out the remaining outliers. The algorithm was tested on a test track at 60km/h and a highway at 100km/h. Results showed the algorithm was reliable and precise. There was a clear improvement when using reflectivity data in comparison to the use of the raw intensity data both of them provided by the LIDAR sensor.
### OPA-3D: Occlusion-Aware Pixel-Wise Aggregation for Monocular 3D Object  Detection
 - **Authors:** Yongzhi Su, Yan Di, Fabian Manhardt, Guangyao Zhai, Jason Rambach, Benjamin Busam, Didier Stricker, Federico Tombari
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2211.01142
 - **Pdf link:** https://arxiv.org/pdf/2211.01142
 - **Abstract**
 Despite monocular 3D object detection having recently made a significant leap forward thanks to the use of pre-trained depth estimators for pseudo-LiDAR recovery, such two-stage methods typically suffer from overfitting and are incapable of explicitly encapsulating the geometric relation between depth and object bounding box. To overcome this limitation, we instead propose OPA-3D, a single-stage, end-to-end, Occlusion-Aware Pixel-Wise Aggregation network that to jointly estimate dense scene depth with depth-bounding box residuals and object bounding boxes, allowing a two-stream detection of 3D objects, leading to significantly more robust detections. Thereby, the geometry stream denoted as the Geometry Stream, combines visible depth and depth-bounding box residuals to recover the object bounding box via explicit occlusion-aware optimization. In addition, a bounding box based geometry projection scheme is employed in an effort to enhance distance perception. The second stream, named as the Context Stream, directly regresses 3D object location and size. This novel two-stream representation further enables us to enforce cross-stream consistency terms which aligns the outputs of both streams, improving the overall performance. Extensive experiments on the public benchmark demonstrate that OPA-3D outperforms state-of-the-art methods on the main Car category, whilst keeping a real-time inference speed. We plan to release all codes and trained models soon.
### A comparison of uncertainty estimation approaches for DNN-based camera  localization
 - **Authors:** Matteo Vaghi, Augusto Luis Ballardini, Simone Fontana, Domenico Giorgio Sorrenti
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2211.01234
 - **Pdf link:** https://arxiv.org/pdf/2211.01234
 - **Abstract**
 Camera localization, i.e., camera pose regression, represents a very important task in computer vision, since it has many practical applications, such as autonomous driving. A reliable estimation of the uncertainties in camera localization is also important, as it would allow to intercept localization failures, which would be dangerous. Even though the literature presents some uncertainty estimation methods, to the best of our knowledge their effectiveness has not been thoroughly examined. This work compares the performances of three consolidated epistemic uncertainty estimation methods: Monte Carlo Dropout (MCD), Deep Ensemble (DE), and Deep Evidential Regression (DER), in the specific context of camera localization. We exploited CMRNet, a DNN approach for multi-modal image to LiDAR map registration, by modifying its internal configuration to allow for an extensive experimental activity with the three methods on the KITTI dataset. Particularly significant has been the application of DER. We achieve accurate camera localization and a calibrated uncertainty, to the point that some method can be used for detecting localization failures.
## Keyword: loop detection
There is no result 
## Keyword: nerf
There is no result 
## Keyword: mapping
### MAgNET: A Graph U-Net Architecture for Mesh-Based Simulations
 - **Authors:** Saurabh Deshpande, Jakub Lengiewicz, Stéphane P.A. Bordas
 - **Subjects:** Machine Learning (cs.LG); Computational Engineering, Finance, and Science (cs.CE)
 - **Arxiv link:** https://arxiv.org/abs/2211.00713
 - **Pdf link:** https://arxiv.org/pdf/2211.00713
 - **Abstract**
 Mesh-based approaches are fundamental to solving physics-based simulations, however, they require significant computational efforts, especially for highly non-linear problems. Deep learning techniques accelerate physics-based simulations, however, they fail to perform efficiently as the size and complexity of the problem increases. Hence in this work, we propose MAgNET: Multi-channel Aggregation Network, a novel geometric deep learning framework for performing supervised learning on mesh-based graph data. MAgNET is based on the proposed MAg (Multichannel Aggregation) operation which generalises the concept of multi-channel local operations in convolutional neural networks to arbitrary non-grid inputs. MAg can efficiently perform non-linear regression mapping for graph-structured data. MAg layers are interleaved with the proposed novel graph pooling operations to constitute a graph U-Net architecture that is robust, handles arbitrary complex meshes and scales efficiently with the size of the problem. Although not limited to the type of discretisation, we showcase the predictive capabilities of MAgNET for several non-linear finite element simulations.
### Comparision Of Adversarial And Non-Adversarial LSTM Music Generative  Models
 - **Authors:** Moseli Mots'oehli, Anna Sergeevna Bosman, Johan Pieter De Villiers
 - **Subjects:** Machine Learning (cs.LG); Symbolic Computation (cs.SC); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2211.00731
 - **Pdf link:** https://arxiv.org/pdf/2211.00731
 - **Abstract**
 Algorithmic music composition is a way of composing musical pieces with minimal to no human intervention. While recurrent neural networks are traditionally applied to many sequence-to-sequence prediction tasks, including successful implementations of music composition, their standard supervised learning approach based on input-to-output mapping leads to a lack of note variety. These models can therefore be seen as potentially unsuitable for tasks such as music generation. Generative adversarial networks learn the generative distribution of data and lead to varied samples. This work implements and compares adversarial and non-adversarial training of recurrent neural network music composers on MIDI data. The resulting music samples are evaluated by human listeners, their preferences recorded. The evaluation indicates that adversarial training produces more aesthetically pleasing music.
### Synthesizing Programs with Continuous Optimization
 - **Authors:** Shantanu Mandal, Todd A. Anderson, Javier Turek, Justin Gottschlich, Abdullah Muzahid
 - **Subjects:** Artificial Intelligence (cs.AI); Programming Languages (cs.PL)
 - **Arxiv link:** https://arxiv.org/abs/2211.00828
 - **Pdf link:** https://arxiv.org/pdf/2211.00828
 - **Abstract**
 Automatic software generation based on some specification is known as program synthesis. Most existing approaches formulate program synthesis as a search problem with discrete parameters. In this paper, we present a novel formulation of program synthesis as a continuous optimization problem and use a state-of-the-art evolutionary approach, known as Covariance Matrix Adaptation Evolution Strategy to solve the problem. We then propose a mapping scheme to convert the continuous formulation into actual programs. We compare our system, called GENESYS, with several recent program synthesis techniques (in both discrete and continuous domains) and show that GENESYS synthesizes more programs within a fixed time budget than those existing schemes. For example, for programs of length 10, GENESYS synthesizes 28% more programs than those existing schemes within the same time budget.
### DyAnNet: A Scene Dynamicity Guided Self-Trained Video Anomaly Detection  Network
 - **Authors:** Kamalakar Thakare, Yash Raghuwanshi, Debi Prosad Dogra, Heeseung Choi, Ig-Jae Kim
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2211.00882
 - **Pdf link:** https://arxiv.org/pdf/2211.00882
 - **Abstract**
 Unsupervised approaches for video anomaly detection may not perform as good as supervised approaches. However, learning unknown types of anomalies using an unsupervised approach is more practical than a supervised approach as annotation is an extra burden. In this paper, we use isolation tree-based unsupervised clustering to partition the deep feature space of the video segments. The RGB- stream generates a pseudo anomaly score and the flow stream generates a pseudo dynamicity score of a video segment. These scores are then fused using a majority voting scheme to generate preliminary bags of positive and negative segments. However, these bags may not be accurate as the scores are generated only using the current segment which does not represent the global behavior of a typical anomalous event. We then use a refinement strategy based on a cross-branch feed-forward network designed using a popular I3D network to refine both scores. The bags are then refined through a segment re-mapping strategy. The intuition of adding the dynamicity score of a segment with the anomaly score is to enhance the quality of the evidence. The method has been evaluated on three popular video anomaly datasets, i.e., UCF-Crime, CCTV-Fights, and UBI-Fights. Experimental results reveal that the proposed framework achieves competitive accuracy as compared to the state-of-the-art video anomaly detection methods.
### Gradient Knowledge Distillation for Pre-trained Language Models
 - **Authors:** Lean Wang, Lei Li, Xu Sun
 - **Subjects:** Computation and Language (cs.CL); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2211.01071
 - **Pdf link:** https://arxiv.org/pdf/2211.01071
 - **Abstract**
 Knowledge distillation (KD) is an effective framework to transfer knowledge from a large-scale teacher to a compact yet well-performing student. Previous KD practices for pre-trained language models mainly transfer knowledge by aligning instance-wise outputs between the teacher and student, while neglecting an important knowledge source, i.e., the gradient of the teacher. The gradient characterizes how the teacher responds to changes in inputs, which we assume is beneficial for the student to better approximate the underlying mapping function of the teacher. Therefore, we propose Gradient Knowledge Distillation (GKD) to incorporate the gradient alignment objective into the distillation process. Experimental results show that GKD outperforms previous KD methods regarding student performance. Further analysis shows that incorporating gradient knowledge makes the student behave more consistently with the teacher, improving the interpretability greatly.
## Keyword: localization
### Optical Channel Impulse Response-Based Localization Using An Artificial  Neural Network
 - **Authors:** Hamid Hosseinianfar, Hami Rabbani, Maite Bradnt-Pearce
 - **Subjects:** Information Theory (cs.IT); Machine Learning (cs.LG); Signal Processing (eess.SP); Optics (physics.optics)
 - **Arxiv link:** https://arxiv.org/abs/2211.00806
 - **Pdf link:** https://arxiv.org/pdf/2211.00806
 - **Abstract**
 Visible light positioning has the potential to yield sub-centimeter accuracy in indoor environments, yet conventional received signal strength (RSS)-based localization algorithms cannot achieve this because their performance degrades from optical multipath reflection. However, this part of the optical received signal is deterministic due to the often static and predictable nature of the optical wireless channel. In this paper, the performance of optical channel impulse response (OCIR)-based localization is studied using an artificial neural network (ANN) to map embedded features of the OCIR to the user equipment's location. Numerical results show that OCIR-based localization outperforms conventional RSS techniques by two orders of magnitude using only two photodetectors as anchor points. The ANN technique can take advantage of multipath features in a wide range of scenarios, from using only the DC value to relying on high-resolution time sampling that can result in sub-centimeter accuracy.
### P$^3$OVD: Fine-grained Visual-Text Prompt-Driven Self-Training for  Open-Vocabulary Object Detection
 - **Authors:** Yanxin Long, Jianhua Han, Runhui Huang, Xu Hang, Yi Zhu, Chunjing Xu, Xiaodan Liang
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2211.00849
 - **Pdf link:** https://arxiv.org/pdf/2211.00849
 - **Abstract**
 Inspired by the success of visual-language methods (VLMs) in zero-shot classification, recent works attempt to extend this line of work into object detection by leveraging the localization ability of pre-trained VLMs and generating pseudo labels for unseen classes in a self-training manner. However, since the current VLMs are usually pre-trained with aligning sentence embedding with global image embedding, the direct use of them lacks fine-grained alignment for object instances, which is the core of detection. In this paper, we propose a simple but effective Pretrain-adaPt-Pseudo labeling paradigm for Open-Vocabulary Detection (P$^3$OVD) that introduces a fine-grained visual-text prompt adapting stage to enhance the current self-training paradigm with a more powerful fine-grained alignment. During the adapting stage, we enable VLM to obtain fine-grained alignment by using learnable text prompts to resolve an auxiliary dense pixel-wise prediction task. Furthermore, we propose a visual prompt module to provide the prior task information (i.e., the categories need to be predicted) for the vision branch to better adapt the pretrained VLM to the downstream tasks. Experiments show that our method achieves the state-of-the-art performance for open-vocabulary object detection, e.g., 31.5% mAP on unseen classes of COCO.
### Deep Learning Computer Vision Algorithms for Real-time UAVs On-board  Camera Image Processing
 - **Authors:** Alessandro Palmas, Pietro Andronico
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2211.01037
 - **Pdf link:** https://arxiv.org/pdf/2211.01037
 - **Abstract**
 This paper describes how advanced deep learning based computer vision algorithms are applied to enable real-time on-board sensor processing for small UAVs. Four use cases are considered: target detection, classification and localization, road segmentation for autonomous navigation in GNSS-denied zones, human body segmentation, and human action recognition. All algorithms have been developed using state-of-the-art image processing methods based on deep neural networks. Acquisition campaigns have been carried out to collect custom datasets reflecting typical operational scenarios, where the peculiar point of view of a multi-rotor UAV is replicated. Algorithms architectures and trained models performances are reported, showing high levels of both accuracy and inference speed. Output examples and on-field videos are presented, demonstrating models operation when deployed on a GPU-powered commercial embedded device (NVIDIA Jetson Xavier) mounted on board of a custom quad-rotor, paving the way to enabling high level autonomy.
### A comparison of uncertainty estimation approaches for DNN-based camera  localization
 - **Authors:** Matteo Vaghi, Augusto Luis Ballardini, Simone Fontana, Domenico Giorgio Sorrenti
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2211.01234
 - **Pdf link:** https://arxiv.org/pdf/2211.01234
 - **Abstract**
 Camera localization, i.e., camera pose regression, represents a very important task in computer vision, since it has many practical applications, such as autonomous driving. A reliable estimation of the uncertainties in camera localization is also important, as it would allow to intercept localization failures, which would be dangerous. Even though the literature presents some uncertainty estimation methods, to the best of our knowledge their effectiveness has not been thoroughly examined. This work compares the performances of three consolidated epistemic uncertainty estimation methods: Monte Carlo Dropout (MCD), Deep Ensemble (DE), and Deep Evidential Regression (DER), in the specific context of camera localization. We exploited CMRNet, a DNN approach for multi-modal image to LiDAR map registration, by modifying its internal configuration to allow for an extensive experimental activity with the three methods on the KITTI dataset. Particularly significant has been the application of DER. We achieve accurate camera localization and a calibrated uncertainty, to the point that some method can be used for detecting localization failures.
## Keyword: transformer
### CascadeXML: Rethinking Transformers for End-to-end Multi-resolution  Training in Extreme Multi-label Classification
 - **Authors:** Siddhant Kharbanda, Atmadeep Banerjee, Erik Schultheis, Rohit Babbar
 - **Subjects:** Machine Learning (cs.LG); Computation and Language (cs.CL); Machine Learning (stat.ML)
 - **Arxiv link:** https://arxiv.org/abs/2211.00640
 - **Pdf link:** https://arxiv.org/pdf/2211.00640
 - **Abstract**
 Extreme Multi-label Text Classification (XMC) involves learning a classifier that can assign an input with a subset of most relevant labels from millions of label choices. Recent approaches, such as XR-Transformer and LightXML, leverage a transformer instance to achieve state-of-the-art performance. However, in this process, these approaches need to make various trade-offs between performance and computational requirements. A major shortcoming, as compared to the Bi-LSTM based AttentionXML, is that they fail to keep separate feature representations for each resolution in a label tree. We thus propose CascadeXML, an end-to-end multi-resolution learning pipeline, which can harness the multi-layered architecture of a transformer model for attending to different label resolutions with separate feature representations. CascadeXML significantly outperforms all existing approaches with non-trivial gains obtained on benchmark datasets consisting of up to three million labels. Code for CascadeXML will be made publicly available at \url{https://github.com/xmc-aalto/cascadexml}.
### tSF: Transformer-based Semantic Filter for Few-Shot Learning
 - **Authors:** Jinxiang Lai, Siqian Yang, Wenlong Liu, Yi Zeng, Zhongyi Huang, Wenlong Wu, Jun Liu, Bin-Bin Gao, Chengjie Wang
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2211.00868
 - **Pdf link:** https://arxiv.org/pdf/2211.00868
 - **Abstract**
 Few-Shot Learning (FSL) alleviates the data shortage challenge via embedding discriminative target-aware features among plenty seen (base) and few unseen (novel) labeled samples. Most feature embedding modules in recent FSL methods are specially designed for corresponding learning tasks (e.g., classification, segmentation, and object detection), which limits the utility of embedding features. To this end, we propose a light and universal module named transformer-based Semantic Filter (tSF), which can be applied for different FSL tasks. The proposed tSF redesigns the inputs of a transformer-based structure by a semantic filter, which not only embeds the knowledge from whole base set to novel set but also filters semantic features for target category. Furthermore, the parameters of tSF is equal to half of a standard transformer block (less than 1M). In the experiments, our tSF is able to boost the performances in different classic few-shot learning tasks (about 2% improvement), especially outperforms the state-of-the-arts on multiple benchmark datasets in few-shot classification task.
### Pop2Piano : Pop Audio-based Piano Cover Generation
 - **Authors:** Jongho Choi, Kyogu Lee
 - **Subjects:** Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2211.00895
 - **Pdf link:** https://arxiv.org/pdf/2211.00895
 - **Abstract**
 The piano cover of pop music is widely enjoyed by people. However, the generation task of the pop piano cover is still understudied. This is partly due to the lack of synchronized {Pop, Piano Cover} data pairs, which made it challenging to apply the latest data-intensive deep learning-based methods. To leverage the power of the data-driven approach, we make a large amount of paired and synchronized {pop, piano cover} data using an automated pipeline. In this paper, we present Pop2Piano, a Transformer network that generates piano covers given waveforms of pop music. To the best of our knowledge, this is the first model to directly generate a piano cover from pop audio without melody and chord extraction modules. We show that Pop2Piano trained with our dataset can generate plausible piano covers.
### WITT: A Wireless Image Transmission Transformer for Semantic  Communications
 - **Authors:** Ke Yang, Sixian Wang, Jincheng Dai, Kailin Tan, Kai Niu, Ping Zhang
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV); Information Theory (cs.IT)
 - **Arxiv link:** https://arxiv.org/abs/2211.00937
 - **Pdf link:** https://arxiv.org/pdf/2211.00937
 - **Abstract**
 In this paper, we aim to redesign the vision Transformer (ViT) as a new backbone to realize semantic image transmission, termed wireless image transmission transformer (WITT). Previous works build upon convolutional neural networks (CNNs), which are inefficient in capturing global dependencies, resulting in degraded end-to-end transmission performance especially for high-resolution images. To tackle this, the proposed WITT employs Swin Transformers as a more capable backbone to extract long-range information. Different from ViTs in image classification tasks, WITT is highly optimized for image transmission while considering the effect of the wireless channel. Specifically, we propose a spatial modulation module to scale the latent representations according to channel state information, which enhances the ability of a single model to deal with various channel conditions. As a result, extensive experiments verify that our WITT attains better performance for different image resolutions, distortion metrics, and channel conditions. The code is available at https://github.com/KeYang8/WITT.
### Processing Long Legal Documents with Pre-trained Transformers: Modding  LegalBERT and Longformer
 - **Authors:** Dimitris Mamakas, Petros Tsotsi, Ion Androutsopoulos, Ilias Chalkidis
 - **Subjects:** Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2211.00974
 - **Pdf link:** https://arxiv.org/pdf/2211.00974
 - **Abstract**
 Pre-trained Transformers currently dominate most NLP tasks. They impose, however, limits on the maximum input length (512 sub-words in BERT), which are too restrictive in the legal domain. Even sparse-attention models, such as Longformer and BigBird, which increase the maximum input length to 4,096 sub-words, severely truncate texts in three of the six datasets of LexGLUE. Simpler linear classifiers with TF-IDF features can handle texts of any length, require far less resources to train and deploy, but are usually outperformed by pre-trained Transformers. We explore two directions to cope with long legal texts: (i) modifying a Longformer warm-started from LegalBERT to handle even longer texts (up to 8,192 sub-words), and (ii) modifying LegalBERT to use TF-IDF representations. The first approach is the best in terms of performance, surpassing a hierarchical version of LegalBERT, which was the previous state of the art in LexGLUE. The second approach leads to computationally more efficient models at the expense of lower performance, but the resulting models still outperform overall a linear SVM with TF-IDF features in long legal document classification.
### Transformer-based encoder-encoder architecture for Spoken Term Detection
 - **Authors:** Jan Švec, Luboš Šmídl, Jan Lehečka
 - **Subjects:** Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2211.01089
 - **Pdf link:** https://arxiv.org/pdf/2211.01089
 - **Abstract**
 The paper presents a method for spoken term detection based on the Transformer architecture. We propose the encoder-encoder architecture employing two BERT-like encoders with additional modifications, including convolutional and upsampling layers, attention masking, and shared parameters. The encoders project a recognized hypothesis and a searched term into a shared embedding space, where the score of the putative hit is computed using the calibrated dot product. In the experiments, we used the Wav2Vec 2.0 speech recognizer, and the proposed system outperformed a baseline method based on deep LSTMs on the English and Czech STD datasets based on USC Shoah Foundation Visual History Archive (MALACH).
### UniASM: Binary Code Similarity Detection without Fine-tuning
 - **Authors:** Yeming Gu, Hui Shu, Fan Hu
 - **Subjects:** Cryptography and Security (cs.CR); Machine Learning (cs.LG); Software Engineering (cs.SE)
 - **Arxiv link:** https://arxiv.org/abs/2211.01144
 - **Pdf link:** https://arxiv.org/pdf/2211.01144
 - **Abstract**
 Binary code similarity detection (BCSD) is widely used in various binary analysis tasks such as vulnerability search, malware detection, clone detection, and patch analysis. Recent studies have shown that the learning-based binary code embedding models perform better than the traditional feature-based approaches. In this paper, we proposed a novel transformer-based binary code embedding model, named UniASM, to learn representations of the binary functions. We designed two new training tasks to make the spatial distribution of the generated vectors more uniform, which can be used directly in BCSD without any fine-tuning. In addition, we proposed a new tokenization approach for binary functions, increasing the token's semantic information while mitigating the out-of-vocabulary (OOV) problem. The experimental results show that UniASM outperforms state-of-the-art (SOTA) approaches on the evaluation dataset. We achieved the average scores of recall@1 on cross-compilers, cross-optimization-levels and cross-obfuscations are 0.72, 0.63, and 0.77, which is higher than existing SOTA baselines. In a real-world task of known vulnerability searching, UniASM outperforms all the current baselines.
### RegCLR: A Self-Supervised Framework for Tabular Representation Learning  in the Wild
 - **Authors:** Weiyao Wang, Byung-Hak Kim, Varun Ganapathi
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2211.01165
 - **Pdf link:** https://arxiv.org/pdf/2211.01165
 - **Abstract**
 Recent advances in self-supervised learning (SSL) using large models to learn visual representations from natural images are rapidly closing the gap between the results produced by fully supervised learning and those produced by SSL on downstream vision tasks. Inspired by this advancement and primarily motivated by the emergence of tabular and structured document image applications, we investigate which self-supervised pretraining objectives, architectures, and fine-tuning strategies are most effective. To address these questions, we introduce RegCLR, a new self-supervised framework that combines contrastive and regularized methods and is compatible with the standard Vision Transformer architecture. Then, RegCLR is instantiated by integrating masked autoencoders as a representative example of a contrastive method and enhanced Barlow Twins as a representative example of a regularized method with configurable input image augmentations in both branches. Several real-world table recognition scenarios (e.g., extracting tables from document images), ranging from standard Word and Latex documents to even more challenging electronic health records (EHR) computer screen images, have been shown to benefit greatly from the representations learned from this new framework, with detection average-precision (AP) improving relatively by 4.8% for Table, 11.8% for Column, and 11.1% for GUI objects over a previous fully supervised baseline on real-world EHR screen images.
### Audio Language Modeling using Perceptually-Guided Discrete  Representations
 - **Authors:** Felix Kreuk, Yaniv Taigman, Adam Polyak, Jade Copet, Gabriel Synnaeve, Alexandre Défossez, Yossi Adi
 - **Subjects:** Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2211.01223
 - **Pdf link:** https://arxiv.org/pdf/2211.01223
 - **Abstract**
 In this work, we study the task of Audio Language Modeling, in which we aim at learning probabilistic models for audio that can be used for generation and completion. We use a state-of-the-art perceptually-guided audio compression model, to encode audio to discrete representations. Next, we train a transformer-based causal language model using these representations. At inference time, we perform audio auto-completion by encoding an audio prompt as a discrete sequence, feeding it to the audio language model, sampling from the model, and synthesizing the corresponding time-domain signal. We evaluate the quality of samples generated by our method on Audioset, the largest dataset for general audio to date, and show that it is superior to the evaluated baseline audio encoders. We additionally provide an extensive analysis to better understand the trade-off between audio-quality and language-modeling capabilities. Samples:link.
### Attention-based Neural Cellular Automata
 - **Authors:** Mattie Tesfaldet, Derek Nowrouzezahrai, Christopher Pal
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV); Artificial Intelligence (cs.AI); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2211.01233
 - **Pdf link:** https://arxiv.org/pdf/2211.01233
 - **Abstract**
 Recent extensions of Cellular Automata (CA) have incorporated key ideas from modern deep learning, dramatically extending their capabilities and catalyzing a new family of Neural Cellular Automata (NCA) techniques. Inspired by Transformer-based architectures, our work presents a new class of $\textit{attention-based}$ NCAs formed using a spatially localized$\unicode{x2014}$yet globally organized$\unicode{x2014}$self-attention scheme. We introduce an instance of this class named $\textit{Vision Transformer Cellular Automata}$ (ViTCA). We present quantitative and qualitative results on denoising autoencoding across six benchmark datasets, comparing ViTCA to a U-Net, a U-Net-based CA baseline (UNetCA), and a Vision Transformer (ViT). When comparing across architectures configured to similar parameter complexity, ViTCA architectures yield superior performance across all benchmarks and for nearly every evaluation metric. We present an ablation study on various architectural configurations of ViTCA, an analysis of its effect on cell states, and an investigation on its inductive biases. Finally, we examine its learned representations via linear probes on its converged cell state hidden representations, yielding, on average, superior results when compared to our U-Net, ViT, and UNetCA baselines.
### Characterizing Intrinsic Compositionality In Transformers With Tree  Projections
 - **Authors:** Shikhar Murty, Pratyusha Sharma, Jacob Andreas, Christopher D. Manning
 - **Subjects:** Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2211.01288
 - **Pdf link:** https://arxiv.org/pdf/2211.01288
 - **Abstract**
 When trained on language data, do transformers learn some arbitrary computation that utilizes the full capacity of the architecture or do they learn a simpler, tree-like computation, hypothesized to underlie compositional meaning systems like human languages? There is an apparent tension between compositional accounts of human language understanding, which are based on a restricted bottom-up computational process, and the enormous success of neural models like transformers, which can route information arbitrarily between different parts of their input. One possibility is that these models, while extremely flexible in principle, in practice learn to interpret language hierarchically, ultimately building sentence representations close to those predictable by a bottom-up, tree-structured model. To evaluate this possibility, we describe an unsupervised and parameter-free method to \emph{functionally project} the behavior of any transformer into the space of tree-structured networks. Given an input sentence, we produce a binary tree that approximates the transformer's representation-building process and a score that captures how "tree-like" the transformer's behavior is on the input. While calculation of this score does not require training any additional models, it provably upper-bounds the fit between a transformer and any tree-structured approximation. Using this method, we show that transformers for three different tasks become more tree-like over the course of training, in some cases unsupervisedly recovering the same trees as supervised parsers. These trees, in turn, are predictive of model behavior, with more tree-like models generalizing better on tests of compositional generalization.
### A Transformer-based Framework for POI-level Social Post Geolocation
 - **Authors:** Menglin Li, Kwan Hui Lim, Teng Guo, Junhua Liu
 - **Subjects:** Information Retrieval (cs.IR); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2211.01336
 - **Pdf link:** https://arxiv.org/pdf/2211.01336
 - **Abstract**
 POI-level geo-information of social posts is critical to many location-based applications and services. However, the multi-modality, complexity and diverse nature of social media data and their platforms limit the performance of inferring such fine-grained locations and their subsequent applications. To address this issue, we present a transformer-based general framework, which builds upon pre-trained language models and considers non-textual data, for social post geolocation at the POI level. To this end, inputs are categorized to handle different social data, and an optimal combination strategy is provided for feature representations. Moreover, a uniform representation of hierarchy is proposed to learn temporal information, and a concatenated version of encodings is employed to capture feature-wise positions better. Experimental results on various social datasets demonstrate that three variants of our proposed framework outperform multiple state-of-art baselines by a large margin in terms of accuracy and distance error metrics.
## Keyword: autonomous driving
### A comparison of uncertainty estimation approaches for DNN-based camera  localization
 - **Authors:** Matteo Vaghi, Augusto Luis Ballardini, Simone Fontana, Domenico Giorgio Sorrenti
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2211.01234
 - **Pdf link:** https://arxiv.org/pdf/2211.01234
 - **Abstract**
 Camera localization, i.e., camera pose regression, represents a very important task in computer vision, since it has many practical applications, such as autonomous driving. A reliable estimation of the uncertainties in camera localization is also important, as it would allow to intercept localization failures, which would be dangerous. Even though the literature presents some uncertainty estimation methods, to the best of our knowledge their effectiveness has not been thoroughly examined. This work compares the performances of three consolidated epistemic uncertainty estimation methods: Monte Carlo Dropout (MCD), Deep Ensemble (DE), and Deep Evidential Regression (DER), in the specific context of camera localization. We exploited CMRNet, a DNN approach for multi-modal image to LiDAR map registration, by modifying its internal configuration to allow for an extensive experimental activity with the three methods on the KITTI dataset. Particularly significant has been the application of DER. We achieve accurate camera localization and a calibrated uncertainty, to the point that some method can be used for detecting localization failures.
