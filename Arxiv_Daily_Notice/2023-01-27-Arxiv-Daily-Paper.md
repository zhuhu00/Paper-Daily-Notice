# New submissions for Fri, 27 Jan 23
## Keyword: SLAM
### Distributed Optimization Methods for Multi-Robot Systems: Part I -- A  Tutorial
 - **Authors:** Ola Shorinwa, Trevor Halsted, Javier Yu, Mac Schwager
 - **Subjects:** Robotics (cs.RO); Multiagent Systems (cs.MA)
 - **Arxiv link:** https://arxiv.org/abs/2301.11313
 - **Pdf link:** https://arxiv.org/pdf/2301.11313
 - **Abstract**
 Distributed optimization provides a framework for deriving distributed algorithms for a variety of multi-robot problems. This tutorial constitutes the first part of a two-part series on distributed optimization applied to multi-robot problems, which seeks to advance the application of distributed optimization in robotics. In this tutorial, we demonstrate that many canonical multi-robot problems can be cast within the distributed optimization framework, such as multi-robot simultaneous localization and planning (SLAM), multi-robot target tracking, and multi-robot task assignment problems. We identify three broad categories of distributed optimization algorithms: distributed first-order methods, distributed sequential convex programming, and the alternating direction method of multipliers (ADMM). We describe the basic structure of each category and provide representative algorithms within each category. We then work through a simulation case study of multiple drones collaboratively tracking a ground vehicle. We compare solutions to this problem using a number of different distributed optimization algorithms. In addition, we implement a distributed optimization algorithm in hardware on a network of Rasberry Pis communicating with XBee modules to illustrate robustness to the challenges of real-world communication networks.
## Keyword: odometry
There is no result 
## Keyword: livox
There is no result 
## Keyword: loam
There is no result 
## Keyword: lidar
### Learning from Mistakes: Self-Regularizing Hierarchical Semantic  Representations in Point Cloud Segmentation
 - **Authors:** Elena Camuffo, Umberto Michieli, Simone Milani
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV); Multimedia (cs.MM); Machine Learning (stat.ML)
 - **Arxiv link:** https://arxiv.org/abs/2301.11145
 - **Pdf link:** https://arxiv.org/pdf/2301.11145
 - **Abstract**
 Recent advances in autonomous robotic technologies have highlighted the growing need for precise environmental analysis. LiDAR semantic segmentation has gained attention to accomplish fine-grained scene understanding by acting directly on raw content provided by sensors. Recent solutions showed how different learning techniques can be used to improve the performance of the model, without any architectural or dataset change. Following this trend, we present a coarse-to-fine setup that LEArns from classification mistaKes (LEAK) derived from a standard model. First, classes are clustered into macro groups according to mutual prediction errors; then, the learning process is regularized by: (1) aligning class-conditional prototypical feature representation for both fine and coarse classes, (2) weighting instances with a per-class fairness index. Our LEAK approach is very general and can be seamlessly applied on top of any segmentation architecture; indeed, experimental results showed that it enables state-of-the-art performances on different architectures, datasets and tasks, while ensuring more balanced class-wise results and faster convergence.
### Light-Weight Pointcloud Representation with Sparse Gaussian Process
 - **Authors:** Mahmoud Ali, Lantao Liu
 - **Subjects:** Robotics (cs.RO)
 - **Arxiv link:** https://arxiv.org/abs/2301.11251
 - **Pdf link:** https://arxiv.org/pdf/2301.11251
 - **Abstract**
 This paper presents a framework to represent high-fidelity pointcloud sensor observations for efficient communication and storage. The proposed approach exploits Sparse Gaussian Process to encode pointcloud into a compact form. Our approach represents both the free space and the occupied space using only one model (one 2D Sparse Gaussian Process) instead of the existing two-model framework (two 3D Gaussian Mixture Models). We achieve this by proposing a variance-based sampling technique that effectively discriminates between the free and occupied space. The new representation requires less memory footprint and can be transmitted across limitedbandwidth communication channels. The framework is extensively evaluated in simulation and it is also demonstrated using a real mobile robot equipped with a 3D LiDAR. Our method results in a 70 to 100 times reduction in the communication rate compared to sending the raw pointcloud.
## Keyword: loop detection
There is no result 
## Keyword: nerf
### GeCoNeRF: Few-shot Neural Radiance Fields via Geometric Consistency
 - **Authors:** Minseop Kwak, Jiuhn Song, Seungryong Kim
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2301.10941
 - **Pdf link:** https://arxiv.org/pdf/2301.10941
 - **Abstract**
 We present a novel framework to regularize Neural Radiance Field (NeRF) in a few-shot setting with a geometry-aware consistency regularization. The proposed approach leverages a rendered depth map at unobserved viewpoint to warp sparse input images to the unobserved viewpoint and impose them as pseudo ground truths to facilitate learning of NeRF. By encouraging such geometry-aware consistency at a feature-level instead of using pixel-level reconstruction loss, we regularize the NeRF at semantic and structural levels while allowing for modeling view dependent radiance to account for color variations across viewpoints. We also propose an effective method to filter out erroneous warped solutions, along with training strategies to stabilize training during optimization. We show that our model achieves competitive results compared to state-of-the-art few-shot NeRF models. Project page is available at https://ku-cvlab.github.io/GeCoNeRF/.
### Text-To-4D Dynamic Scene Generation
 - **Authors:** Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, Yaniv Taigman
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV); Artificial Intelligence (cs.AI); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2301.11280
 - **Pdf link:** https://arxiv.org/pdf/2301.11280
 - **Abstract**
 We present MAV3D (Make-A-Video3D), a method for generating three-dimensional dynamic scenes from text descriptions. Our approach uses a 4D dynamic Neural Radiance Field (NeRF), which is optimized for scene appearance, density, and motion consistency by querying a Text-to-Video (T2V) diffusion-based model. The dynamic video output generated from the provided text can be viewed from any camera location and angle, and can be composited into any 3D environment. MAV3D does not require any 3D or 4D data and the T2V model is trained only on Text-Image pairs and unlabeled videos. We demonstrate the effectiveness of our approach using comprehensive quantitative and qualitative experiments and show an improvement over previously established internal baselines. To the best of our knowledge, our method is the first to generate 3D dynamic scenes given a text description.
## Keyword: mapping
### Learning Gradients of Convex Functions with Monotone Gradient Networks
 - **Authors:** Shreyas Chaudhari, Srinivasa Pranav, José M. F. Moura
 - **Subjects:** Machine Learning (cs.LG); Optimization and Control (math.OC)
 - **Arxiv link:** https://arxiv.org/abs/2301.10862
 - **Pdf link:** https://arxiv.org/pdf/2301.10862
 - **Abstract**
 While much effort has been devoted to deriving and studying effective convex formulations of signal processing problems, the gradients of convex functions also have critical applications ranging from gradient-based optimization to optimal transport. Recent works have explored data-driven methods for learning convex objectives, but learning their monotone gradients is seldom studied. In this work, we propose Cascaded and Modular Monotone Gradient Networks (C-MGN and M-MGN respectively), two monotone gradient neural network architectures for directly learning the gradients of convex functions. We show that our networks are simpler to train, learn monotone gradient fields more accurately, and use significantly fewer parameters than state of the art methods. We further demonstrate their ability to learn optimal transport mappings to augment driving image data.
### Reef-insight: A framework for reef habitat mapping with clustering  methods via remote sensing
 - **Authors:** Saharsh Barve, Jody Webster, Rohitash Chandra
 - **Subjects:** Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2301.10876
 - **Pdf link:** https://arxiv.org/pdf/2301.10876
 - **Abstract**
 Environmental damage has been of much concern, particularly coastal areas and the oceans given climate change and drastic effects of pollution and extreme climate events. Our present day analytical capabilities along with the advancements in information acquisition techniques such as remote sensing can be utilized for the management and study of coral reef ecosystems. In this paper, we present Reef-insight, an unsupervised machine learning framework that features advanced clustering methods and remote sensing for reef community mapping. Our framework compares different clustering methods to evaluate them for reef community mapping using remote sensing data. We evaluate four major clustering approaches such as k- means, hierarchical clustering, Gaussian mixture model, and density-based clustering based on qualitative and visual assessment. We utilise remote sensing data featuring Heron reef island region in the Great Barrier Reef of Australia. Our results indicate that clustering methods using remote sensing data can well identify benthic and geomorphic clusters that are found in reefs when compared to other studies. Our results indicate that Reef-insight can generate detailed reef community maps outlining distinct reef habitats and has the potential to enable further insights for reef restoration projects. We release our framework as open source software to enable its extension to different parts of the world
### ITstyler: Image-optimized Text-based Style Transfer
 - **Authors:** Yunpeng Bai, Jiayue Liu, Chao Dong, Chun Yuan
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2301.10916
 - **Pdf link:** https://arxiv.org/pdf/2301.10916
 - **Abstract**
 Text-based style transfer is a newly-emerging research topic that uses text information instead of style image to guide the transfer process, significantly extending the application scenario of style transfer. However, previous methods require extra time for optimization or text-image paired data, leading to limited effectiveness. In this work, we achieve a data-efficient text-based style transfer method that does not require optimization at the inference stage. Specifically, we convert text input to the style space of the pre-trained VGG network to realize a more effective style swap. We also leverage CLIP's multi-modal embedding space to learn the text-to-style mapping with the image dataset only. Our method can transfer arbitrary new styles of text input in real-time and synthesize high-quality artistic images.
### On the Mathematics of Diffusion Models
 - **Authors:** David McAllester
 - **Subjects:** Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Probability (math.PR)
 - **Arxiv link:** https://arxiv.org/abs/2301.11108
 - **Pdf link:** https://arxiv.org/pdf/2301.11108
 - **Abstract**
 This paper attempts to present the stochastic differential equations of diffusion models in a manner that is accessible to a broad audience. The diffusion process is defined over a population density in R^d. Of particular interest is a population of images. In a diffusion model one first defines a diffusion process that takes a sample from the population and gradually adds noise until only noise remains. The fundamental idea is to sample from the population by a reverse-diffusion process mapping pure noise to a population sample. The diffusion process is defined independent of any ``interpretation'' but can be analyzed using the mathematics of variational auto-encoders (the ``VAE interpretation'') or the Fokker-Planck equation (the ``score-matching intgerpretation''). Both analyses yield reverse-diffusion methods involving the score function. The Fokker-Planck analysis yields a family of reverse-diffusion SDEs parameterized by any desired level of reverse-diffusion noise including zero (deterministic reverse-diffusion). The VAE analysis yields the reverse-diffusion SDE at the same noise level as the diffusion SDE. The VAE analysis also yields a useful expression for computing the population probabilities of a given point (image). This formula for the probability of a given point does not seem to follow naturally from the Fokker-Planck analysis. Much, but apparently not all, of the mathematics presented here can be found in the literature. Attributions are given at the end of the paper.
### Neural Inverse Operators for Solving PDE Inverse Problems
 - **Authors:** Roberto Molinaro, Yunan Yang, Björn Engquist, Siddhartha Mishra
 - **Subjects:** Machine Learning (cs.LG); Mathematical Physics (math-ph); Analysis of PDEs (math.AP)
 - **Arxiv link:** https://arxiv.org/abs/2301.11167
 - **Pdf link:** https://arxiv.org/pdf/2301.11167
 - **Abstract**
 A large class of inverse problems for PDEs are only well-defined as mappings from operators to functions. Existing operator learning frameworks map functions to functions and need to be modified to learn inverse maps from data. We propose a novel architecture termed Neural Inverse Operators (NIOs) to solve these PDE inverse problems. Motivated by the underlying mathematical structure, NIO is based on a suitable composition of DeepONets and FNOs to approximate mappings from operators to functions. A variety of experiments are presented to demonstrate that NIOs significantly outperform baselines and solve PDE inverse problems robustly, accurately and are several orders of magnitude faster than existing direct and PDE-constrained optimization methods.
### Learning Good Features to Transfer Across Tasks and Domains
 - **Authors:** Pierluigi Zama Ramirez, Adriano Cardace, Luca De Luigi, Alessio Tonioni, Samuele Salti, Luigi Di Stefano
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2301.11310
 - **Pdf link:** https://arxiv.org/pdf/2301.11310
 - **Abstract**
 Availability of labelled data is the major obstacle to the deployment of deep learning algorithms for computer vision tasks in new domains. The fact that many frameworks adopted to solve different tasks share the same architecture suggests that there should be a way of reusing the knowledge learned in a specific setting to solve novel tasks with limited or no additional supervision. In this work, we first show that such knowledge can be shared across tasks by learning a mapping between task-specific deep features in a given domain. Then, we show that this mapping function, implemented by a neural network, is able to generalize to novel unseen domains. Besides, we propose a set of strategies to constrain the learned feature spaces, to ease learning and increase the generalization capability of the mapping network, thereby considerably improving the final performance of our framework. Our proposal obtains compelling results in challenging synthetic-to-real adaptation scenarios by transferring knowledge between monocular depth estimation and semantic segmentation tasks.
### Certified Interpretability Robustness for Class Activation Mapping
 - **Authors:** Alex Gu, Tsui-Wei Weng, Pin-Yu Chen, Sijia Liu, Luca Daniel
 - **Subjects:** Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2301.11324
 - **Pdf link:** https://arxiv.org/pdf/2301.11324
 - **Abstract**
 Interpreting machine learning models is challenging but crucial for ensuring the safety of deep networks in autonomous driving systems. Due to the prevalence of deep learning based perception models in autonomous vehicles, accurately interpreting their predictions is crucial. While a variety of such methods have been proposed, most are shown to lack robustness. Yet, little has been done to provide certificates for interpretability robustness. Taking a step in this direction, we present CORGI, short for Certifiably prOvable Robustness Guarantees for Interpretability mapping. CORGI is an algorithm that takes in an input image and gives a certifiable lower bound for the robustness of the top k pixels of its CAM interpretability map. We show the effectiveness of CORGI via a case study on traffic sign data, certifying lower bounds on the minimum adversarial perturbation not far from (4-5x) state-of-the-art attack methods.
## Keyword: localization
### Distributed Optimization Methods for Multi-Robot Systems: Part I -- A  Tutorial
 - **Authors:** Ola Shorinwa, Trevor Halsted, Javier Yu, Mac Schwager
 - **Subjects:** Robotics (cs.RO); Multiagent Systems (cs.MA)
 - **Arxiv link:** https://arxiv.org/abs/2301.11313
 - **Pdf link:** https://arxiv.org/pdf/2301.11313
 - **Abstract**
 Distributed optimization provides a framework for deriving distributed algorithms for a variety of multi-robot problems. This tutorial constitutes the first part of a two-part series on distributed optimization applied to multi-robot problems, which seeks to advance the application of distributed optimization in robotics. In this tutorial, we demonstrate that many canonical multi-robot problems can be cast within the distributed optimization framework, such as multi-robot simultaneous localization and planning (SLAM), multi-robot target tracking, and multi-robot task assignment problems. We identify three broad categories of distributed optimization algorithms: distributed first-order methods, distributed sequential convex programming, and the alternating direction method of multipliers (ADMM). We describe the basic structure of each category and provide representative algorithms within each category. We then work through a simulation case study of multiple drones collaboratively tracking a ground vehicle. We compare solutions to this problem using a number of different distributed optimization algorithms. In addition, we implement a distributed optimization algorithm in hardware on a network of Rasberry Pis communicating with XBee modules to illustrate robustness to the challenges of real-world communication networks.
### Cut and Learn for Unsupervised Object Detection and Instance  Segmentation
 - **Authors:** Xudong Wang, Rohit Girdhar, Stella X. Yu, Ishan Misra
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV); Artificial Intelligence (cs.AI); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2301.11320
 - **Pdf link:** https://arxiv.org/pdf/2301.11320
 - **Abstract**
 We propose Cut-and-LEaRn (CutLER), a simple approach for training unsupervised object detection and segmentation models. We leverage the property of self-supervised models to 'discover' objects without supervision and amplify it to train a state-of-the-art localization model without any human labels. CutLER first uses our proposed MaskCut approach to generate coarse masks for multiple objects in an image and then learns a detector on these masks using our robust loss function. We further improve the performance by self-training the model on its predictions. Compared to prior work, CutLER is simpler, compatible with different detection architectures, and detects multiple objects. CutLER is also a zero-shot unsupervised detector and improves detection performance AP50 by over 2.7 times on 11 benchmarks across domains like video frames, paintings, sketches, etc. With finetuning, CutLER serves as a low-shot detector surpassing MoCo-v2 by 7.3% APbox and 6.6% APmask on COCO when training with 5% labels.
## Keyword: transformer
### TranSOP: Transformer-based Multimodal Classification for Stroke  Treatment Outcome Prediction
 - **Authors:** Zeynel A. Samak, Philip Clatworthy, Majid Mirmehdi
 - **Subjects:** Image and Video Processing (eess.IV); Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2301.10829
 - **Pdf link:** https://arxiv.org/pdf/2301.10829
 - **Abstract**
 Acute ischaemic stroke, caused by an interruption in blood flow to brain tissue, is a leading cause of disability and mortality worldwide. The selection of patients for the most optimal ischaemic stroke treatment is a crucial step for a successful outcome, as the effect of treatment highly depends on the time to treatment. We propose a transformer-based multimodal network (TranSOP) for a classification approach that employs clinical metadata and imaging information, acquired on hospital admission, to predict the functional outcome of stroke treatment based on the modified Rankin Scale (mRS). This includes a fusion module to efficiently combine 3D non-contrast computed tomography (NCCT) features and clinical information. In comparative experiments using unimodal and multimodal data on the MRCLEAN dataset, we achieve a state-of-the-art AUC score of 0.85.
### Enhancing Medical Image Segmentation with TransCeption: A Multi-Scale  Feature Fusion Approach
 - **Authors:** Reza Azad, Yiwei Jia, Ehsan Khodapanah Aghdam, Julien Cohen-Adad, Dorit Merhof
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2301.10847
 - **Pdf link:** https://arxiv.org/pdf/2301.10847
 - **Abstract**
 While CNN-based methods have been the cornerstone of medical image segmentation due to their promising performance and robustness, they suffer from limitations in capturing long-range dependencies. Transformer-based approaches are currently prevailing since they enlarge the reception field to model global contextual correlation. To further extract rich representations, some extensions of the U-Net employ multi-scale feature extraction and fusion modules and obtain improved performance. Inspired by this idea, we propose TransCeption for medical image segmentation, a pure transformer-based U-shape network featured by incorporating the inception-like module into the encoder and adopting a contextual bridge for better feature fusion. The design proposed in this work is based on three core principles: (1) The patch merging module in the encoder is redesigned with ResInception Patch Merging (RIPM). Multi-branch transformer (MB transformer) adopts the same number of branches as the outputs of RIPM. Combining the two modules enables the model to capture a multi-scale representation within a single stage. (2) We construct an Intra-stage Feature Fusion (IFF) module following the MB transformer to enhance the aggregation of feature maps from all the branches and particularly focus on the interaction between the different channels of all the scales. (3) In contrast to a bridge that only contains token-wise self-attention, we propose a Dual Transformer Bridge that also includes channel-wise self-attention to exploit correlations between scales at different stages from a dual perspective. Extensive experiments on multi-organ and skin lesion segmentation tasks present the superior performance of TransCeption compared to previous work. The code is publicly available at \url{https://github.com/mindflow-institue/TransCeption}.
### Qualitative Analysis of a Graph Transformer Approach to Addressing Hate  Speech: Adapting to Dynamically Changing Content
 - **Authors:** Liam Hebert, Hong Yi Chen, Robin Cohen, Lukasz Golab
 - **Subjects:** Machine Learning (cs.LG); Computation and Language (cs.CL); Social and Information Networks (cs.SI)
 - **Arxiv link:** https://arxiv.org/abs/2301.10871
 - **Pdf link:** https://arxiv.org/pdf/2301.10871
 - **Abstract**
 Our work advances an approach for predicting hate speech in social media, drawing out the critical need to consider the discussions that follow a post to successfully detect when hateful discourse may arise. Using graph transformer networks, coupled with modelling attention and BERT-level natural language processing, our approach can capture context and anticipate upcoming anti-social behaviour. In this paper, we offer a detailed qualitative analysis of this solution for hate speech detection in social networks, leading to insights into where the method has the most impressive outcomes in comparison with competitors and identifying scenarios where there are challenges to achieving ideal performance. Included is an exploration of the kinds of posts that permeate social media today, including the use of hateful images. This suggests avenues for extending our model to be more comprehensive. A key insight is that the focus on reasoning about the concept of context positions us well to be able to support multi-modal analysis of online posts. We conclude with a reflection on how the problem we are addressing relates especially well to the theme of dynamic change, a critical concern for all AI solutions for social impact. We also comment briefly on how mental health well-being can be advanced with our work, through curated content attuned to the extent of hate in posts.
### Facial Emotion Recognition
 - **Authors:** Arpita Vats, Aman Chadha
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2301.10906
 - **Pdf link:** https://arxiv.org/pdf/2301.10906
 - **Abstract**
 We present a facial emotion recognition framework, built upon Swin vision Transformers jointly with squeeze and excitation block (SE). A transformer model based on an attention mechanism has been presented recently to address vision tasks. Our method uses a vision transformer with a Squeeze excitation block (SE) and sharpness-aware minimizer (SAM). We have used a hybrid dataset, to train our model and the AffectNet dataset to evaluate the result of our model
### Compact Transformer Tracker with Correlative Masked Modeling
 - **Authors:** Zikai Song, Run Luo, Junqing Yu, Yi-Ping Phoebe Chen, Wei Yang
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2301.10938
 - **Pdf link:** https://arxiv.org/pdf/2301.10938
 - **Abstract**
 Transformer framework has been showing superior performances in visual object tracking for its great strength in information aggregation across the template and search image with the well-known attention mechanism. Most recent advances focus on exploring attention mechanism variants for better information aggregation. We find these schemes are equivalent to or even just a subset of the basic self-attention mechanism. In this paper, we prove that the vanilla self-attention structure is sufficient for information aggregation, and structural adaption is unnecessary. The key is not the attention structure, but how to extract the discriminative feature for tracking and enhance the communication between the target and search image. Based on this finding, we adopt the basic vision transformer (ViT) architecture as our main tracker and concatenate the template and search image for feature embedding. To guide the encoder to capture the invariant feature for tracking, we attach a lightweight correlative masked decoder which reconstructs the original template and search image from the corresponding masked tokens. The correlative masked decoder serves as a plugin for the compact transform tracker and is skipped in inference. Our compact tracker uses the most simple structure which only consists of a ViT backbone and a box head, and can run at 40 fps. Extensive experiments show the proposed compact transform tracker outperforms existing approaches, including advanced attention variants, and demonstrates the sufficiency of self-attention in tracking tasks. Our method achieves state-of-the-art performance on five challenging datasets, along with the VOT2020, UAV123, LaSOT, TrackingNet, and GOT-10k benchmarks. Our project is available at https://github.com/HUSTDML/CTTrack.
### Concrat: An Automatic C-to-Rust Lock API Translator for Concurrent  Programs
 - **Authors:** Jaemin Hong, Sukyoung Ryu
 - **Subjects:** Software Engineering (cs.SE)
 - **Arxiv link:** https://arxiv.org/abs/2301.10943
 - **Pdf link:** https://arxiv.org/pdf/2301.10943
 - **Abstract**
 Concurrent programs suffer from data races. To prevent data races, programmers use locks. However, programs can eliminate data races only when they acquire and release correct locks at correct timing. The lock API of C, in which people have developed a large portion of legacy system programs, does not validate the correct use of locks. On the other hand, Rust, a recently developed system programming language, provides a lock API that guarantees the correct use of locks via type checking. This makes rewriting legacy system programs in Rust a promising way to retrofit safety into them. Unfortunately, manual C-to-Rust translation is extremely laborious due to the discrepancies between their lock APIs. Even the state-of-the-art automatic C-to-Rust translator retains the C lock API, expecting developers to replace them with the Rust lock API. In this work, we propose an automatic tool to replace the C lock API with the Rust lock API. It facilitates C-to-Rust translation of concurrent programs with less human effort than the current practice. Our tool consists of a Rust code transformer that takes a lock summary as an input and a static analyzer that efficiently generates precise lock summaries. We show that the transformer is scalable and widely applicable while preserving the semantics; it transforms 66 KLOC in 2.6 seconds and successfully handles 74% of real-world programs. We also show that the analyzer is scalable and precise; it analyzes 66 KLOC in 4.3 seconds.
### Semantic Segmentation Enhanced Transformer Model for Human Attention  Prediction
 - **Authors:** Shuo Zhang
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2301.11022
 - **Pdf link:** https://arxiv.org/pdf/2301.11022
 - **Abstract**
 Saliency Prediction aims to predict the attention distribution of human eyes given an RGB image. Most of the recent state-of-the-art methods are based on deep image feature representations from traditional CNNs. However, the traditional convolution could not capture the global features of the image well due to its small kernel size. Besides, the high-level factors which closely correlate to human visual perception, e.g., objects, color, light, etc., are not considered. Inspired by these, we propose a Transformer-based method with semantic segmentation as another learning objective. More global cues of the image could be captured by Transformer. In addition, simultaneously learning the object segmentation simulates the human visual perception, which we would verify in our investigation of human gaze control in cognitive science. We build an extra decoder for the subtask and the multiple tasks share the same Transformer encoder, forcing it to learn from multiple feature spaces. We find in practice simply adding the subtask might confuse the main task learning, hence Multi-task Attention Module is proposed to deal with the feature interaction between the multiple learning targets. Our method achieves competitive performance compared to other state-of-the-art methods.
### A benchmark for toxic comment classification on Civil Comments dataset
 - **Authors:** Corentin Duchene, Henri Jamet, Pierre Guillaume, Reda Dehak
 - **Subjects:** Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2301.11125
 - **Pdf link:** https://arxiv.org/pdf/2301.11125
 - **Abstract**
 Toxic comment detection on social media has proven to be essential for content moderation. This paper compares a wide set of different models on a highly skewed multi-label hate speech dataset. We consider inference time and several metrics to measure performance and bias in our comparison. We show that all BERTs have similar performance regardless of the size, optimizations or language used to pre-train the models. RNNs are much faster at inference than any of the BERT. BiLSTM remains a good compromise between performance and inference time. RoBERTa with Focal Loss offers the best performance on biases and AUROC. However, DistilBERT combines both good AUROC and a low inference time. All models are affected by the bias of associating identities. BERT, RNN, and XLNet are less sensitive than the CNN and Compact Convolutional Transformers.
## Keyword: autonomous driving
### Certified Interpretability Robustness for Class Activation Mapping
 - **Authors:** Alex Gu, Tsui-Wei Weng, Pin-Yu Chen, Sijia Liu, Luca Daniel
 - **Subjects:** Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2301.11324
 - **Pdf link:** https://arxiv.org/pdf/2301.11324
 - **Abstract**
 Interpreting machine learning models is challenging but crucial for ensuring the safety of deep networks in autonomous driving systems. Due to the prevalence of deep learning based perception models in autonomous vehicles, accurately interpreting their predictions is crucial. While a variety of such methods have been proposed, most are shown to lack robustness. Yet, little has been done to provide certificates for interpretability robustness. Taking a step in this direction, we present CORGI, short for Certifiably prOvable Robustness Guarantees for Interpretability mapping. CORGI is an algorithm that takes in an input image and gives a certifiable lower bound for the robustness of the top k pixels of its CAM interpretability map. We show the effectiveness of CORGI via a case study on traffic sign data, certifying lower bounds on the minimum adversarial perturbation not far from (4-5x) state-of-the-art attack methods.
