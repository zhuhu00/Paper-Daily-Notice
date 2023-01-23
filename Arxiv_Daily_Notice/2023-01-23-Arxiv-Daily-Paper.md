# New submissions for Mon, 23 Jan 23
## Keyword: SLAM
There is no result 
## Keyword: odometry
There is no result 
## Keyword: livox
There is no result 
## Keyword: loam
There is no result 
## Keyword: lidar
There is no result 
## Keyword: loop detection
There is no result 
## Keyword: nerf
### NeRF in the Palm of Your Hand: Corrective Augmentation for Robotics via  Novel-View Synthesis
 - **Authors:** Allan Zhou, Moo Jin Kim, Lirui Wang, Pete Florence, Chelsea Finn
 - **Subjects:** Machine Learning (cs.LG); Computer Vision and Pattern Recognition (cs.CV); Robotics (cs.RO)
 - **Arxiv link:** https://arxiv.org/abs/2301.08556
 - **Pdf link:** https://arxiv.org/pdf/2301.08556
 - **Abstract**
 Expert demonstrations are a rich source of supervision for training visual robotic manipulation policies, but imitation learning methods often require either a large number of demonstrations or expensive online expert supervision to learn reactive closed-loop behaviors. In this work, we introduce SPARTN (Synthetic Perturbations for Augmenting Robot Trajectories via NeRF): a fully-offline data augmentation scheme for improving robot policies that use eye-in-hand cameras. Our approach leverages neural radiance fields (NeRFs) to synthetically inject corrective noise into visual demonstrations, using NeRFs to generate perturbed viewpoints while simultaneously calculating the corrective actions. This requires no additional expert supervision or environment interaction, and distills the geometric information in NeRFs into a real-time reactive RGB-only policy. In a simulated 6-DoF visual grasping benchmark, SPARTN improves success rates by 2.8$\times$ over imitation learning without the corrective augmentations and even outperforms some methods that use online supervision. It additionally closes the gap between RGB-only and RGB-D success rates, eliminating the previous need for depth sensors. In real-world 6-DoF robotic grasping experiments from limited human demonstrations, our method improves absolute success rates by $22.5\%$ on average, including objects that are traditionally challenging for depth-based methods. See video results at \url{https://bland.website/spartn}.
## Keyword: mapping
### Blind as a bat: audible echolocation on small robots
 - **Authors:** Frederike Dümbgen, Adrien Hoffet, Mihailo Kolundžija, Adam Scholefield, Martin Vetterli
 - **Subjects:** Robotics (cs.RO)
 - **Arxiv link:** https://arxiv.org/abs/2301.08327
 - **Pdf link:** https://arxiv.org/pdf/2301.08327
 - **Abstract**
 For safe and efficient operation, mobile robots need to perceive their environment, and in particular, perform tasks such as obstacle detection, localization, and mapping. Although robots are often equipped with microphones and speakers, the audio modality is rarely used for these tasks. Compared to the localization of sound sources, for which many practical solutions exist, algorithms for active echolocation are less developed and often rely on hardware requirements that are out of reach for small robots. We propose an end-to-end pipeline for sound-based localization and mapping that is targeted at, but not limited to, robots equipped with only simple buzzers and low-end microphones. The method is model-based, runs in real time, and requires no prior calibration or training. We successfully test the algorithm on the e-puck robot with its integrated audio hardware, and on the Crazyflie drone, for which we design a reproducible audio extension deck. We achieve centimeter-level wall localization on both platforms when the robots are static during the measurement process. Even in the more challenging setting of a flying drone, we can successfully localize walls, which we demonstrate in a proof-of-concept multi-wall localization and mapping demo.
### Screen Correspondence: Mapping Interchangeable Elements between UIs
 - **Authors:** Jason Wu, Amanda Swearngin, Xiaoyi Zhang, Jeffrey Nichols, Jeffrey P. Bigham
 - **Subjects:** Human-Computer Interaction (cs.HC)
 - **Arxiv link:** https://arxiv.org/abs/2301.08372
 - **Pdf link:** https://arxiv.org/pdf/2301.08372
 - **Abstract**
 Understanding user interface (UI) functionality is a useful yet challenging task for both machines and people. In this paper, we investigate a machine learning approach for screen correspondence, which allows reasoning about UIs by mapping their elements onto previously encountered examples with known functionality and properties. We describe and implement a model that incorporates element semantics, appearance, and text to support correspondence computation without requiring any labeled examples. Through a comprehensive performance evaluation, we show that our approach improves upon baselines by incorporating multi-modal properties of UIs. Finally, we show three example applications where screen correspondence facilitates better UI understanding for humans and machines: (i) instructional overlay generation, (ii) semantic UI element search, and (iii) automated interface testing.
## Keyword: localization
### Learning ultrasound plane pose regression: assessing generalized pose  coordinates in the fetal brain
 - **Authors:** Chiara Di Vece, Maela Le Lous, Brian Dromey, Francisco Vasconcelos, Anna L David, Donald Peebles, Danail Stoyanov
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2301.08317
 - **Pdf link:** https://arxiv.org/pdf/2301.08317
 - **Abstract**
 In obstetric ultrasound (US) scanning, the learner's ability to mentally build a three-dimensional (3D) map of the fetus from a two-dimensional (2D) US image represents a significant challenge in skill acquisition. We aim to build a US plane localization system for 3D visualization, training, and guidance without integrating additional sensors. This work builds on top of our previous work, which predicts the six-dimensional (6D) pose of arbitrarily-oriented US planes slicing the fetal brain with respect to a normalized reference frame using a convolutional neural network (CNN) regression network. Here, we analyze in detail the assumptions of the normalized fetal brain reference frame and quantify its accuracy with respect to the acquisition of transventricular (TV) standard plane (SP) for fetal biometry. We investigate the impact of registration quality in the training and testing data and its subsequent effect on trained models. Finally, we introduce data augmentations and larger training sets that improve the results of our previous work, achieving median errors of 3.53 mm and 6.42 degrees for translation and rotation, respectively.
### Blind as a bat: audible echolocation on small robots
 - **Authors:** Frederike Dümbgen, Adrien Hoffet, Mihailo Kolundžija, Adam Scholefield, Martin Vetterli
 - **Subjects:** Robotics (cs.RO)
 - **Arxiv link:** https://arxiv.org/abs/2301.08327
 - **Pdf link:** https://arxiv.org/pdf/2301.08327
 - **Abstract**
 For safe and efficient operation, mobile robots need to perceive their environment, and in particular, perform tasks such as obstacle detection, localization, and mapping. Although robots are often equipped with microphones and speakers, the audio modality is rarely used for these tasks. Compared to the localization of sound sources, for which many practical solutions exist, algorithms for active echolocation are less developed and often rely on hardware requirements that are out of reach for small robots. We propose an end-to-end pipeline for sound-based localization and mapping that is targeted at, but not limited to, robots equipped with only simple buzzers and low-end microphones. The method is model-based, runs in real time, and requires no prior calibration or training. We successfully test the algorithm on the e-puck robot with its integrated audio hardware, and on the Crazyflie drone, for which we design a reproducible audio extension deck. We achieve centimeter-level wall localization on both platforms when the robots are static during the measurement process. Even in the more challenging setting of a flying drone, we can successfully localize walls, which we demonstrate in a proof-of-concept multi-wall localization and mapping demo.
### Adjoint-Based Identification of Sound Sources for Sound Reinforcement  and Source Localization
 - **Authors:** Mathias Lemke, Lewin Stein
 - **Subjects:** Sound (cs.SD); Audio and Speech Processing (eess.AS); Fluid Dynamics (physics.flu-dyn)
 - **Arxiv link:** https://arxiv.org/abs/2301.08620
 - **Pdf link:** https://arxiv.org/pdf/2301.08620
 - **Abstract**
 The identification of sound sources is a common problem in acoustics. Different parameters are sought, among these are signal and position of the sources. We present an adjoint-based approach for sound source identification, which employs computational aeroacoustic techniques. Two different applications are presented as a proof-of-concept: optimization of a sound reinforcement setup and the localization of (moving) sound sources.
## Keyword: transformer
### Which Features are Learned by CodeBert: An Empirical Study of the  BERT-based Source Code Representation Learning
 - **Authors:** Lan Zhang, Chen Cao, Zhilong Wang, Peng Liu
 - **Subjects:** Computation and Language (cs.CL); Cryptography and Security (cs.CR); Machine Learning (cs.LG); Programming Languages (cs.PL)
 - **Arxiv link:** https://arxiv.org/abs/2301.08427
 - **Pdf link:** https://arxiv.org/pdf/2301.08427
 - **Abstract**
 The Bidirectional Encoder Representations from Transformers (BERT) were proposed in the natural language process (NLP) and shows promising results. Recently researchers applied the BERT to source-code representation learning and reported some good news on several downstream tasks. However, in this paper, we illustrated that current methods cannot effectively understand the logic of source codes. The representation of source code heavily relies on the programmer-defined variable and function names. We design and implement a set of experiments to demonstrate our conjecture and provide some insights for future works.
### Accelerating Multi-Agent Planning Using Graph Transformers with Bounded  Suboptimality
 - **Authors:** Chenning Yu, Qingbiao Li, Sicun Gao, Amanda Prorok
 - **Subjects:** Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Multiagent Systems (cs.MA); Robotics (cs.RO)
 - **Arxiv link:** https://arxiv.org/abs/2301.08451
 - **Pdf link:** https://arxiv.org/pdf/2301.08451
 - **Abstract**
 Conflict-Based Search is one of the most popular methods for multi-agent path finding. Though it is complete and optimal, it does not scale well. Recent works have been proposed to accelerate it by introducing various heuristics. However, whether these heuristics can apply to non-grid-based problem settings while maintaining their effectiveness remains an open question. In this work, we find that the answer is prone to be no. To this end, we propose a learning-based component, i.e., the Graph Transformer, as a heuristic function to accelerate the planning. The proposed method is provably complete and bounded-suboptimal with any desired factor. We conduct extensive experiments on two environments with dense graphs. Results show that the proposed Graph Transformer can be trained in problem instances with relatively few agents and generalizes well to a larger number of agents, while achieving better performance than state-of-the-art methods.
### Ontology Pre-training for Poison Prediction
 - **Authors:** Martin Glauer, Fabian Neuhaus, Till Mossakowski, Janna Hastings
 - **Subjects:** Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Quantitative Methods (q-bio.QM)
 - **Arxiv link:** https://arxiv.org/abs/2301.08577
 - **Pdf link:** https://arxiv.org/pdf/2301.08577
 - **Abstract**
 Integrating human knowledge into neural networks has the potential to improve their robustness and interpretability. We have developed a novel approach to integrate knowledge from ontologies into the structure of a Transformer network which we call ontology pre-training: we train the network to predict membership in ontology classes as a way to embed the structure of the ontology into the network, and subsequently fine-tune the network for the particular prediction task. We apply this approach to a case study in predicting the potential toxicity of a small molecule based on its molecular structure, a challenging task for machine learning in life sciences chemistry. Our approach improves on the state of the art, and moreover has several additional benefits. First, we are able to show that the model learns to focus attention on more meaningful chemical groups when making predictions with ontology pre-training than without, paving a path towards greater robustness and interpretability. Second, the training time is reduced after ontology pre-training, indicating that the model is better placed to learn what matters for toxicity prediction with the ontology pre-training than without. This strategy has general applicability as a neuro-symbolic approach to embed meaningful semantics into neural networks.
### Image Memorability Prediction with Vision Transformers
 - **Authors:** Thomas Hagen, Thomas Espeseth
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2301.08647
 - **Pdf link:** https://arxiv.org/pdf/2301.08647
 - **Abstract**
 Behavioral studies have shown that the memorability of images is similar across groups of people, suggesting that memorability is a function of the intrinsic properties of images, and is unrelated to people's individual experiences and traits. Deep learning networks can be trained on such properties and be used to predict memorability in new data sets. Convolutional neural networks (CNN) have pioneered image memorability prediction, but more recently developed vision transformer (ViT) models may have the potential to yield even better predictions. In this paper, we present the ViTMem, a new memorability model based on ViT, and evaluate memorability predictions obtained by it with state-of-the-art CNN-derived models. Results showed that ViTMem performed equal to or better than state-of-the-art models on all data sets. Additional semantic level analyses revealed that ViTMem is particularly sensitive to the semantic content that drives memorability in images. We conclude that ViTMem provides a new step forward, and propose that ViT-derived models can replace CNNs for computational prediction of image memorability. Researchers, educators, advertisers, visual designers and other interested parties can leverage the model to improve the memorability of their image material.
### Holistically Explainable Vision Transformers
 - **Authors:** Moritz Böhle, Mario Fritz, Bernt Schiele
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV); Machine Learning (stat.ML)
 - **Arxiv link:** https://arxiv.org/abs/2301.08669
 - **Pdf link:** https://arxiv.org/pdf/2301.08669
 - **Abstract**
 Transformers increasingly dominate the machine learning landscape across many tasks and domains, which increases the importance for understanding their outputs. While their attention modules provide partial insight into their inner workings, the attention scores have been shown to be insufficient for explaining the models as a whole. To address this, we propose B-cos transformers, which inherently provide holistic explanations for their decisions. Specifically, we formulate each model component - such as the multi-layer perceptrons, attention layers, and the tokenisation module - to be dynamic linear, which allows us to faithfully summarise the entire transformer via a single linear transform. We apply our proposed design to Vision Transformers (ViTs) and show that the resulting models, dubbed Bcos-ViTs, are highly interpretable and perform competitively to baseline ViTs on ImageNet. Code will be made available soon.
### FlatFormer: Flattened Window Attention for Efficient Point Cloud  Transformer
 - **Authors:** Zhijian Liu, Xinyu Yang, Haotian Tang, Shang Yang, Song Han
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2301.08739
 - **Pdf link:** https://arxiv.org/pdf/2301.08739
 - **Abstract**
 Transformer, as an alternative to CNN, has been proven effective in many modalities (e.g., texts and images). For 3D point cloud transformers, existing efforts focus primarily on pushing their accuracy to the state-of-the-art level. However, their latency lags behind sparse convolution-based models (3x slower), hindering their usage in resource-constrained, latency-sensitive applications (such as autonomous driving). This inefficiency comes from point clouds' sparse and irregular nature, whereas transformers are designed for dense, regular workloads. This paper presents FlatFormer to close this latency gap by trading spatial proximity for better computational regularity. We first flatten the point cloud with window-based sorting and partition points into groups of equal sizes rather than windows of equal shapes. This effectively avoids expensive structuring and padding overheads. We then apply self-attention within groups to extract local features, alternate sorting axis to gather features from different directions, and shift windows to exchange features across groups. FlatFormer delivers state-of-the-art accuracy on Waymo Open Dataset with 4.6x speedup over (transformer-based) SST and 1.4x speedup over (sparse convolutional) CenterPoint. This is the first point cloud transformer that achieves real-time performance on edge GPUs and is faster than sparse convolutional methods while achieving on-par or even superior accuracy on large-scale benchmarks. Code to reproduce our results will be made publicly available.
## Keyword: autonomous driving
### FlatFormer: Flattened Window Attention for Efficient Point Cloud  Transformer
 - **Authors:** Zhijian Liu, Xinyu Yang, Haotian Tang, Shang Yang, Song Han
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2301.08739
 - **Pdf link:** https://arxiv.org/pdf/2301.08739
 - **Abstract**
 Transformer, as an alternative to CNN, has been proven effective in many modalities (e.g., texts and images). For 3D point cloud transformers, existing efforts focus primarily on pushing their accuracy to the state-of-the-art level. However, their latency lags behind sparse convolution-based models (3x slower), hindering their usage in resource-constrained, latency-sensitive applications (such as autonomous driving). This inefficiency comes from point clouds' sparse and irregular nature, whereas transformers are designed for dense, regular workloads. This paper presents FlatFormer to close this latency gap by trading spatial proximity for better computational regularity. We first flatten the point cloud with window-based sorting and partition points into groups of equal sizes rather than windows of equal shapes. This effectively avoids expensive structuring and padding overheads. We then apply self-attention within groups to extract local features, alternate sorting axis to gather features from different directions, and shift windows to exchange features across groups. FlatFormer delivers state-of-the-art accuracy on Waymo Open Dataset with 4.6x speedup over (transformer-based) SST and 1.4x speedup over (sparse convolutional) CenterPoint. This is the first point cloud transformer that achieves real-time performance on edge GPUs and is faster than sparse convolutional methods while achieving on-par or even superior accuracy on large-scale benchmarks. Code to reproduce our results will be made publicly available.
