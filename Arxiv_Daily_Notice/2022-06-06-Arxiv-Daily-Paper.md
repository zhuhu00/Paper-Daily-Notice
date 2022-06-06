# New submissions for Mon,  6 Jun 22
## Keyword: SLAM
There is no result 
## Keyword: odometry
### OdomBeyondVision: An Indoor Multi-modal Multi-platform Odometry Dataset  Beyond the Visible Spectrum
 - **Authors:** Peize Li, Kaiwen Cai, Muhamad Risqi U. Saputra, Zhuangzhuang Dai, Chris Xiaoxuan Lu
 - **Subjects:** Robotics (cs.RO)
 - **Arxiv link:** https://arxiv.org/abs/2206.01589
 - **Pdf link:** https://arxiv.org/pdf/2206.01589
 - **Abstract**
 This paper presents a multimodal indoor odometry dataset, OdomBeyondVision, featuring multiple sensors across the different spectrum and collected with different mobile platforms. Not only does OdomBeyondVision contain the traditional navigation sensors, sensors such as IMUs, mechanical LiDAR, RGBD camera, it also includes several emerging sensors such as the single-chip mmWave radar, LWIR thermal camera and solid-state LiDAR. With the above sensors on UAV, UGV and handheld platforms, we respectively recorded the multimodal odometry data and their movement trajectories in various indoor scenes and different illumination conditions. We release the exemplar radar, radar-inertial and thermal-inertial odometry implementations to demonstrate their results for future works to compare against and improve upon. The full dataset including toolkit and documentation is publicly available at: https://github.com/MAPS-Lab/OdomBeyondVision.
## Keyword: livox
There is no result 
## Keyword: loam
There is no result 
## Keyword: lidar
### Points2NeRF: Generating Neural Radiance Fields from 3D point cloud
 - **Authors:** D. Zimny, T. Trzciński, P. Spurek
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2206.01290
 - **Pdf link:** https://arxiv.org/pdf/2206.01290
 - **Abstract**
 Contemporary registration devices for 3D visual information, such as LIDARs and various depth cameras, capture data as 3D point clouds. In turn, such clouds are challenging to be processed due to their size and complexity. Existing methods address this problem by fitting a mesh to the point cloud and rendering it instead. This approach, however, leads to the reduced fidelity of the resulting visualization and misses color information of the objects crucial in computer graphics applications. In this work, we propose to mitigate this challenge by representing 3D objects as Neural Radiance Fields (NeRFs). We leverage a hypernetwork paradigm and train the model to take a 3D point cloud with the associated color values and return a NeRF network's weights that reconstruct 3D objects from input 2D images. Our method provides efficient 3D object representation and offers several advantages over the existing approaches, including the ability to condition NeRFs and improved generalization beyond objects seen in training. The latter we also confirmed in the results of our empirical evaluation.
### OdomBeyondVision: An Indoor Multi-modal Multi-platform Odometry Dataset  Beyond the Visible Spectrum
 - **Authors:** Peize Li, Kaiwen Cai, Muhamad Risqi U. Saputra, Zhuangzhuang Dai, Chris Xiaoxuan Lu
 - **Subjects:** Robotics (cs.RO)
 - **Arxiv link:** https://arxiv.org/abs/2206.01589
 - **Pdf link:** https://arxiv.org/pdf/2206.01589
 - **Abstract**
 This paper presents a multimodal indoor odometry dataset, OdomBeyondVision, featuring multiple sensors across the different spectrum and collected with different mobile platforms. Not only does OdomBeyondVision contain the traditional navigation sensors, sensors such as IMUs, mechanical LiDAR, RGBD camera, it also includes several emerging sensors such as the single-chip mmWave radar, LWIR thermal camera and solid-state LiDAR. With the above sensors on UAV, UGV and handheld platforms, we respectively recorded the multimodal odometry data and their movement trajectories in various indoor scenes and different illumination conditions. We release the exemplar radar, radar-inertial and thermal-inertial odometry implementations to demonstrate their results for future works to compare against and improve upon. The full dataset including toolkit and documentation is publicly available at: https://github.com/MAPS-Lab/OdomBeyondVision.
## Keyword: loop detection
There is no result 
## Keyword: nerf
### Points2NeRF: Generating Neural Radiance Fields from 3D point cloud
 - **Authors:** D. Zimny, T. Trzciński, P. Spurek
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2206.01290
 - **Pdf link:** https://arxiv.org/pdf/2206.01290
 - **Abstract**
 Contemporary registration devices for 3D visual information, such as LIDARs and various depth cameras, capture data as 3D point clouds. In turn, such clouds are challenging to be processed due to their size and complexity. Existing methods address this problem by fitting a mesh to the point cloud and rendering it instead. This approach, however, leads to the reduced fidelity of the resulting visualization and misses color information of the objects crucial in computer graphics applications. In this work, we propose to mitigate this challenge by representing 3D objects as Neural Radiance Fields (NeRFs). We leverage a hypernetwork paradigm and train the model to take a 3D point cloud with the associated color values and return a NeRF network's weights that reconstruct 3D objects from input 2D images. Our method provides efficient 3D object representation and offers several advantages over the existing approaches, including the ability to condition NeRFs and improved generalization beyond objects seen in training. The latter we also confirmed in the results of our empirical evaluation.
### Reinforcement Learning with Neural Radiance Fields
 - **Authors:** Danny Driess, Ingmar Schubert, Pete Florence, Yunzhu Li, Marc Toussaint
 - **Subjects:** Machine Learning (cs.LG); Computer Vision and Pattern Recognition (cs.CV); Robotics (cs.RO)
 - **Arxiv link:** https://arxiv.org/abs/2206.01634
 - **Pdf link:** https://arxiv.org/pdf/2206.01634
 - **Abstract**
 It is a long-standing problem to find effective representations for training reinforcement learning (RL) agents. This paper demonstrates that learning state representations with supervision from Neural Radiance Fields (NeRFs) can improve the performance of RL compared to other learned representations or even low-dimensional, hand-engineered state information. Specifically, we propose to train an encoder that maps multiple image observations to a latent space describing the objects in the scene. The decoder built from a latent-conditioned NeRF serves as the supervision signal to learn the latent space. An RL algorithm then operates on the learned latent space as its state representation. We call this NeRF-RL. Our experiments indicate that NeRF as supervision leads to a latent space better suited for the downstream RL tasks involving robotic object manipulations like hanging mugs on hooks, pushing objects, or opening doors. Video: https://dannydriess.github.io/nerf-rl
## Keyword: mapping
### Entangled Residual Mappings
 - **Authors:** Mathias Lechner, Ramin Hasani, Zahra Babaiee, Radu Grosu, Daniela Rus, Thomas A. Henzinger, Sepp Hochreiter
 - **Subjects:** Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Neural and Evolutionary Computing (cs.NE)
 - **Arxiv link:** https://arxiv.org/abs/2206.01261
 - **Pdf link:** https://arxiv.org/pdf/2206.01261
 - **Abstract**
 Residual mappings have been shown to perform representation learning in the first layers and iterative feature refinement in higher layers. This interplay, combined with their stabilizing effect on the gradient norms, enables them to train very deep networks. In this paper, we take a step further and introduce entangled residual mappings to generalize the structure of the residual connections and evaluate their role in iterative learning representations. An entangled residual mapping replaces the identity skip connections with specialized entangled mappings such as orthogonal, sparse, and structural correlation matrices that share key attributes (eigenvalues, structure, and Jacobian norm) with identity mappings. We show that while entangled mappings can preserve the iterative refinement of features across various deep models, they influence the representation learning process in convolutional networks differently than attention-based models and recurrent neural networks. In general, we find that for CNNs and Vision Transformers entangled sparse mapping can help generalization while orthogonal mappings hurt performance. For recurrent networks, orthogonal residual mappings form an inductive bias for time-variant sequences, which degrades accuracy on time-invariant tasks.
### SPD domain-specific batch normalization to crack interpretable  unsupervised domain adaptation in EEG
 - **Authors:** Reinmar J Kobler, Jun-ichiro Hirayama, Qibin Zhao, Motoaki Kawanabe
 - **Subjects:** Machine Learning (cs.LG); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2206.01323
 - **Pdf link:** https://arxiv.org/pdf/2206.01323
 - **Abstract**
 Electroencephalography (EEG) provides access to neuronal dynamics non-invasively with millisecond resolution, rendering it a viable method in neuroscience and healthcare. However, its utility is limited as current EEG technology does not generalize well across domains (i.e., sessions and subjects) without expensive supervised re-calibration. Contemporary methods cast this transfer learning (TL) problem as a multi-source/-target unsupervised domain adaptation (UDA) problem and address it with deep learning or shallow, Riemannian geometry aware alignment methods. Both directions have, so far, failed to consistently close the performance gap to state-of-the-art domain-specific methods based on tangent space mapping (TSM) on the symmetric positive definite (SPD) manifold. Here, we propose a theory-based machine learning framework that enables, for the first time, learning domain-invariant TSM models in an end-to-end fashion. To achieve this, we propose a new building block for geometric deep learning, which we denote SPD domain-specific momentum batch normalization (SPDDSMBN). A SPDDSMBN layer can transform domain-specific SPD inputs into domain-invariant SPD outputs, and can be readily applied to multi-source/-target and online UDA scenarios. In extensive experiments with 6 diverse EEG brain-computer interface (BCI) datasets, we obtain state-of-the-art performance in inter-session and -subject TL with a simple, intrinsically interpretable network architecture, which we denote TSMNet.
### Long Scale Error Control in Low Light Image and Video Enhancement Using  Equivariance
 - **Authors:** Sara Aghajanzadeh, David Forsyth
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2206.01334
 - **Pdf link:** https://arxiv.org/pdf/2206.01334
 - **Abstract**
 Image frames obtained in darkness are special. Just multiplying by a constant doesn't restore the image. Shot noise, quantization effects and camera non-linearities mean that colors and relative light levels are estimated poorly. Current methods learn a mapping using real dark-bright image pairs. These are very hard to capture. A recent paper has shown that simulated data pairs produce real improvements in restoration, likely because huge volumes of simulated data are easy to obtain. In this paper, we show that respecting equivariance -- the color of a restored pixel should be the same, however the image is cropped -- produces real improvements over the state of the art for restoration. We show that a scale selection mechanism can be used to improve reconstructions. Finally, we show that our approach produces improvements on video restoration as well. Our methods are evaluated both quantitatively and qualitatively.
### Thread and Data Mapping in Software Transactional Memory: An Overview
 - **Authors:** Douglas Pereira Pasqualin, Matthias Diener, André Rauber Du Bois, Maurício Lima Pilla
 - **Subjects:** Distributed, Parallel, and Cluster Computing (cs.DC)
 - **Arxiv link:** https://arxiv.org/abs/2206.01359
 - **Pdf link:** https://arxiv.org/pdf/2206.01359
 - **Abstract**
 In current microarchitectures, due to the complex memory hierarchies and different latencies on memory accesses, thread and data mapping are important issues to improve application performance. Software transactional memory (STM) is an abstraction used for thread synchronization, replacing the use of locks in parallel programming. Regarding thread and data mapping, STM presents new challenges and mapping opportunities, since (1) STM can use different conflict detection and resolution strategies, making the behavior of the application less predictable and; (2) the STM runtime has precise information about shared data and the intensity with each thread accesses them. These unique characteristics provide many opportunities for low-overhead, but precise statistics to guide mapping strategies for STM applications. The main objective of this paper is to survey the existing work about thread and data mapping that uses solely information gathered from the STM runtime to guide thread and data mapping decisions. We also discuss future research directions within this research area.
### Adversarial Attacks on Human Vision
 - **Authors:** Victor A. Mateescu, Ivan V. Bajić
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV); Image and Video Processing (eess.IV)
 - **Arxiv link:** https://arxiv.org/abs/2206.01365
 - **Pdf link:** https://arxiv.org/pdf/2206.01365
 - **Abstract**
 This article presents an introduction to visual attention retargeting, its connection to visual saliency, the challenges associated with it, and ideas for how it can be approached. The difficulty of attention retargeting as a saliency inversion problem lies in the lack of one-to-one mapping between saliency and the image domain, in addition to the possible negative impact of saliency alterations on image aesthetics. A few approaches from recent literature to solve this challenging problem are reviewed, and several suggestions for future development are presented.
### Constraining Gaussian processes for physics-informed acoustic emission  mapping
 - **Authors:** Matthew R Jones, Timothy J Rogers, Elizabeth J Cross
 - **Subjects:** Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2206.01495
 - **Pdf link:** https://arxiv.org/pdf/2206.01495
 - **Abstract**
 The automated localisation of damage in structures is a challenging but critical ingredient in the path towards predictive or condition-based maintenance of high value structures. The use of acoustic emission time of arrival mapping is a promising approach to this challenge, but is severely hindered by the need to collect a dense set of artificial acoustic emission measurements across the structure, resulting in a lengthy and often impractical data acquisition process. In this paper, we consider the use of physics-informed Gaussian processes for learning these maps to alleviate this problem. In the approach, the Gaussian process is constrained to the physical domain such that information relating to the geometry and boundary conditions of the structure are embedded directly into the learning process, returning a model that guarantees that any predictions made satisfy physically-consistent behaviour at the boundary. A number of scenarios that arise when training measurement acquisition is limited, including where training data are sparse, and also of limited coverage over the structure of interest. Using a complex plate-like structure as an experimental case study, we show that our approach significantly reduces the burden of data collection, where it is seen that incorporation of boundary condition knowledge significantly improves predictive accuracy as training observations are reduced, particularly when training measurements are not available across all parts of the structure.
### Metrics reloaded: Pitfalls and recommendations for image analysis  validation
 - **Authors:** Lena Maier-Hein, Annika Reinke, Evangelia Christodoulou, Ben Glocker, Patrick Godau, Fabian Isensee, Jens Kleesiek, Michal Kozubek, Mauricio Reyes, Michael A. Riegler, Manuel Wiesenfarth, Michael Baumgartner, Matthias Eisenmann, Doreen Heckmann-Nötzel, A. Emre Kavur, Tim Rädsch, Minu D. Tizabi, Laura Acion, Michela Antonelli, Tal Arbel, Spyridon Bakas, Peter Bankhead, Arriel Benis, M. Jorge Cardoso, Veronika Cheplygina, Beth Cimini, Gary S. Collins, Keyvan Farahani, Bram van Ginneken, Daniel A. Hashimoto, Michael M. Hoffman, Merel Huisman, Pierre Jannin, Charles E. Kahn, Alexandros Karargyris, Alan Karthikesalingam, Hannes Kenngott, Annette Kopp-Schneider, Anna Kreshuk, Tahsin Kurc, Bennett A. Landman, Geert Litjens, Amin Madani, Klaus Maier-Hein, Anne L. Martel, Peter Mattson,  et al. (21 additional authors not shown)
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2206.01653
 - **Pdf link:** https://arxiv.org/pdf/2206.01653
 - **Abstract**
 The field of automatic biomedical image analysis crucially depends on robust and meaningful performance metrics for algorithm validation. Current metric usage, however, is often ill-informed and does not reflect the underlying domain interest. Here, we present a comprehensive framework that guides researchers towards choosing performance metrics in a problem-aware manner. Specifically, we focus on biomedical image analysis problems that can be interpreted as a classification task at image, object or pixel level. The framework first compiles domain interest-, target structure-, data set- and algorithm output-related properties of a given problem into a problem fingerprint, while also mapping it to the appropriate problem category, namely image-level classification, semantic segmentation, instance segmentation, or object detection. It then guides users through the process of selecting and applying a set of appropriate validation metrics while making them aware of potential pitfalls related to individual choices. In this paper, we describe the current status of the Metrics Reloaded recommendation framework, with the goal of obtaining constructive feedback from the image analysis community. The current version has been developed within an international consortium of more than 60 image analysis experts and will be made openly available as a user-friendly toolkit after community-driven optimization.
## Keyword: localization
There is no result 
## Keyword: transformer
### Entangled Residual Mappings
 - **Authors:** Mathias Lechner, Ramin Hasani, Zahra Babaiee, Radu Grosu, Daniela Rus, Thomas A. Henzinger, Sepp Hochreiter
 - **Subjects:** Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Neural and Evolutionary Computing (cs.NE)
 - **Arxiv link:** https://arxiv.org/abs/2206.01261
 - **Pdf link:** https://arxiv.org/pdf/2206.01261
 - **Abstract**
 Residual mappings have been shown to perform representation learning in the first layers and iterative feature refinement in higher layers. This interplay, combined with their stabilizing effect on the gradient norms, enables them to train very deep networks. In this paper, we take a step further and introduce entangled residual mappings to generalize the structure of the residual connections and evaluate their role in iterative learning representations. An entangled residual mapping replaces the identity skip connections with specialized entangled mappings such as orthogonal, sparse, and structural correlation matrices that share key attributes (eigenvalues, structure, and Jacobian norm) with identity mappings. We show that while entangled mappings can preserve the iterative refinement of features across various deep models, they influence the representation learning process in convolutional networks differently than attention-based models and recurrent neural networks. In general, we find that for CNNs and Vision Transformers entangled sparse mapping can help generalization while orthogonal mappings hurt performance. For recurrent networks, orthogonal residual mappings form an inductive bias for time-variant sequences, which degrades accuracy on time-invariant tasks.
### MMTM: Multi-Tasking Multi-Decoder Transformer for Math Word Problems
 - **Authors:** Keyur Faldu, Amit Sheth, Prashant Kikani, Darshan Patel
 - **Subjects:** Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2206.01268
 - **Pdf link:** https://arxiv.org/pdf/2206.01268
 - **Abstract**
 Recently, quite a few novel neural architectures were derived to solve math word problems by predicting expression trees. These architectures varied from seq2seq models, including encoders leveraging graph relationships combined with tree decoders. These models achieve good performance on various MWPs datasets but perform poorly when applied to an adversarial challenge dataset, SVAMP. We present a novel model MMTM that leverages multi-tasking and multi-decoder during pre-training. It creates variant tasks by deriving labels using pre-order, in-order and post-order traversal of expression trees, and uses task-specific decoders in a multi-tasking framework. We leverage transformer architectures with lower dimensionality and initialize weights from RoBERTa model. MMTM model achieves better mathematical reasoning ability and generalisability, which we demonstrate by outperforming the best state of the art baseline models from Seq2Seq, GTS, and Graph2Tree with a relative improvement of 19.4% on an adversarial challenge dataset SVAMP.
### Fair Classification via Transformer Neural Networks: Case Study of an  Educational Domain
 - **Authors:** Modar Sulaiman, Kallol Roy
 - **Subjects:** Machine Learning (cs.LG); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2206.01410
 - **Pdf link:** https://arxiv.org/pdf/2206.01410
 - **Abstract**
 Educational technologies nowadays increasingly use data and Machine Learning (ML) models. This gives the students, instructors, and administrators support and insights for the optimum policy. However, it is well acknowledged that ML models are subject to bias, which raises concern about the fairness, bias, and discrimination of using these automated ML algorithms in education and its unintended and unforeseen negative consequences. The contribution of bias during the decision-making comes from datasets used for training ML models and the model architecture. This paper presents a preliminary investigation of fairness constraint in transformer neural networks on Law School and Student-Mathematics datasets. The used transformer models transform these raw datasets into a richer representation space of natural language processing (NLP) while solving fairness classification. We have employed fairness metrics for evaluation and check the trade-off between fairness and accuracy. We have reported the various metrics of F1, SPD, EOD, and accuracy for different architectures from the transformer model class.
### Exploring Transformers for Behavioural Biometrics: A Case Study in Gait  Recognition
 - **Authors:** Paula Delgado-Santos, Ruben Tolosana, Richard Guest, Farzin Deravi, Ruben Vera-Rodriguez
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV); Human-Computer Interaction (cs.HC)
 - **Arxiv link:** https://arxiv.org/abs/2206.01441
 - **Pdf link:** https://arxiv.org/pdf/2206.01441
 - **Abstract**
 Biometrics on mobile devices has attracted a lot of attention in recent years as it is considered a user-friendly authentication method. This interest has also been motivated by the success of Deep Learning (DL). Architectures based on Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) have been established to be convenient for the task, improving the performance and robustness in comparison to traditional machine learning techniques. However, some aspects must still be revisited and improved. To the best of our knowledge, this is the first article that intends to explore and propose novel gait biometric recognition systems based on Transformers, which currently obtain state-of-the-art performance in many applications. Several state-of-the-art architectures (Vanilla, Informer, Autoformer, Block-Recurrent Transformer, and THAT) are considered in the experimental framework. In addition, new configurations of the Transformers are proposed to further increase the performance. Experiments are carried out using the two popular public databases whuGAIT and OU-ISIR. The results achieved prove the high ability of the proposed Transformer, outperforming state-of-the-art CNN and RNN architectures.
### YOLOv5s-GTB: light-weighted and improved YOLOv5s for bridge crack  detection
 - **Authors:** Xiao Ruiqiang
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2206.01498
 - **Pdf link:** https://arxiv.org/pdf/2206.01498
 - **Abstract**
 In response to the situation that the conventional bridge crack manual detection method has a large amount of human and material resources wasted, this study is aimed to propose a light-weighted, high-precision, deep learning-based bridge apparent crack recognition model that can be deployed in mobile devices' scenarios. In order to enhance the performance of YOLOv5, firstly, the data augmentation methods are supplemented, and then the YOLOv5 series algorithm is trained to select a suitable basic framework. The YOLOv5s is identified as the basic framework for the light-weighted crack detection model through experiments for comparison and validation.By replacing the traditional DarkNet backbone network of YOLOv5s with GhostNet backbone network, introducing Transformer multi-headed self-attention mechanism and bi-directional feature pyramid network (BiFPN) to replace the commonly used feature pyramid network, the improved model not only has 42% fewer parameters and faster inference response, but also significantly outperforms the original model in terms of accuracy and mAP (8.5% and 1.1% improvement, respectively). Luckily each improved part has a positive impact on the result. This paper provides a feasible idea to establish a digital operation management system in the field of highway and bridge in the future and to implement the whole life cycle structure health monitoring of civil infrastructure in China.
### Anomaly detection in surveillance videos using transformer based  attention model
 - **Authors:** Kapil Deshpande, Narinder Singh Punn, Sanjay Kumar Sonbhadra, Sonali Agarwal
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2206.01524
 - **Pdf link:** https://arxiv.org/pdf/2206.01524
 - **Abstract**
 Surveillance footage can catch a wide range of realistic anomalies. This research suggests using a weakly supervised strategy to avoid annotating anomalous segments in training videos, which is time consuming. In this approach only video level labels are used to obtain frame level anomaly scores. Weakly supervised video anomaly detection (WSVAD) suffers from the wrong identification of abnormal and normal instances during the training process. Therefore it is important to extract better quality features from the available videos. WIth this motivation, the present paper uses better quality transformer-based features named Videoswin Features followed by the attention layer based on dilated convolution and self attention to capture long and short range dependencies in temporal domain. This gives us a better understanding of available videos. The proposed framework is validated on real-world dataset i.e. ShanghaiTech Campus dataset which results in competitive performance than current state-of-the-art methods. The model and the code are available at https://github.com/kapildeshpande/Anomaly-Detection-in-Surveillance-Videos
### Neural Differential Equations for Learning to Program Neural Nets  Through Continuous Learning Rules
 - **Authors:** Kazuki Irie, Francesco Faccio, Jürgen Schmidhuber
 - **Subjects:** Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2206.01649
 - **Pdf link:** https://arxiv.org/pdf/2206.01649
 - **Abstract**
 Neural ordinary differential equations (ODEs) have attracted much attention as continuous-time counterparts of deep residual neural networks (NNs), and numerous extensions for recurrent NNs have been proposed. Since the 1980s, ODEs have also been used to derive theoretical results for NN learning rules, e.g., the famous connection between Oja's rule and principal component analysis. Such rules are typically expressed as additive iterative update processes which have straightforward ODE counterparts. Here we introduce a novel combination of learning rules and Neural ODEs to build continuous-time sequence processing nets that learn to manipulate short-term memory in rapidly changing synaptic connections of other nets. This yields continuous-time counterparts of Fast Weight Programmers and linear Transformers. Our novel models outperform the best existing Neural Controlled Differential Equation based models on various time series classification tasks, while also addressing their scalability limitations. Our code is public.
## Keyword: autonomous driving
### GINK: Graph-based Interaction-aware Kinodynamic Planning via  Reinforcement Learning for Autonomous Driving
 - **Authors:** Se-Wook Yoo, Seung-Woo Seo
 - **Subjects:** Robotics (cs.RO); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2206.01488
 - **Pdf link:** https://arxiv.org/pdf/2206.01488
 - **Abstract**
 There are many challenges in applying deep reinforcement learning (DRL) to autonomous driving in a structured environment such as an urban area. This is because the massive traffic flows moving along the road network change dynamically. It is a key factor to detect changes in the intentions of surrounding vehicles and quickly find a response strategy. In this paper, we suggest a new framework that effectively combines graph-based intention representation learning and reinforcement learning for kinodynamic planning. Specifically, the movement of dynamic agents is expressed as a graph. The spatio-temporal locality of node features is conserved and the features are aggregated by considering the interaction between adjacent nodes. We simultaneously learn motion planner and controller that share the aggregated information via a safe RL framework. We intuitively interpret a given situation with predicted trajectories to generate additional cost signals. The dense cost signals encourage the policy to be safe for dynamic risk. Moreover, by utilizing the data obtained through the direct rollout of learned policy, robust intention inference is achieved for various situations encountered in training. We set up a navigation scenario in which various situations exist by using CARLA, an urban driving simulator. The experiments show the state-of-the-art performance of our approach compared to the existing baselines.
