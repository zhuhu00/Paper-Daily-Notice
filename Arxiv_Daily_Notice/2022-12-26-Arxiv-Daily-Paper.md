# New submissions for Mon, 26 Dec 22
## Keyword: SLAM
### Implementation of a Blind navigation method in outdoors/indoors areas
 - **Authors:** Mohammad Javadian Farzaneh, Hossein Mahvash Mohammadi
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV); Robotics (cs.RO)
 - **Arxiv link:** https://arxiv.org/abs/2212.12185
 - **Pdf link:** https://arxiv.org/pdf/2212.12185
 - **Abstract**
 Based on WHO statistics, many individuals are suffering from visual problems, and their number is increasing yearly. One of the most critical needs they have is the ability to navigate safely, which is why researchers are trying to create and improve various navigation systems. This paper provides a navigation concept based on the visual slam and Yolo concepts using monocular cameras. Using the ORB-SLAM algorithm, our concept creates a map from a predefined route that a blind person most uses. Since visually impaired people are curious about their environment and, of course, to guide them properly, obstacle detection has been added to the system. As mentioned earlier, safe navigation is vital for visually impaired people, so our concept has a path-following part. This part consists of three steps: obstacle distance estimation, path deviation detection, and next-step prediction, done by monocular cameras.
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
There is no result 
## Keyword: mapping
### A Topic Modeling Approach to Classifying Open Street Map Health Clinics  and Schools in Sub-Saharan Africa
 - **Authors:** Joshua W. Anderson, Luis Iñaki Alberro Encina, Tina George Karippacheril, Jonathan Hersh, Cadence Stringer
 - **Subjects:** Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2212.12084
 - **Pdf link:** https://arxiv.org/pdf/2212.12084
 - **Abstract**
 Data deprivation, or the lack of easily available and actionable information on the well-being of individuals, is a significant challenge for the developing world and an impediment to the design and operationalization of policies intended to alleviate poverty. In this paper we explore the suitability of data derived from OpenStreetMap to proxy for the location of two crucial public services: schools and health clinics. Thanks to the efforts of thousands of digital humanitarians, online mapping repositories such as OpenStreetMap contain millions of records on buildings and other structures, delineating both their location and often their use. Unfortunately much of this data is locked in complex, unstructured text rendering it seemingly unsuitable for classifying schools or clinics. We apply a scalable, unsupervised learning method to unlabeled OpenStreetMap building data to extract the location of schools and health clinics in ten countries in Africa. We find the topic modeling approach greatly improves performance versus reliance on structured keys alone. We validate our results by comparing schools and clinics identified by our OSM method versus those identified by the WHO, and describe OSM coverage gaps more broadly.
### Unpaired Overwater Image Defogging Using Prior Map Guided CycleGAN
 - **Authors:** Yaozong Mo, Chaofeng Li, Wenqi Ren, Shaopeng Shang, Wenwu Wang, Xiao-jun Wu
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2212.12116
 - **Pdf link:** https://arxiv.org/pdf/2212.12116
 - **Abstract**
 Deep learning-based methods have achieved significant performance for image defogging. However, existing methods are mainly developed for land scenes and perform poorly when dealing with overwater foggy images, since overwater scenes typically contain large expanses of sky and water. In this work, we propose a Prior map Guided CycleGAN (PG-CycleGAN) for defogging of images with overwater scenes. To promote the recovery of the objects on water in the image, two loss functions are exploited for the network where a prior map is designed to invert the dark channel and the min-max normalization is used to suppress the sky and emphasize objects. However, due to the unpaired training set, the network may learn an under-constrained domain mapping from foggy to fog-free image, leading to artifacts and loss of details. Thus, we propose an intuitive Upscaling Inception Module (UIM) and a Long-range Residual Coarse-to-fine framework (LRC) to mitigate this issue. Extensive experiments on qualitative and quantitative comparisons demonstrate that the proposed method outperforms the state-of-the-art supervised, semi-supervised, and unsupervised defogging approaches.
### Monotonous Parameter Estimation of One Class of Nonlinearly  Parameterized Regressions without Overparameterization
 - **Authors:** Anton Glushchenko, Konstantin Lastochkin
 - **Subjects:** Systems and Control (eess.SY)
 - **Arxiv link:** https://arxiv.org/abs/2212.12184
 - **Pdf link:** https://arxiv.org/pdf/2212.12184
 - **Abstract**
 The estimation law of unknown parameters vector ${\theta}$ is proposed for one class of nonlinearly parametrized regression equations $y\left( t \right) = \Omega \left( t \right)\Theta \left( \theta \right)$. We restrict our attention to parametrizations that are widely obtained in practical scenarios when polynomials or power functions in $\theta$ are used to form $y(t)$. For them we introduce a new "linearizability" assumption that a mapping from overparametrized vector of parameters $\Theta \left( \theta \right)$ to original one $\theta$ exists in terms of standard inverse algebraic function. Under such assumption and weak requirement of the regressor finite excitation, on the basis of dynamic regressor extension and mixing technique we propose a procedure to reduce the nonlinear regression equation to the linear parameterization without application of singularity causing operations and the need to identify the overparametrized parameters vector. As a result, an estimation law with exponential convergence rate is derived, which, unlike known solutions, ($\textit{i}$) does not require a strict $\textit{P}$-monotonicity condition to be met and a priori information about $\theta$ to be known, ($\textit{ii}$) ensures elementwise monotonicity for the parameter error vector. The effectiveness of our approach is illustrated with both academic example and 2-DOF robot manipulator control problem.
### FFNeRV: Flow-Guided Frame-Wise Neural Representations for Videos
 - **Authors:** Joo Chan Lee, Daniel Rho, Jong Hwan Ko, Eunbyung Park
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2212.12294
 - **Pdf link:** https://arxiv.org/pdf/2212.12294
 - **Abstract**
 Neural fields, also known as coordinate-based or implicit neural representations, have shown a remarkable capability of representing, generating, and manipulating various forms of signals. For video representations, however, mapping pixel-wise coordinates to RGB colors has shown relatively low compression performance and slow convergence and inference speed. Frame-wise video representation, which maps a temporal coordinate to its entire frame, has recently emerged as an alternative method to represent videos, improving compression rates and encoding speed. While promising, it has still failed to reach the performance of state-of-the-art video compression algorithms. In this work, we propose FFNeRV, a novel method for incorporating flow information into frame-wise representations to exploit the temporal redundancy across the frames in videos inspired by the standard video codecs. Furthermore, we introduce a fully convolutional architecture, enabled by one-dimensional temporal grids, improving the continuity of spatial features. Experimental results show that FFNeRV yields the best performance for video compression and frame interpolation among the methods using frame-wise representations or neural fields. To reduce the model size even further, we devise a more compact convolutional architecture using the group and pointwise convolutions. With model compression techniques, including quantization-aware training and entropy coding, FFNeRV outperforms widely-used standard video codecs (H.264 and HEVC) and performs on par with state-of-the-art video compression algorithms.
### Channel charting based beamforming
 - **Authors:** Luc Le Magoarou (IRT b-com, Hypermedia, INSA Rennes, IETR), Taha Yassine (IRT b-com, Hypermedia, INSA Rennes, IETR), Stephane Paquelet (IRT b-com, Hypermedia, IETR), Matthieu Crussière (IRT b-com, Hypermedia, INSA Rennes, IETR)
 - **Subjects:** Networking and Internet Architecture (cs.NI); Machine Learning (cs.LG); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2212.12340
 - **Pdf link:** https://arxiv.org/pdf/2212.12340
 - **Abstract**
 Channel charting (CC) is an unsupervised learning method allowing to locate users relative to each other without reference. From a broader perspective, it can be viewed as a way to discover a low-dimensional latent space charting the channel manifold. In this paper, this latent modeling vision is leveraged together with a recently proposed location-based beamforming (LBB) method to show that channel charting can be used for mapping channels in space or frequency. Combining CC and LBB yields a neural network resembling an autoencoder. The proposed method is empirically assessed on a channel mapping task whose objective is to predict downlink channels from uplink channels.
### Comparison of Three Job Mapping Algorithms for Supercomputer Resource  Managers
 - **Authors:** A. V. Baranov, E. A. Kiselev, B. M. Shabanov, A. A. Sorokin, P. N. Telegin
 - **Subjects:** Performance (cs.PF)
 - **Arxiv link:** https://arxiv.org/abs/2212.12443
 - **Pdf link:** https://arxiv.org/pdf/2212.12443
 - **Abstract**
 Performance of supercomputer depends on the quality of resource manager, one of its functions is assignment of jobs to the nodes of clusters or MPP computers. Parts of parallel programs interact with each other with different intensity, and mapping of program to supercomputer nodes influence efficiency of the run. At each program run graph representing application program is to be mapped onto graph of nodes representing a subset of computer system. The both graphs are not known beforehand, hence the mapping must be done in reasonable time while scheduling resources. Three mapping algorithms were explored: parallel versions of simulated annealing, genetic and composite algorithms. A set of experimental runs with different algorithms parameters was performed, comparison of mapping quality and runtime was made, and suggestions on applicability of algorithms for resource managers were provided.
### An Exact Mapping From ReLU Networks to Spiking Neural Networks
 - **Authors:** Ana Stanojevic, Stanisław Woźniak, Guillaume Bellec, Giovanni Cherubini, Angeliki Pantazi, Wulfram Gerstner
 - **Subjects:** Neural and Evolutionary Computing (cs.NE); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2212.12522
 - **Pdf link:** https://arxiv.org/pdf/2212.12522
 - **Abstract**
 Deep spiking neural networks (SNNs) offer the promise of low-power artificial intelligence. However, training deep SNNs from scratch or converting deep artificial neural networks to SNNs without loss of performance has been a challenge. Here we propose an exact mapping from a network with Rectified Linear Units (ReLUs) to an SNN that fires exactly one spike per neuron. For our constructive proof, we assume that an arbitrary multi-layer ReLU network with or without convolutional layers, batch normalization and max pooling layers was trained to high performance on some training set. Furthermore, we assume that we have access to a representative example of input data used during training and to the exact parameters (weights and biases) of the trained ReLU network. The mapping from deep ReLU networks to SNNs causes zero percent drop in accuracy on CIFAR10, CIFAR100 and the ImageNet-like data sets Places365 and PASS. More generally our work shows that an arbitrary deep ReLU network can be replaced by an energy-efficient single-spike neural network without any loss of performance.
## Keyword: localization
### Push-the-Boundary: Boundary-aware Feature Propagation for Semantic  Segmentation of 3D Point Clouds
 - **Authors:** Shenglan Du, Nail Ibrahimli, Jantien Stoter, Julian Kooij, Liangliang Nan
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2212.12402
 - **Pdf link:** https://arxiv.org/pdf/2212.12402
 - **Abstract**
 Feedforward fully convolutional neural networks currently dominate in semantic segmentation of 3D point clouds. Despite their great success, they suffer from the loss of local information at low-level layers, posing significant challenges to accurate scene segmentation and precise object boundary delineation. Prior works either address this issue by post-processing or jointly learn object boundaries to implicitly improve feature encoding of the networks. These approaches often require additional modules which are difficult to integrate into the original architecture. To improve the segmentation near object boundaries, we propose a boundary-aware feature propagation mechanism. This mechanism is achieved by exploiting a multi-task learning framework that aims to explicitly guide the boundaries to their original locations. With one shared encoder, our network outputs (i) boundary localization, (ii) prediction of directions pointing to the object's interior, and (iii) semantic segmentation, in three parallel streams. The predicted boundaries and directions are fused to propagate the learned features to refine the segmentation. We conduct extensive experiments on the S3DIS and SensatUrban datasets against various baseline methods, demonstrating that our proposed approach yields consistent improvements by reducing boundary errors. Our code is available at https://github.com/shenglandu/PushBoundary.
## Keyword: transformer
### When are Lemons Purple? The Concept Association Bias of CLIP
 - **Authors:** Yutaro Yamada, Yingtian Tang, Ilker Yildirim
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV); Computation and Language (cs.CL); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2212.12043
 - **Pdf link:** https://arxiv.org/pdf/2212.12043
 - **Abstract**
 Large-scale vision-language models such as CLIP have shown impressive performance on zero-shot image classification and image-to-text retrieval. However, such zero-shot performance of CLIP-based models does not realize in tasks that require a finer-grained correspondence between vision and language, such as Visual Question Answering (VQA). We investigate why this is the case, and report an interesting phenomenon of CLIP, which we call the Concept Association Bias (CAB), as a potential cause of the difficulty of applying CLIP to VQA and similar tasks. CAB is especially apparent when two concepts are present in the given image while a text prompt only contains a single concept. In such a case, we find that CLIP tends to treat input as a bag of concepts and attempts to fill in the other missing concept crossmodally, leading to an unexpected zero-shot prediction. For example, when asked for the color of a lemon in an image, CLIP predicts ``purple'' if the image contains a lemon and an eggplant. We demonstrate the Concept Association Bias of CLIP by showing that CLIP's zero-shot classification performance greatly suffers when there is a strong concept association between an object (e.g. lemon) and an attribute (e.g. its color). On the other hand, when the association between object and attribute is weak, we do not see this phenomenon. Furthermore, we show that CAB is significantly mitigated when we enable CLIP to learn deeper structure across image and text embeddings by adding an additional Transformer on top of CLIP and fine-tuning it on VQA. We find that across such fine-tuned variants of CLIP, the strength of CAB in a model predicts how well it performs on VQA.
### Why Does Surprisal From Larger Transformer-Based Language Models Provide  a Poorer Fit to Human Reading Times?
 - **Authors:** Byung-Doh Oh, William Schuler
 - **Subjects:** Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2212.12131
 - **Pdf link:** https://arxiv.org/pdf/2212.12131
 - **Abstract**
 This work presents a detailed linguistic analysis into why larger Transformer-based pre-trained language models with more parameters and lower perplexity nonetheless yield surprisal estimates that are less predictive of human reading times. First, regression analyses show a strictly monotonic, positive log-linear relationship between perplexity and fit to reading times for the more recently released five GPT-Neo variants and eight OPT variants on two separate datasets, replicating earlier results limited to just GPT-2 (Oh et al., 2022). Subsequently, analysis of residual errors reveals a systematic deviation of the larger variants, such as underpredicting reading times of named entities and making compensatory overpredictions for reading times of function words such as modals and conjunctions. These results suggest that the propensity of larger Transformer-based models to 'memorize' sequences during training makes their surprisal estimates diverge from humanlike expectations, which warrants caution in using pre-trained language models to study human language processing.
### HiTSKT: A Hierarchical Transformer Model for Session-Aware Knowledge  Tracing
 - **Authors:** Fucai Ke, Weiqing Wang, Weicong Tan, Lan Du, Yuan Jin, Yujin Huang, Hongzhi Yin
 - **Subjects:** Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2212.12139
 - **Pdf link:** https://arxiv.org/pdf/2212.12139
 - **Abstract**
 Knowledge tracing (KT) aims to leverage students' learning histories to estimate their mastery levels on a set of pre-defined skills, based on which the corresponding future performance can be accurately predicted. In practice, a student's learning history comprises answers to sets of massed questions, each known as a session, rather than merely being a sequence of independent answers. Theoretically, within and across these sessions, students' learning dynamics can be very different. Therefore, how to effectively model the dynamics of students' knowledge states within and across the sessions is crucial for handling the KT problem. Most existing KT models treat student's learning records as a single continuing sequence, without capturing the sessional shift of students' knowledge state. To address the above issue, we propose a novel hierarchical transformer model, named HiTSKT, comprises an interaction(-level) encoder to capture the knowledge a student acquires within a session, and a session(-level) encoder to summarise acquired knowledge across the past sessions. To predict an interaction in the current session, a knowledge retriever integrates the summarised past-session knowledge with the previous interactions' information into proper knowledge representations. These representations are then used to compute the student's current knowledge state. Additionally, to model the student's long-term forgetting behaviour across the sessions, a power-law-decay attention mechanism is designed and deployed in the session encoder, allowing it to emphasize more on the recent sessions. Extensive experiments on three public datasets demonstrate that HiTSKT achieves new state-of-the-art performance on all the datasets compared with six state-of-the-art KT models.
### PanoViT: Vision Transformer for Room Layout Estimation from a Single  Panoramic Image
 - **Authors:** Weichao Shen, Yuan Dong, Zonghao Chen, Zhengyi Zhao, Yang Gao, Zhu Liu
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2212.12156
 - **Pdf link:** https://arxiv.org/pdf/2212.12156
 - **Abstract**
 In this paper, we propose PanoViT, a panorama vision transformer to estimate the room layout from a single panoramic image. Compared to CNN models, our PanoViT is more proficient in learning global information from the panoramic image for the estimation of complex room layouts. Considering the difference between a perspective image and an equirectangular image, we design a novel recurrent position embedding and a patch sampling method for the processing of panoramic images. In addition to extracting global information, PanoViT also includes a frequency-domain edge enhancement module and a 3D loss to extract local geometric features in a panoramic image. Experimental results on several datasets demonstrate that our method outperforms state-of-the-art solutions in room layout prediction accuracy.
### Text classification in shipping industry using unsupervised models and  Transformer based supervised models
 - **Authors:** Ying Xie, Dongping Song
 - **Subjects:** Computation and Language (cs.CL); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2212.12407
 - **Pdf link:** https://arxiv.org/pdf/2212.12407
 - **Abstract**
 Obtaining labelled data in a particular context could be expensive and time consuming. Although different algorithms, including unsupervised learning, semi-supervised learning, self-learning have been adopted, the performance of text classification varies with context. Given the lack of labelled dataset, we proposed a novel and simple unsupervised text classification model to classify cargo content in international shipping industry using the Standard International Trade Classification (SITC) codes. Our method stems from representing words using pretrained Glove Word Embeddings and finding the most likely label using Cosine Similarity. To compare unsupervised text classification model with supervised classification, we also applied several Transformer models to classify cargo content. Due to lack of training data, the SITC numerical codes and the corresponding textual descriptions were used as training data. A small number of manually labelled cargo content data was used to evaluate the classification performances of the unsupervised classification and the Transformer based supervised classification. The comparison reveals that unsupervised classification significantly outperforms Transformer based supervised classification even after increasing the size of the training dataset by 30%. Lacking training data is a key bottleneck that prohibits deep learning models (such as Transformers) from successful practical applications. Unsupervised classification can provide an alternative efficient and effective method to classify text when there is scarce training data.
### MicroBERT: Effective Training of Low-resource Monolingual BERTs through  Parameter Reduction and Multitask Learning
 - **Authors:** Luke Gessler, Amir Zeldes
 - **Subjects:** Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2212.12510
 - **Pdf link:** https://arxiv.org/pdf/2212.12510
 - **Abstract**
 Transformer language models (TLMs) are critical for most NLP tasks, but they are difficult to create for low-resource languages because of how much pretraining data they require. In this work, we investigate two techniques for training monolingual TLMs in a low-resource setting: greatly reducing TLM size, and complementing the masked language modeling objective with two linguistically rich supervised tasks (part-of-speech tagging and dependency parsing). Results from 7 diverse languages indicate that our model, MicroBERT, is able to produce marked improvements in downstream task evaluations relative to a typical monolingual TLM pretraining approach. Specifically, we find that monolingual MicroBERT models achieve gains of up to 18% for parser LAS and 11% for NER F1 compared to a multilingual baseline, mBERT, while having less than 1% of its parameter count. We conclude reducing TLM parameter count and using labeled data for pretraining low-resource TLMs can yield large quality benefits and in some cases produce models that outperform multilingual approaches.
## Keyword: autonomous driving
### A Novel Method for Lane-change Maneuver in Urban Driving Using  Predictive Markov Decision Process
 - **Authors:** Avinash Prabu, Niranjan Ravi, Lingxi Li
 - **Subjects:** Systems and Control (eess.SY)
 - **Arxiv link:** https://arxiv.org/abs/2212.12008
 - **Pdf link:** https://arxiv.org/pdf/2212.12008
 - **Abstract**
 Lane-change maneuver has always been a challenging task for both manual and autonomous driving, especially in an urban setting. In particular, the uncertainty in predicting the behavior of other vehicles on the road leads to indecisive actions while changing lanes, which, might result in traffic congestion and cause safety concerns. This paper analyzes the factors related to uncertainty such as speed range change and lane change so as to design a predictive Markov decision process for lane-change maneuver in the urban setting. A hidden Markov model is developed for modeling uncertainties of surrounding vehicles. The reward model uses the crash probabilities and the feasibility/distance to the goal as primary parameters. Numerical simulation and analysis of two traffic scenarios are completed to demonstrate the effectiveness of the proposed approach.
### Technical Report: Automating Vehicle SOA Threat Analysis using a  Model-Based Methodology
 - **Authors:** Yuri Gil Dantas, Simon Barner, Pei Ke, Vivek Nigam, Ulrich Schoepp
 - **Subjects:** Logic in Computer Science (cs.LO)
 - **Arxiv link:** https://arxiv.org/abs/2212.12347
 - **Pdf link:** https://arxiv.org/pdf/2212.12347
 - **Abstract**
 While the adoption of Service-Oriented Architectures (SOA) eases the implementation of features such as autonomous driving and over-the-air updates, it also increases the vehicle's exposure to attacks that may place road-users in harm. To address this problem, standards (ISO 21434/UNECE) expect manufacturers to produce security arguments and evidence by carrying out appropriate threat analysis. As key threat analysis steps, e.g., damage/threat scenario and attack path enumeration, are often carried out manually and not rigorously, security arguments lack precise guarantees, e.g., traceability w.r.t. safety goals, especially under system updates. This article proposes automated methods for threat analysis using a model-based engineering methodology that provides precise guarantees with respect to safety goals. This is accomplished by proposing an intruder model for automotive SOA which together with the system architecture and the loss scenarios identified by safety analysis are used as input for computing assets, impact rating, damage/threat scenarios, and attack paths. To validate the proposed methodology, we developed a faithful model of the autonomous driving functions of the Apollo framework, a widely used open-source autonomous driving stack. The proposed machinery automatically enumerates several attack paths on Apollo, including attack paths not reported in the literature.
