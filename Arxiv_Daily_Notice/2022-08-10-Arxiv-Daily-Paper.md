# New submissions for Wed, 10 Aug 22
## Keyword: SLAM
There is no result 
## Keyword: odometry
### Deep Patch Visual Odometry
 - **Authors:** Zachary Teed, Lahav Lipson, Jia Deng
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2208.04726
 - **Pdf link:** https://arxiv.org/pdf/2208.04726
 - **Abstract**
 We propose Deep Patch Visual Odometry (DPVO), a new deep learning system for monocular Visual Odometry (VO). DPVO is accurate and robust while running at 2x-5x real-time speeds on a single RTX-3090 GPU using only 4GB of memory. We perform evaluation on standard benchmarks and outperform all prior work (classical or learned) in both accuracy and speed. Code is available at https://github.com/princeton-vl/DPVO.
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
### Lotse: A Practical Framework for Guidance in Visual Analytics
 - **Authors:** Fabian Sperrle, Davide Ceneda, Mennatallah El-Assady
 - **Subjects:** Human-Computer Interaction (cs.HC)
 - **Arxiv link:** https://arxiv.org/abs/2208.04434
 - **Pdf link:** https://arxiv.org/pdf/2208.04434
 - **Abstract**
 Co-adaptive guidance aims to enable efficient human-machine collaboration in visual analytics, as proposed by multiple theoretical frameworks. This paper bridges the gap between such conceptual frameworks and practical implementation by introducing an accessible model of guidance and an accompanying guidance library, mapping theory into practice. We contribute a model of system-provided guidance based on design templates and derived strategies. We instantiate the model in a library called Lotse that allows specifying guidance strategies in definition files and generates running code from them. Lotse is the first guidance library using such an approach. It supports the creation of reusable guidance strategies to retrofit existing applications with guidance and fosters the creation of general guidance strategy patterns. We demonstrate its effectiveness through first-use case studies with VA researchers of varying guidance design expertise and find that they are able to effectively and quickly implement guidance with Lotse. Further, we analyze our framework's cognitive dimensions to evaluate its expressiveness and outline a summary of open research questions for aligning guidance practice with its intricate theory.
### Comparative Evaluation of Bipartite, Node-Link, and Matrix-Based Network  Representations
 - **Authors:** Moataz Abdelaal, Nathan D. Schiele, Katrin Angerbauer, Kuno Kurzhals, Michael Sedlmair, Daniel Weiskopf
 - **Subjects:** Human-Computer Interaction (cs.HC)
 - **Arxiv link:** https://arxiv.org/abs/2208.04458
 - **Pdf link:** https://arxiv.org/pdf/2208.04458
 - **Abstract**
 This work investigates and compares the performance of node-link diagrams, adjacency matrices, and bipartite layouts for visualizing networks. In a crowd-sourced user study (n = 150), we measure the task accuracy and completion time of the three representations for different network classes and properties. In contrast to the literature, which covers mostly topology-based tasks (e.g., path finding) in small datasets, we mainly focus on overview tasks for large and directed networks. We consider three overview tasks on networks with 500 nodes: (T1) network class identification, (T2) cluster detection, and (T3) network density estimation, and two detailed tasks: (T4) node in-degree vs. out-degree and (T5) representation mapping, on networks with 50 and 20 nodes, respectively. Our results show that bipartite layouts are beneficial for revealing the overall network structure, while adjacency matrices are most reliable across the different tasks.
### A Theoretical View on Sparsely Activated Networks
 - **Authors:** Cenk Baykal, Nishanth Dikkala, Rina Panigrahy, Cyrus Rashtchian, Xin Wang
 - **Subjects:** Machine Learning (cs.LG); Machine Learning (stat.ML)
 - **Arxiv link:** https://arxiv.org/abs/2208.04461
 - **Pdf link:** https://arxiv.org/pdf/2208.04461
 - **Abstract**
 Deep and wide neural networks successfully fit very complex functions today, but dense models are starting to be prohibitively expensive for inference. To mitigate this, one promising direction is networks that activate a sparse subgraph of the network. The subgraph is chosen by a data-dependent routing function, enforcing a fixed mapping of inputs to subnetworks (e.g., the Mixture of Experts (MoE) paradigm in Switch Transformers). However, prior work is largely empirical, and while existing routing functions work well in practice, they do not lead to theoretical guarantees on approximation ability. We aim to provide a theoretical explanation for the power of sparse networks. As our first contribution, we present a formal model of data-dependent sparse networks that captures salient aspects of popular architectures. We then introduce a routing function based on locality sensitive hashing (LSH) that enables us to reason about how well sparse networks approximate target functions. After representing LSH-based sparse networks with our model, we prove that sparse networks can match the approximation power of dense networks on Lipschitz functions. Applying LSH on the input vectors means that the experts interpolate the target function in different subregions of the input space. To support our theory, we define various datasets based on Lipschitz target functions, and we show that sparse networks give a favorable trade-off between number of active units and approximation quality.
### An Unconstrained Symmetric Nonnegative Latent Factor Analysis for  Large-scale Undirected Weighted Networks
 - **Authors:** Zhe Xie, Weiling Li, Yurong Zhong
 - **Subjects:** Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2208.04811
 - **Pdf link:** https://arxiv.org/pdf/2208.04811
 - **Abstract**
 Large-scale undirected weighted networks are usually found in big data-related research fields. It can naturally be quantified as a symmetric high-dimensional and incomplete (SHDI) matrix for implementing big data analysis tasks. A symmetric non-negative latent-factor-analysis (SNL) model is able to efficiently extract latent factors (LFs) from an SHDI matrix. Yet it relies on a constraint-combination training scheme, which makes it lack flexibility. To address this issue, this paper proposes an unconstrained symmetric nonnegative latent-factor-analysis (USNL) model. Its main idea is two-fold: 1) The output LFs are separated from the decision parameters via integrating a nonnegative mapping function into an SNL model; and 2) Stochastic gradient descent (SGD) is adopted for implementing unconstrained model training along with ensuring the output LFs nonnegativity. Empirical studies on four SHDI matrices generated from real big data applications demonstrate that an USNL model achieves higher prediction accuracy of missing data than an SNL model, as well as highly competitive computational efficiency.
### Design of High-Throughput Mixed-Precision CNN Accelerators on FPGA
 - **Authors:** Cecilia Latotzke, Tim Ciesielski, Tobias Gemmeke
 - **Subjects:** Hardware Architecture (cs.AR); Artificial Intelligence (cs.AI); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2208.04854
 - **Pdf link:** https://arxiv.org/pdf/2208.04854
 - **Abstract**
 Convolutional Neural Networks (CNNs) reach high accuracies in various application domains, but require large amounts of computation and incur costly data movements. One method to decrease these costs while trading accuracy is weight and/or activation word-length reduction. Thereby, layer-wise mixed-precision quantization allows for more efficient results while inflating the design space. In this work, we present an in-depth quantitative methodology to efficiently explore the design space considering the limited hardware resources of a given FPGA. Our holistic exploration approach vertically traverses the various design entry levels from the architectural down to the logic level, and laterally covers optimization from processing elements to dataflow for an efficient mixed-precision CNN accelerator. Our resulting hardware accelerators implement truly mixed-precision operations that enable efficient execution of layer-wise and channel-wise quantized CNNs. Mapping feed-forward and identity-shortcut-connection mixed-precision CNNs result in competitive accuracy-throughout trade-offs: 245 frames/s with 87.48% Top-5 accuracy for ResNet-18 and 92.9% Top-5 accuracy with 1.13 TOps/s for ResNet-152, respectively. Thereby, the required memory footprint for parameters is reduced by 4.9x and 9.4x compared to the respective floating-point baseline.
## Keyword: localization
### Object Detection with Deep Reinforcement Learning
 - **Authors:** Manoosh Samiei, Ruofeng Li
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2208.04511
 - **Pdf link:** https://arxiv.org/pdf/2208.04511
 - **Abstract**
 Object localization has been a crucial task in computer vision field. Methods of localizing objects in an image have been proposed based on the features of the attended pixels. Recently researchers have proposed methods to formulate object localization as a dynamic decision process, which can be solved by a reinforcement learning approach. In this project, we implement a novel active object localization algorithm based on deep reinforcement learning. We compare two different action settings for this MDP: a hierarchical method and a dynamic method. We further perform some ablation studies on the performance of the models by investigating different hyperparameters and various architecture changes.
### TSRFormer: Table Structure Recognition with Transformers
 - **Authors:** Weihong Lin, Zheng Sun, Chixiang Ma, Mingze Li, Jiawei Wang, Lei Sun, Qiang Huo
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2208.04921
 - **Pdf link:** https://arxiv.org/pdf/2208.04921
 - **Abstract**
 We present a new table structure recognition (TSR) approach, called TSRFormer, to robustly recognizing the structures of complex tables with geometrical distortions from various table images. Unlike previous methods, we formulate table separation line prediction as a line regression problem instead of an image segmentation problem and propose a new two-stage DETR based separator prediction approach, dubbed \textbf{Sep}arator \textbf{RE}gression \textbf{TR}ansformer (SepRETR), to predict separation lines from table images directly. To make the two-stage DETR framework work efficiently and effectively for the separation line prediction task, we propose two improvements: 1) A prior-enhanced matching strategy to solve the slow convergence issue of DETR; 2) A new cross attention module to sample features from a high-resolution convolutional feature map directly so that high localization accuracy is achieved with low computational cost. After separation line prediction, a simple relation network based cell merging module is used to recover spanning cells. With these new techniques, our TSRFormer achieves state-of-the-art performance on several benchmark datasets, including SciTSR, PubTabNet and WTW. Furthermore, we have validated the robustness of our approach to tables with complex structures, borderless cells, large blank spaces, empty or spanning cells as well as distorted or even curved shapes on a more challenging real-world in-house dataset.
## Keyword: transformer
### Investigating Efficiently Extending Transformers for Long Input  Summarization
 - **Authors:** Jason Phang, Yao Zhao, Peter J. Liu
 - **Subjects:** Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2208.04347
 - **Pdf link:** https://arxiv.org/pdf/2208.04347
 - **Abstract**
 While large pretrained Transformer models have proven highly capable at tackling natural language tasks, handling long sequence inputs continues to be a significant challenge. One such task is long input summarization, where inputs are longer than the maximum input context of most pretrained models. Through an extensive set of experiments, we investigate what model architectural changes and pretraining paradigms can most efficiently adapt a pretrained Transformer for long input summarization. We find that a staggered, block-local Transformer with global encoder tokens strikes a good balance of performance and efficiency, and that an additional pretraining phase on long sequences meaningfully improves downstream summarization performance. Based on our findings, we introduce PEGASUS-X, an extension of the PEGASUS model with additional long input pretraining to handle inputs of up to 16K tokens. PEGASUS-X achieves strong performance on long input summarization tasks comparable with much larger models while adding few additional parameters and not requiring model parallelism to train.
### Learning to Learn to Predict Performance Regressions in Production at  Meta
 - **Authors:** Moritz Beller, Hongyu Li, Vivek Nair, Vijayaraghavan Murali, Imad Ahmad, Jürgen Cito, Drew Carlson, Ari Aye, Wes Dyer
 - **Subjects:** Software Engineering (cs.SE); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2208.04351
 - **Pdf link:** https://arxiv.org/pdf/2208.04351
 - **Abstract**
 Catching and attributing code change-induced performance regressions in production is hard; predicting them beforehand, even harder. A primer on automatically learning to predict performance regressions in software, this article gives an account of the experiences we gained when researching and deploying an ML-based regression prediction pipeline at Meta. In this paper, we report on a comparative study with four ML models of increasing complexity, from (1) code-opaque, over (2) Bag of Words, (3) off-the-shelve Transformer-based, to (4) a bespoke Transformer-based model, coined SuperPerforator. Our investigation shows the inherent difficulty of the performance prediction problem, which is characterized by a large imbalance of benign onto regressing changes. Our results also call into question the general applicability of Transformer-based architectures for performance prediction: an off-the-shelve CodeBERT-based approach had surprisingly poor performance; our highly customized SuperPerforator architecture initially achieved prediction performance that was just on par with simpler Bag of Words models, and only outperformed them for down-stream use cases. This ability of SuperPerforator to transfer to an application with few learning examples afforded an opportunity to deploy it in practice at Meta: it can act as a pre-filter to sort out changes that are unlikely to introduce a regression, truncating the space of changes to search a regression in by up to 43%, a 45x improvement over a random baseline. To gain further insight into SuperPerforator, we explored it via a series of experiments computing counterfactual explanations. These highlight which parts of a code change the model deems important, thereby validating the learned black-box model.
### Occlusion-Aware Instance Segmentation via BiLayer Network Architectures
 - **Authors:** Lei Ke, Yu-Wing Tai, Chi-Keung Tang
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2208.04438
 - **Pdf link:** https://arxiv.org/pdf/2208.04438
 - **Abstract**
 Segmenting highly-overlapping image objects is challenging, because there is typically no distinction between real object contours and occlusion boundaries on images. Unlike previous instance segmentation methods, we model image formation as a composition of two overlapping layers, and propose Bilayer Convolutional Network (BCNet), where the top layer detects occluding objects (occluders) and the bottom layer infers partially occluded instances (occludees). The explicit modeling of occlusion relationship with bilayer structure naturally decouples the boundaries of both the occluding and occluded instances, and considers the interaction between them during mask regression. We investigate the efficacy of bilayer structure using two popular convolutional network designs, namely, Fully Convolutional Network (FCN) and Graph Convolutional Network (GCN). Further, we formulate bilayer decoupling using the vision transformer (ViT), by representing instances in the image as separate learnable occluder and occludee queries. Large and consistent improvements using one/two-stage and query-based object detectors with various backbones and network layer choices validate the generalization ability of bilayer decoupling, as shown by extensive experiments on image instance segmentation benchmarks (COCO, KINS, COCOA) and video instance segmentation benchmarks (YTVIS, OVIS, BDD100K MOTS), especially for heavy occlusion cases. Code and data are available at https://github.com/lkeab/BCNet.
### A Theoretical View on Sparsely Activated Networks
 - **Authors:** Cenk Baykal, Nishanth Dikkala, Rina Panigrahy, Cyrus Rashtchian, Xin Wang
 - **Subjects:** Machine Learning (cs.LG); Machine Learning (stat.ML)
 - **Arxiv link:** https://arxiv.org/abs/2208.04461
 - **Pdf link:** https://arxiv.org/pdf/2208.04461
 - **Abstract**
 Deep and wide neural networks successfully fit very complex functions today, but dense models are starting to be prohibitively expensive for inference. To mitigate this, one promising direction is networks that activate a sparse subgraph of the network. The subgraph is chosen by a data-dependent routing function, enforcing a fixed mapping of inputs to subnetworks (e.g., the Mixture of Experts (MoE) paradigm in Switch Transformers). However, prior work is largely empirical, and while existing routing functions work well in practice, they do not lead to theoretical guarantees on approximation ability. We aim to provide a theoretical explanation for the power of sparse networks. As our first contribution, we present a formal model of data-dependent sparse networks that captures salient aspects of popular architectures. We then introduce a routing function based on locality sensitive hashing (LSH) that enables us to reason about how well sparse networks approximate target functions. After representing LSH-based sparse networks with our model, we prove that sparse networks can match the approximation power of dense networks on Lipschitz functions. Applying LSH on the input vectors means that the experts interpolate the target function in different subregions of the input space. To support our theory, we define various datasets based on Lipschitz target functions, and we show that sparse networks give a favorable trade-off between number of active units and approximation quality.
### In the Eye of Transformer: Global-Local Correlation for Egocentric Gaze  Estimation
 - **Authors:** Bolin Lai, Miao Liu, Fiona Ryan, James Rehg
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2208.04464
 - **Pdf link:** https://arxiv.org/pdf/2208.04464
 - **Abstract**
 In this paper, we present the first transformer-based model to address the challenging problem of egocentric gaze estimation. We observe that the connection between the global scene context and local visual information is vital for localizing the gaze fixation from egocentric video frames. To this end, we design the transformer encoder to embed the global context as one additional visual token and further propose a novel Global-Local Correlation (GLC) module to explicitly model the correlation of the global token and each local token. We validate our model on two egocentric video datasets - EGTEA Gaze+ and Ego4D. Our detailed ablation studies demonstrate the benefits of our method. In addition, our approach exceeds previous state-of-the-arts by a large margin. We also provide additional visualizations to support our claim that global-local correlation serves a key representation for predicting gaze fixation from egocentric videos. More details can be found in our website (https://bolinlai.github.io/GLC-EgoGazeEst).
### Emotion Detection From Tweets Using a BERT and SVM Ensemble Model
 - **Authors:** Ionuţ-Alexandru Albu, Stelian Spînu
 - **Subjects:** Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2208.04547
 - **Pdf link:** https://arxiv.org/pdf/2208.04547
 - **Abstract**
 Automatic identification of emotions expressed in Twitter data has a wide range of applications. We create a well-balanced dataset by adding a neutral class to a benchmark dataset consisting of four emotions: fear, sadness, joy, and anger. On this extended dataset, we investigate the use of Support Vector Machine (SVM) and Bidirectional Encoder Representations from Transformers (BERT) for emotion recognition. We propose a novel ensemble model by combining the two BERT and SVM models. Experiments show that the proposed model achieves a state-of-the-art accuracy of 0.91 on emotion recognition in tweets.
### High Recall Data-to-text Generation with Progressive Edit
 - **Authors:** Choonghan Kim, Gary Geunbae Lee
 - **Subjects:** Computation and Language (cs.CL); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2208.04558
 - **Pdf link:** https://arxiv.org/pdf/2208.04558
 - **Abstract**
 Data-to-text (D2T) generation is the task of generating texts from structured inputs. We observed that when the same target sentence was repeated twice, Transformer (T5) based model generates an output made up of asymmetric sentences from structured inputs. In other words, these sentences were different in length and quality. We call this phenomenon "Asymmetric Generation" and we exploit this in D2T generation. Once asymmetric sentences are generated, we add the first part of the output with a no-repeated-target. As this goes through progressive edit (ProEdit), the recall increases. Hence, this method better covers structured inputs than before editing. ProEdit is a simple but effective way to improve performance in D2T generation and it achieves the new stateof-the-art result on the ToTTo dataset
### More Interpretable Graph Similarity Computation via Maximum Common  Subgraph Inference
 - **Authors:** Zixun Lan, Binjie Hong, Ye Ma, Fei Ma
 - **Subjects:** Machine Learning (cs.LG); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2208.04580
 - **Pdf link:** https://arxiv.org/pdf/2208.04580
 - **Abstract**
 Graph similarity measurement, which computes the distance/similarity between two graphs, arises in various graph-related tasks. Recent learning-based methods lack interpretability, as they directly transform interaction information between two graphs into one hidden vector and then map it to similarity. To cope with this problem, this study proposes a more interpretable end-to-end paradigm for graph similarity learning, named Similarity Computation via Maximum Common Subgraph Inference (INFMCS). Our critical insight into INFMCS is the strong correlation between similarity score and Maximum Common Subgraph (MCS). We implicitly infer MCS to obtain the normalized MCS size, with the supervision information being only the similarity score during training. To capture more global information, we also stack some vanilla transformer encoder layers with graph convolution layers and propose a novel permutation-invariant node Positional Encoding. The entire model is quite simple yet effective. Comprehensive experiments demonstrate that INFMCS consistently outperforms state-of-the-art baselines for graph-graph classification and regression tasks. Ablation experiments verify the effectiveness of the proposed computation paradigm and other components. Also, visualization and statistics of results reveal the interpretability of INFMCS.
### How Well Do Vision Transformers (VTs) Transfer To The Non-Natural Image  Domain? An Empirical Study Involving Art Classification
 - **Authors:** Vincent Tonkes, Matthia Sabatelli
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2208.04693
 - **Pdf link:** https://arxiv.org/pdf/2208.04693
 - **Abstract**
 Vision Transformers (VTs) are becoming a valuable alternative to Convolutional Neural Networks (CNNs) when it comes to problems involving high-dimensional and spatially organized inputs such as images. However, their Transfer Learning (TL) properties are not yet well studied, and it is not fully known whether these neural architectures can transfer across different domains as well as CNNs. In this paper we study whether VTs that are pre-trained on the popular ImageNet dataset learn representations that are transferable to the non-natural image domain. To do so we consider three well-studied art classification problems and use them as a surrogate for studying the TL potential of four popular VTs. Their performance is extensively compared against that of four common CNNs across several TL experiments. Our results show that VTs exhibit strong generalization properties and that these networks are more powerful feature extractors than CNNs.
### Early Stage Sparse Retrieval with Entity Linking
 - **Authors:** Dahlia Shehata, Negar Arabzadeh, Charles L. A. Clarke
 - **Subjects:** Information Retrieval (cs.IR)
 - **Arxiv link:** https://arxiv.org/abs/2208.04887
 - **Pdf link:** https://arxiv.org/pdf/2208.04887
 - **Abstract**
 Despite the advantages of their low-resource settings, traditional sparse retrievers depend on exact matching approaches between high-dimensional bag-of-words (BoW) representations of both the queries and the collection. As a result, retrieval performance is restricted by semantic discrepancies and vocabulary gaps. On the other hand, transformer-based dense retrievers introduce significant improvements on information retrieval tasks by exploiting low-dimensional contextualized representations of the corpus. While dense retrievers are known for their relative effectiveness, they suffer from lower efficiency and lack of generalization issues, when compared to sparse retrievers. For a light-weight retrieval task, high computational resources and time consumption are major barriers encouraging the renunciation of dense models despite potential gains. In this work, we propose boosting the performance of sparse retrievers by expanding both the queries and the documents with linked entities in two formats for the entity names: 1) explicit and 2) hashed. We employ a zero-shot end-to-end dense entity linking system for entity recognition and disambiguation to augment the corpus. By leveraging the advanced entity linking methods, we believe that the effectiveness gap between sparse and dense retrievers can be narrowed. We conduct our experiments on the MS MARCO passage dataset. Since we are concerned with the early stage retrieval in cascaded ranking architectures of large information retrieval systems, we evaluate our results using recall@1000. Our approach is also capable of retrieving documents for query subsets judged to be particularly difficult in prior work. We further demonstrate that the non-expanded and the expanded runs with both explicit and hashed entities retrieve complementary results. Consequently, we adopt a run fusion approach to maximize the benefits of entity linking.
### Sports Video Analysis on Large-Scale Data
 - **Authors:** Dekun Wu, He Zhao, Xingce Bao, Richard P. Wildes
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2208.04897
 - **Pdf link:** https://arxiv.org/pdf/2208.04897
 - **Abstract**
 This paper investigates the modeling of automated machine description on sports video, which has seen much progress recently. Nevertheless, state-of-the-art approaches fall quite short of capturing how human experts analyze sports scenes. There are several major reasons: (1) The used dataset is collected from non-official providers, which naturally creates a gap between models trained on those datasets and real-world applications; (2) previously proposed methods require extensive annotation efforts (i.e., player and ball segmentation at pixel level) on localizing useful visual features to yield acceptable results; (3) very few public datasets are available. In this paper, we propose a novel large-scale NBA dataset for Sports Video Analysis (NSVA) with a focus on captioning, to address the above challenges. We also design a unified approach to process raw videos into a stack of meaningful features with minimum labelling efforts, showing that cross modeling on such features using a transformer architecture leads to strong performance. In addition, we demonstrate the broad application of NSVA by addressing two additional tasks, namely fine-grained sports action recognition and salient player identification. Code and dataset are available at https://github.com/jackwu502/NSVA.
### Simplified State Space Layers for Sequence Modeling
 - **Authors:** Jimmy T.H. Smith, Andrew Warrington, Scott W. Linderman
 - **Subjects:** Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2208.04933
 - **Pdf link:** https://arxiv.org/pdf/2208.04933
 - **Abstract**
 Efficiently modeling long-range dependencies is an important goal in sequence modeling. Recently, models using structured state space sequence (S4) layers achieved state-of-the-art performance on many long-range tasks. The S4 layer combines linear state space models (SSMs) with deep learning techniques and leverages the HiPPO framework for online function approximation to achieve high performance. However, this framework led to architectural constraints and computational difficulties that make the S4 approach complicated to understand and implement. We revisit the idea that closely following the HiPPO framework is necessary for high performance. Specifically, we replace the bank of many independent single-input, single-output (SISO) SSMs the S4 layer uses with one multi-input, multi-output (MIMO) SSM with a reduced latent dimension. The reduced latent dimension of the MIMO system allows for the use of efficient parallel scans which simplify the computations required to apply the S5 layer as a sequence-to-sequence transformation. In addition, we initialize the state matrix of the S5 SSM with an approximation to the HiPPO-LegS matrix used by S4's SSMs and show that this serves as an effective initialization for the MIMO setting. S5 matches S4's performance on long-range tasks, including achieving an average of 82.46% on the suite of Long Range Arena benchmarks compared to S4's 80.48% and the best transformer variant's 61.41%.
## Keyword: autonomous driving
### VectorFlow: Combining Images and Vectors for Traffic Occupancy and Flow  Prediction
 - **Authors:** Xin Huang, Xiaoyu Tian, Junru Gu, Qiao Sun, Hang Zhao
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV); Artificial Intelligence (cs.AI); Robotics (cs.RO)
 - **Arxiv link:** https://arxiv.org/abs/2208.04530
 - **Pdf link:** https://arxiv.org/pdf/2208.04530
 - **Abstract**
 Predicting future behaviors of road agents is a key task in autonomous driving. While existing models have demonstrated great success in predicting marginal agent future behaviors, it remains a challenge to efficiently predict consistent joint behaviors of multiple agents. Recently, the occupancy flow fields representation was proposed to represent joint future states of road agents through a combination of occupancy grid and flow, which supports efficient and consistent joint predictions. In this work, we propose a novel occupancy flow fields predictor to produce accurate occupancy and flow predictions, by combining the power of an image encoder that learns features from a rasterized traffic image and a vector encoder that captures information of continuous agent trajectories and map states. The two encoded features are fused by multiple attention modules before generating final predictions. Our simple but effective model ranks 3rd place on the Waymo Open Dataset Occupancy and Flow Prediction Challenge, and achieves the best performance in the occluded occupancy and flow prediction task.
