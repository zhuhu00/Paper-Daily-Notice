# New submissions for Thu, 22 Dec 22
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
### PaletteNeRF: Palette-based Appearance Editing of Neural Radiance Fields
 - **Authors:** Zhengfei Kuang, Fujun Luan, Sai Bi, Zhixin Shu, Gordon Wetzstein, Kalyan Sunkavalli
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV); Graphics (cs.GR)
 - **Arxiv link:** https://arxiv.org/abs/2212.10699
 - **Pdf link:** https://arxiv.org/pdf/2212.10699
 - **Abstract**
 Recent advances in neural radiance fields have enabled the high-fidelity 3D reconstruction of complex scenes for novel view synthesis. However, it remains underexplored how the appearance of such representations can be efficiently edited while maintaining photorealism. In this work, we present PaletteNeRF, a novel method for photorealistic appearance editing of neural radiance fields (NeRF) based on 3D color decomposition. Our method decomposes the appearance of each 3D point into a linear combination of palette-based bases (i.e., 3D segmentations defined by a group of NeRF-type functions) that are shared across the scene. While our palette-based bases are view-independent, we also predict a view-dependent function to capture the color residual (e.g., specular shading). During training, we jointly optimize the basis functions and the color palettes, and we also introduce novel regularizers to encourage the spatial coherence of the decomposition. Our method allows users to efficiently edit the appearance of the 3D scene by modifying the color palettes. We also extend our framework with compressed semantic features for semantic-aware appearance editing. We demonstrate that our technique is superior to baseline methods both quantitatively and qualitatively for appearance editing of complex real-world scenes.
### Incremental Learning for Neural Radiance Field with Uncertainty-Filtered  Knowledge Distillation
 - **Authors:** Mengqi Guo, Chen Li, Gim Hee Lee
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2212.10950
 - **Pdf link:** https://arxiv.org/pdf/2212.10950
 - **Abstract**
 Recent neural radiance field (NeRF) representation has achieved great success in the tasks of novel view synthesis and 3D reconstruction. However, they suffer from the catastrophic forgetting problem when continuously learning from streaming data without revisiting the previous training data. This limitation prohibits the application of existing NeRF models to scenarios where images come in sequentially. In view of this, we explore the task of incremental learning for neural radiance field representation in this work. We first propose a student-teacher pipeline to mitigate the catastrophic forgetting problem. Specifically, we iterate the process of using the student as the teacher at the end of each incremental step and let the teacher guide the training of the student in the next step. In this way, the student network is able to learn new information from the streaming data and retain old knowledge from the teacher network simultaneously. Given that not all information from the teacher network is helpful since it is only trained with the old data, we further introduce a random inquirer and an uncertainty-based filter to filter useful information. We conduct experiments on the NeRF-synthetic360 and NeRF-real360 datasets, where our approach significantly outperforms the baselines by 7.3% and 25.2% in terms of PSNR. Furthermore, we also show that our approach can be applied to the large-scale camera facing-outwards dataset ScanNet, where we surpass the baseline by 60.0% in PSNR.
## Keyword: mapping
### Requirements Engineering for Artificial Intelligence Systems: A  Systematic Mapping Study
 - **Authors:** Khlood Ahmad, Mohamed Abdelrazek, Chetan Arora, Muneera Bano, John Grundy
 - **Subjects:** Software Engineering (cs.SE)
 - **Arxiv link:** https://arxiv.org/abs/2212.10693
 - **Pdf link:** https://arxiv.org/pdf/2212.10693
 - **Abstract**
 [Context] In traditional software systems, Requirements Engineering (RE) activities are well-established and researched. However, building Artificial Intelligence (AI) based software with limited or no insight into the system's inner workings poses significant new challenges to RE. Existing literature has focused on using AI to manage RE activities, with limited research on RE for AI (RE4AI). [Objective] This paper investigates current approaches for specifying requirements for AI systems, identifies available frameworks, methodologies, tools, and techniques used to model requirements, and finds existing challenges and limitations. [Method] We performed a systematic mapping study to find papers on current RE4AI approaches. We identified 43 primary studies and analysed the existing methodologies, models, tools, and techniques used to specify and model requirements in real-world scenarios. [Results] We found several challenges and limitations of existing RE4AI practices. The findings highlighted that current RE applications were not adequately adaptable for building AI systems and emphasised the need to provide new techniques and tools to support RE4AI. [Conclusion] Our results showed that most of the empirical studies on RE4AI focused on autonomous, self-driving vehicles and managing data requirements, and areas such as ethics, trust, and explainability need further research.
### ZEROTOP: Zero-Shot Task-Oriented Semantic Parsing using Large Language  Models
 - **Authors:** Dheeraj Mekala, Jason Wolfe, Subhro Roy
 - **Subjects:** Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2212.10815
 - **Pdf link:** https://arxiv.org/pdf/2212.10815
 - **Abstract**
 We explore the use of large language models (LLMs) for zero-shot semantic parsing. Semantic parsing involves mapping natural language utterances to task-specific meaning representations. Language models are generally trained on the publicly available text and code and cannot be expected to directly generalize to domain-specific parsing tasks in a zero-shot setting. In this work, we propose ZEROTOP, a zero-shot task-oriented parsing method that decomposes a semantic parsing problem into a set of abstractive and extractive question-answering (QA) problems, enabling us to leverage the ability of LLMs to zero-shot answer reading comprehension questions. For each utterance, we prompt the LLM with questions corresponding to its top-level intent and a set of slots and use the LLM generations to construct the target meaning representation. We observe that current LLMs fail to detect unanswerable questions; and as a result, cannot handle questions corresponding to missing slots. To address this problem, we fine-tune a language model on public QA datasets using synthetic negative samples. Experimental results show that our QA-based decomposition paired with the fine-tuned LLM can correctly parse ~16% of utterances in the MTOP dataset without requiring any annotated data.
### RECAP: Retrieval Augmented Music Captioner
 - **Authors:** Zihao He, Weituo Hao, Xuchen Song
 - **Subjects:** Sound (cs.SD); Computation and Language (cs.CL); Information Retrieval (cs.IR); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2212.10901
 - **Pdf link:** https://arxiv.org/pdf/2212.10901
 - **Abstract**
 With the prevalence of stream media platforms serving music search and recommendation, interpreting music by understanding audio and lyrics interactively has become an important and challenging task. However, many previous works focus on refining individual components of encoder-decoder architecture mapping music to caption tokens, ignoring the potential usage of audio and lyrics correspondence. In this paper, we propose to explicitly learn the multi-modal alignment with retrieval augmentation by contrastive learning. By learning audio-lyrics correspondence, the model is guided to learn better cross-modal attention weights, thus generating high-quality caption words. We provide both theoretical and empirical results that demonstrate the advantage of the proposed method.
### Automatic Semantic Modeling for Structural Data Source with the Prior  Knowledge from Knowledge Base
 - **Authors:** Jiakang Xu, Wolfgang Mayer, HongYu Zhang, Keqing He, Zaiwen Feng
 - **Subjects:** Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2212.10915
 - **Pdf link:** https://arxiv.org/pdf/2212.10915
 - **Abstract**
 A critical step in sharing semantic content online is to map the structural data source to a public domain ontology. This problem is denoted as the Relational-To-Ontology Mapping Problem (Rel2Onto). A huge effort and expertise are required for manually modeling the semantics of data. Therefore, an automatic approach for learning the semantics of a data source is desirable. Most of the existing work studies the semantic annotation of source attributes. However, although critical, the research for automatically inferring the relationships between attributes is very limited. In this paper, we propose a novel method for semantically annotating structured data sources using machine learning, graph matching and modified frequent subgraph mining to amend the candidate model. In our work, Knowledge graph is used as prior knowledge. Our evaluation shows that our approach outperforms two state-of-the-art solutions in tricky cases where only a few semantic models are known.
## Keyword: localization
### TruFor: Leveraging all-round clues for trustworthy image forgery  detection and localization
 - **Authors:** Fabrizio Guillaro, Davide Cozzolino, Avneesh Sud, Nicholas Dufour, Luisa Verdoliva
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2212.10957
 - **Pdf link:** https://arxiv.org/pdf/2212.10957
 - **Abstract**
 In this paper we present TruFor, a forensic framework that can be applied to a large variety of image manipulation methods, from classic cheapfakes to more recent manipulations based on deep learning. We rely on the extraction of both high-level and low-level traces through a transformer-based fusion architecture that combines the RGB image and a learned noise-sensitive fingerprint. The latter learns to embed the artifacts related to the camera internal and external processing by training only on real data in a self-supervised manner. Forgeries are detected as deviations from the expected regular pattern that characterizes each pristine image. Looking for anomalies makes the approach able to robustly detect a variety of local manipulations, ensuring generalization. In addition to a pixel-level localization map and a whole-image integrity score, our approach outputs a reliability map that highlights areas where localization predictions may be error-prone. This is particularly important in forensic applications in order to reduce false alarms and allow for a large scale analysis. Extensive experiments on several datasets show that our method is able to reliably detect and localize both cheapfakes and deepfakes manipulations outperforming state-of-the-art works. Code will be publicly available at https://grip-unina.github.io/TruFor/
### 3D Highlighter: Localizing Regions on 3D Shapes via Text Descriptions
 - **Authors:** Dale Decatur, Itai Lang, Rana Hanocka
 - **Subjects:** Graphics (cs.GR); Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2212.11263
 - **Pdf link:** https://arxiv.org/pdf/2212.11263
 - **Abstract**
 We present 3D Highlighter, a technique for localizing semantic regions on a mesh using text as input. A key feature of our system is the ability to interpret "out-of-domain" localizations. Our system demonstrates the ability to reason about where to place non-obviously related concepts on an input 3D shape, such as adding clothing to a bare 3D animal model. Our method contextualizes the text description using a neural field and colors the corresponding region of the shape using a probability-weighted blend. Our neural optimization is guided by a pre-trained CLIP encoder, which bypasses the need for any 3D datasets or 3D annotations. Thus, 3D Highlighter is highly flexible, general, and capable of producing localizations on a myriad of input shapes. Our code is publicly available at https://github.com/threedle/3DHighlighter.
## Keyword: transformer
### KronA: Parameter Efficient Tuning with Kronecker Adapter
 - **Authors:** Ali Edalati, Marzieh Tahaei, Ivan Kobyzev, Vahid Partovi Nia, James J. Clark, Mehdi Rezagholizadeh
 - **Subjects:** Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2212.10650
 - **Pdf link:** https://arxiv.org/pdf/2212.10650
 - **Abstract**
 Fine-tuning a Pre-trained Language Model (PLM) on a specific downstream task has been a well-known paradigm in Natural Language Processing. However, with the ever-growing size of PLMs, training the entire model on several downstream tasks becomes very expensive and resource-hungry. Recently, different Parameter Efficient Tuning (PET) techniques are proposed to improve the efficiency of fine-tuning PLMs. One popular category of PET methods is the low-rank adaptation methods which insert learnable truncated SVD modules into the original model either sequentially or in parallel. However, low-rank decomposition suffers from limited representation power. In this work, we address this problem using the Kronecker product instead of the low-rank representation. We introduce KronA, a Kronecker product-based adapter module for efficient fine-tuning of Transformer-based PLMs. We apply the proposed methods for fine-tuning T5 on the GLUE benchmark to show that incorporating the Kronecker-based modules can outperform state-of-the-art PET methods.
### METEOR Guided Divergence for Video Captioning
 - **Authors:** Daniel Lukas Rothenpieler, Shahin Amiriparian
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV); Computation and Language (cs.CL); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2212.10690
 - **Pdf link:** https://arxiv.org/pdf/2212.10690
 - **Abstract**
 Automatic video captioning aims for a holistic visual scene understanding. It requires a mechanism for capturing temporal context in video frames and the ability to comprehend the actions and associations of objects in a given timeframe. Such a system should additionally learn to abstract video sequences into sensible representations as well as to generate natural written language. While the majority of captioning models focus solely on the visual inputs, little attention has been paid to the audiovisual modality. To tackle this issue, we propose a novel two-fold approach. First, we implement a reward-guided KL Divergence to train a video captioning model which is resilient towards token permutations. Second, we utilise a Bi-Modal Hierarchical Reinforcement Learning (BMHRL) Transformer architecture to capture long-term temporal dependencies of the input data as a foundation for our hierarchical captioning module. Using our BMHRL, we show the suitability of the HRL agent in the generation of content-complete and grammatically sound sentences by achieving $4.91$, $2.23$, and $10.80$ in BLEU3, BLEU4, and METEOR scores, respectively on the ActivityNet Captions dataset. Finally, we make our BMHRL framework and trained models publicly available for users and developers at https://github.com/d-rothen/bmhrl.
### Analyzing Semantic Faithfulness of Language Models via Input  Intervention on Conversational Question Answering
 - **Authors:** Akshay Chaturvedi, Swarnadeep Bhar, Soumadeep Saha, Utpal Garain, Nicholas Asher
 - **Subjects:** Computation and Language (cs.CL); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2212.10696
 - **Pdf link:** https://arxiv.org/pdf/2212.10696
 - **Abstract**
 Transformer-based language models have been shown to be highly effective for several NLP tasks. In this paper, we consider three transformer models, BERT, RoBERTa, and XLNet, in both small and large version, and investigate how faithful their representations are with respect to the semantic content of texts. We formalize a notion of semantic faithfulness, in which the semantic content of a text should causally figure in a model's inferences in question answering. We then test this notion by observing a model's behavior on answering questions about a story after performing two novel semantic interventions -- deletion intervention and negation intervention. While transformer models achieve high performance on standard question answering tasks, we show that they fail to be semantically faithful once we perform these interventions for a significant number of cases (~50% for deletion intervention, and ~20% drop in accuracy for negation intervention). We then propose an intervention-based training regime that can mitigate the undesirable effects for deletion intervention by a significant margin (from ~50% to ~6%). We analyze the inner-workings of the models to better understand the effectiveness of intervention-based training for deletion intervention. But we show that this training does not attenuate other aspects of semantic unfaithfulness such as the models' inability to deal with negation intervention or to capture the predicate-argument structure of texts. We also test InstructGPT, via prompting, for its ability to handle the two interventions and to capture predicate-argument structure. While InstructGPT models do achieve very high performance on predicate-argument structure task, they fail to respond adequately to our deletion and negation interventions.
### Zero-shot Triplet Extraction by Template Infilling
 - **Authors:** Bosung Kim, Hayate Iso, Nikita Bhutani, Estevam Hruschka, Ndapa Nakashole
 - **Subjects:** Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2212.10708
 - **Pdf link:** https://arxiv.org/pdf/2212.10708
 - **Abstract**
 Triplet extraction aims to extract entities and their corresponding relations in unstructured text. Most existing methods train an extraction model on high-quality training data, and hence are incapable of extracting relations that were not observed during training. Generalizing the model to unseen relations typically requires fine-tuning on synthetic training data which is often noisy and unreliable. In this paper, we argue that reducing triplet extraction to a template filling task over a pre-trained language model can equip the model with zero-shot learning capabilities and enable it to leverage the implicit knowledge in the language model. Embodying these ideas, we propose a novel framework, ZETT (ZEro-shot Triplet extraction by Template infilling), that is based on end-to-end generative transformers. Our experiments show that without any data augmentation or pipeline systems, ZETT can outperform previous state-of-the-art models with 25% less parameters. We further show that ZETT is more robust in detecting entities and can be incorporated with automatically generated templates for relations.
### Beyond Contrastive Learning: A Variational Generative Model for  Multilingual Retrieval
 - **Authors:** John Wieting, Jonathan H. Clark, William W. Cohen, Graham Neubig, Taylor Berg-Kirkpatrick
 - **Subjects:** Computation and Language (cs.CL); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2212.10726
 - **Pdf link:** https://arxiv.org/pdf/2212.10726
 - **Abstract**
 Contrastive learning has been successfully used for retrieval of semantically aligned sentences, but it often requires large batch sizes or careful engineering to work well. In this paper, we instead propose a generative model for learning multilingual text embeddings which can be used to retrieve or score sentence pairs. Our model operates on parallel data in $N$ languages and, through an approximation we introduce, efficiently encourages source separation in this multilingual setting, separating semantic information that is shared between translations from stylistic or language-specific variation. We show careful large-scale comparisons between contrastive and generation-based approaches for learning multilingual text embeddings, a comparison that has not been done to the best of our knowledge despite the popularity of these approaches. We evaluate this method on a suite of tasks including semantic similarity, bitext mining, and cross-lingual question retrieval -- the last of which we introduce in this paper. Overall, our Variational Multilingual Source-Separation Transformer (VMSST) model outperforms both a strong contrastive and generative baseline on these tasks.
### Spoken Language Understanding for Conversational AI: Recent Advances and  Future Direction
 - **Authors:** Soyeon Caren Han, Siqu Long, Henry Weld, Josiah Poon
 - **Subjects:** Computation and Language (cs.CL); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2212.10728
 - **Pdf link:** https://arxiv.org/pdf/2212.10728
 - **Abstract**
 When a human communicates with a machine using natural language on the web and online, how can it understand the human's intention and semantic context of their talk? This is an important AI task as it enables the machine to construct a sensible answer or perform a useful action for the human. Meaning is represented at the sentence level, identification of which is known as intent detection, and at the word level, a labelling task called slot filling. This dual-level joint task requires innovative thinking about natural language and deep learning network design, and as a result, many approaches and models have been proposed and applied. This tutorial will discuss how the joint task is set up and introduce Spoken Language Understanding/Natural Language Understanding (SLU/NLU) with Deep Learning techniques. We will cover the datasets, experiments and metrics used in the field. We will describe how the machine uses the latest NLP and Deep Learning techniques to address the joint task, including recurrent and attention-based Transformer networks and pre-trained models (e.g. BERT). We will then look in detail at a network that allows the two levels of the task, intent classification and slot filling, to interact to boost performance explicitly. We will do a code demonstration of a Python notebook for this model and attendees will have an opportunity to watch coding demo tasks on this joint NLU to further their understanding.
### SLGTformer: An Attention-Based Approach to Sign Language Recognition
 - **Authors:** Neil Song
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2212.10746
 - **Pdf link:** https://arxiv.org/pdf/2212.10746
 - **Abstract**
 Sign language is the preferred method of communication of deaf or mute people, but similar to any language, it is difficult to learn and represents a significant barrier for those who are hard of hearing or unable to speak. A person's entire frontal appearance dictates and conveys specific meaning. However, this frontal appearance can be quantified as a temporal sequence of human body pose, leading to Sign Language Recognition through the learning of spatiotemporal dynamics of skeleton keypoints. I propose a novel, attention-based approach to Sign Language Recognition exclusively built upon decoupled graph and temporal self-attention: the Sign Language Graph Time Transformer (SLGTformer). SLGTformer first deconstructs spatiotemporal pose sequences separately into spatial graphs and temporal windows. SLGTformer then leverages novel Learnable Graph Relative Positional Encodings (LGRPE) to guide spatial self-attention with the graph neighborhood context of the human skeleton. By modeling the temporal dimension as intra- and inter-window dynamics, I introduce Temporal Twin Self-Attention (TTSA) as the combination of locally-grouped temporal attention (LTA) and global sub-sampled temporal attention (GSTA). I demonstrate the effectiveness of SLGTformer on the World-Level American Sign Language (WLASL) dataset, achieving state-of-the-art performance with an ensemble-free approach on the keypoint modality.
### JASMINE: Arabic GPT Models for Few-Shot Learning
 - **Authors:** El Moatez Billah Nagoudi, Muhammad Abdul-Mageed, AbdelRahim Elmadany, Alcides Alcoba Inciarte, Md Tawkat Islam Khondaker
 - **Subjects:** Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2212.10755
 - **Pdf link:** https://arxiv.org/pdf/2212.10755
 - **Abstract**
 Task agnostic generative pretraining (GPT) has recently proved promising for zero- and few-shot learning, gradually diverting attention from the expensive supervised learning paradigm. Although the community is accumulating knowledge as to capabilities of English-language autoregressive models such as GPT-3 adopting this generative approach, scholarship about these models remains acutely Anglocentric. Consequently, the community currently has serious gaps in its understanding of this class of models, their potential, and their societal impacts in diverse settings, linguistic traditions, and cultures. To alleviate this issue for Arabic, a collection of diverse languages and language varieties with more than $400$ million population, we introduce JASMINE, a suite of powerful Arabic autoregressive Transformer language models ranging in size between 300 million-13 billion parameters. We pretrain our new models with large amounts of diverse data (400GB of text) from different Arabic varieties and domains. We evaluate JASMINE extensively in both intrinsic and extrinsic settings, using a comprehensive benchmark for zero- and few-shot learning across a wide range of NLP tasks. We also carefully develop and release a novel benchmark for both automated and human evaluation of Arabic autoregressive models focused at investigating potential social biases, harms, and toxicity in these models. We aim to responsibly release our models with interested researchers, along with code for experimenting with them
### TruFor: Leveraging all-round clues for trustworthy image forgery  detection and localization
 - **Authors:** Fabrizio Guillaro, Davide Cozzolino, Avneesh Sud, Nicholas Dufour, Luisa Verdoliva
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2212.10957
 - **Pdf link:** https://arxiv.org/pdf/2212.10957
 - **Abstract**
 In this paper we present TruFor, a forensic framework that can be applied to a large variety of image manipulation methods, from classic cheapfakes to more recent manipulations based on deep learning. We rely on the extraction of both high-level and low-level traces through a transformer-based fusion architecture that combines the RGB image and a learned noise-sensitive fingerprint. The latter learns to embed the artifacts related to the camera internal and external processing by training only on real data in a self-supervised manner. Forgeries are detected as deviations from the expected regular pattern that characterizes each pristine image. Looking for anomalies makes the approach able to robustly detect a variety of local manipulations, ensuring generalization. In addition to a pixel-level localization map and a whole-image integrity score, our approach outputs a reliability map that highlights areas where localization predictions may be error-prone. This is particularly important in forensic applications in order to reduce false alarms and allow for a large scale analysis. Extensive experiments on several datasets show that our method is able to reliably detect and localize both cheapfakes and deepfakes manipulations outperforming state-of-the-art works. Code will be publicly available at https://grip-unina.github.io/TruFor/
### What Makes for Good Tokenizers in Vision Transformer?
 - **Authors:** Shengju Qian, Yi Zhu, Wenbo Li, Mu Li, Jiaya Jia
 - **Subjects:** Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2212.11115
 - **Pdf link:** https://arxiv.org/pdf/2212.11115
 - **Abstract**
 The architecture of transformers, which recently witness booming applications in vision tasks, has pivoted against the widespread convolutional paradigm. Relying on the tokenization process that splits inputs into multiple tokens, transformers are capable of extracting their pairwise relationships using self-attention. While being the stemming building block of transformers, what makes for a good tokenizer has not been well understood in computer vision. In this work, we investigate this uncharted problem from an information trade-off perspective. In addition to unifying and understanding existing structural modifications, our derivation leads to better design strategies for vision tokenizers. The proposed Modulation across Tokens (MoTo) incorporates inter-token modeling capability through normalization. Furthermore, a regularization objective TokenProp is embraced in the standard training regime. Through extensive experiments on various transformer architectures, we observe both improved performance and intriguing properties of these two plug-and-play designs with negligible computational overhead. These observations further indicate the importance of the commonly-omitted designs of tokenizers in vision transformer.
### Generating music with sentiment using Transformer-GANs
 - **Authors:** Pedro Neves, Jose Fornari, João Florindo
 - **Subjects:** Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2212.11134
 - **Pdf link:** https://arxiv.org/pdf/2212.11134
 - **Abstract**
 The field of Automatic Music Generation has seen significant progress thanks to the advent of Deep Learning. However, most of these results have been produced by unconditional models, which lack the ability to interact with their users, not allowing them to guide the generative process in meaningful and practical ways. Moreover, synthesizing music that remains coherent across longer timescales while still capturing the local aspects that make it sound ``realistic'' or ``human-like'' is still challenging. This is due to the large computational requirements needed to work with long sequences of data, and also to limitations imposed by the training schemes that are often employed. In this paper, we propose a generative model of symbolic music conditioned by data retrieved from human sentiment. The model is a Transformer-GAN trained with labels that correspond to different configurations of the valence and arousal dimensions that quantitatively represent human affective states. We try to tackle both of the problems above by employing an efficient linear version of Attention and using a Discriminator both as a tool to improve the overall quality of the generated music and its ability to follow the conditioning signals.
### Entropy- and Distance-Based Predictors From GPT-2 Attention Patterns  Predict Reading Times Over and Above GPT-2 Surprisal
 - **Authors:** Byung-Doh Oh, William Schuler
 - **Subjects:** Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2212.11185
 - **Pdf link:** https://arxiv.org/pdf/2212.11185
 - **Abstract**
 Transformer-based large language models are trained to make predictions about the next word by aggregating representations of previous tokens through their self-attention mechanism. In the field of cognitive modeling, such attention patterns have recently been interpreted as embodying the process of cue-based retrieval, in which attention over multiple targets is taken to generate interference and latency during retrieval. Under this framework, this work first defines an entropy-based predictor that quantifies the diffuseness of self-attention, as well as distance-based predictors that capture the incremental change in attention patterns across timesteps. Moreover, following recent studies that question the informativeness of attention weights, we also experiment with alternative methods for incorporating vector norms into attention weights. Regression experiments using predictors calculated from the GPT-2 language model show that these predictors deliver a substantially better fit to held-out self-paced reading and eye-tracking data over a rigorous baseline including GPT-2 surprisal. Additionally, the distance-based predictors generally demonstrated higher predictive power, with effect sizes of up to 6.59 ms per standard deviation on self-paced reading times (compared to 2.82 ms for surprisal) and 1.05 ms per standard deviation on eye-gaze durations (compared to 3.81 ms for surprisal).
## Keyword: autonomous driving
There is no result 