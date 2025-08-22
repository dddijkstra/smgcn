# A traditional Chinese medicine prescription recommendation model based on contrastive pre-training and hierarchical structure network

Hailong Hu a,*, Yaqian Li a, Zeyu Zheng a, Wenjun Hu a*, Riyang Lin b, Yanlei Kang a

a School of Information Engineering, Huzhou University, Huzhou, 313000, China  b Department of Traditional Chinese Medicine, Hangzhou Hospital of Traditional Chinese Medicine, Hangzhou, 310007, China

# ARTICLE INFO

Keywords:  TCM prescription recommendation  Contrastive pre- training  Hierarchical structure network  Network pharmacology

# ABSTRACT

Traditional Chinese Medicine (TCM) prescriptions are personalized treatment plans crafted by Chinese practitioners based on TCM principles and clinical insights, tailored through the examination of patient symptoms, physical constitution, and other relevant data. However, the efficacy of existing TCM prescription recommendation models is often hampered by data scarcity, disparities in node popularity, and challenges in interpreting the recommended prescriptions, leading to outcomes that may need more convincing accuracy and interpretability. This paper introduces a TCM prescription recommendation model utilizing contrastive pre- training and a hierarchical structure network. By fitting node features via multi- view contrastive pretraining, this approach alleviates the issue of data sparsity. It further integrates linked features within homogeneous networks at a granular level. Moreover, a hierarchical structural network focuses on less popular nodes, enriching the representations of symptoms and herbal features. During the analysis of the results, an interpretability analysis of the recommended TCM prescriptions is performed using the network pharmacology method. The performance of our model surpasses the compared methods in the comparison. Compared to the best model, our model shows improvements on both datasets. In Dataset1, Precision@20, Recall@20, and F1- score@20 increase by  $2.14\%$ $5.51\%$  and  $3.07\%$  respectively. In Dataset2, Precision@20, Recall@20, and F1- score@20 rise by  $1.50\%$ $1.64\%$  and  $1.51\%$  respectively. The herbal prescription recommendation model in this study enhances the accuracy of herbal recommendations. It not only provides new insights for TCM clinical practice but also promotes the modernization and innovative development of TCM diagnosis and treatment.

# 1. Introduction

Traditional Chinese Medicine (TCM), with a history spanning millennia, represents a profound medical legacy of China, encapsulating ancient medical expertise and insights (Chu et al., 2020; Gao et al., 2020; Peng & Lu, 2020). According to TCM theory, diseases stem from imbalances in the body's Yin (阳) and Yang (阳) or from inadequate circulation of Qi (气) and Blood (血). TCM aims to harmonize these elements within the body, thereby restoring balance (Huang et al., 2021; Lv et al., 2023). In practice, Chinese physicians employ the Four Diagnostic Methods (Lv et al., 2023) to thoroughly understand a patient's condition. Using the information gleaned from these methods and the holistic principles of TCM, physicians determine the disease's location and characteristics, diagnose the patient's syndrome, and develop tailored treatment strategies and prescriptions (Zhang et al., 2021).

With the continuous advancements in artificial intelligence (AI) and machine learning, AI's application in modernizing TCM spans various areas. These include syndrome diagnosis and classification, TCM medication pattern analysis, knowledge extraction from TCM texts, and other industry- specific technologies. For instance, Li et al. (2024) introduced an enhanced graph convolution network (GCN) framework for symptom recommendation. Chen et al. (2024) combined BERT with CNN and integrated TCM syndrome differentiation features to build a holistic syndrome differentiation model for various classification tasks. Teng et al. (2024) proposed SEHGCN, which incorporates state elements into syndrome classification. Additionally, Xu et al. (2023) used data mining methods to explore TCM prescription patterns for polycystic ovary syndrome (PCOS). Large Language Models (LLMs) have also gained significant attention. Yang et al. (2024) and Hua et al. (2024) worked on enhancing TCM knowledge encoding and adaptability, while Tan et al. (2024) proposed an LLM specifically for

TCM consultation. These interconnected AI applications collectively drive innovation in TCM.

Despite considerable technological progress in recent years, the fundamental practices of diagnosing and prescribing in TCM still rely heavily on the subjective assessments and individual experiences of Chinese doctors. This reliance can lead to variability in the diagnoses and treatment recommendations provided to the same patient by different doctors, highlighting the need for greater standardization. Consequently, many researchers have applied AI to TCM prescription recommendations. For example, PresRecST (Dong et al., 2024) introduced a residual neural network and knowledge graph, implementing a three- stage clinical prediction to achieve a systematic sequential approach to TCM decision- making. PreGenerator (Zhao et al., 2023), proposed using a hybrid neural network to generate herbal prescriptions, guiding the rational combination of herbs by mining and analyzing the correlations and compatibilities between different herbs. Jin, Zhang et al. (2024) presented a GAT- based, knowledge- driven, and TCM- informed herbal recommendation method, integrating various TCM knowledge to enhance the feature representation of herbal entities. Zhao et al. (2024) proposed a multi- label learning framework based on a dual- stream visual transformer (ML- ViT), which constructed herbal prescriptions by robustly modeling patients' facial and tongue images. AI algorithms can explore the interactions between herbs and symptoms (Liu et al., 2024), as well as the combined effects of various herbs, thus providing a scientific foundation for recommending TCM prescriptions. By utilizing these sophisticated algorithms, there is anticipation for a significant improvement in the precision and standardization of TCM therapies.

Currently, efforts have been made to conceptualize TCM diagnosis and treatment as a TCM recommendation system, selecting suitable herb combinations based on comprehensive symptom scores for herbs. Nonetheless, existing TCM prescription recommendation models encounter three primary challenges: (1) Sparsity of prescription data. Unlike modern medicine, which benefits from extensive clinical trial data, TCM prescriptions are grounded in traditional knowledge and individual experience, often resulting in scarce and incomplete data. Moreover, the complexity of TCM prescriptions, which may include various herbs in different types and dosages, exacerbates data sparsity, complicating the training and evaluation of models to uncover potential patterns and correlations between symptoms and herbs. (2) Node popularity differences. The presence of both high- frequency and low- frequency nodes in the dataset can skew the recommendation system, favoring high- frequency nodes at the expense of less common herbs- an issue inadequately addressed by current TCM recommendation approaches. (3) Interpretability of recommended prescriptions. While AI methods can identify potential correlations between herbs and symptoms, the prescriptions they recommend frequently suffer from significant interpretability issues, potentially undermining trust among Chinese doctors.

To address these issues, this paper introduces a TCM prescription recommendation model based on contrastive pre- training and a hierarchical structure network (TCMRGCL). This model comprises four main modules: (1) Contrastive pre- training module (CpreT); (2) Homogenous association network learning module (HANL); (3) Hierarchical structure network learning module (HSNL); and (4) Prescription recommendation module (PR). The key contributions of this study are summarized as follows:

TCMRGCL employs TSVD matrix decomposition and edge perturbation to create two distinct enhanced views. High- order interaction features are subsequently extracted through a graph convolutional network (GCN) for each view. These views undergo contrastive pretraining to derive node representations for subsequent recommendation tasks.

The model merges two distinct coupling features of two node types (heterogeneous similarity and homogeneous similarity) while forming homogeneous networks. By applying the Hadamard product to the homogeneous networks, it refines the features at a finer granularity.

- Utilizing the soft K-means algorithm, the model constructs a hierarchical structured network. It aggregates nodes at various levels within this network to learn node features from the original symptom-herb interaction network.

- Combining with network pharmacology methods allows for the provision of interpretative explanations for the recommended herb combinations.

The structure of this paper is organized as follows: Section 2 delves into related work on TCM recommendation systems. Section 3 describes the detailed methodology of TCMRGCL. Section 4 provides the experimental outcomes and analyses and Section 5 discusses our work. Finally, Section 6 concludes the paper with a summary and outlook on future research directions.

# 2. Related work

# 2.1. TCM recommendation

Recent research has utilized topic models to analyze the tripartite relationship among symptoms, herbs, and prescriptions (Wang et al., 2016; Yao et al., 2018). However, traditional topic models often struggle with short texts. To address this, some researchers have turned to neural networks to more effectively capture the interactions between symptoms and herbs (Li et al., 2020; Liu et al., 2022). For instance, one study applied multiple attention mechanisms to explore the connections between symptom data, medical history, and herbal information, subsequently refining their model with a limited dataset of medical records to generate TCM prescriptions. Another approach (Niu et al., 2023) combined TCM expertise, AI, and network science algorithms to derive a comprehensive score for prescription formulation.

# 2.2. Graph neural networks and recommendation system

Graph neural networks (GNNs), capable of processing irregular graph- structured data, are applied in diverse areas such as social networks (Guo et al., 2023; Kumar et al., 2022; Min et al., 2021), recommendation systems (Gao, Zheng et al., 2023; Yu, Yin et al., 2024), and protein interaction networks in bioinformatics (Gao, Jiang et al., 2023; Zeng et al., 2024). This versatility has led researchers to model TCM prescriptions as graphs (Jin, Ji et al., 2023; Jin et al., 2022, 2020; Yang & Ding, 2023; Yang et al., 2022; Zhao et al., 2022), where nodes represent symptoms and herbs, and edges denote therapeutic connections. GNNs facilitate the learning of latent associations between herbs and symptoms. Some studies (Jin et al., 2022; Yang et al., 2022) have integrated herbal knowledge graphs to improve the model's ability to represent herb features accurately. Another study (Jin, Ji et al., 2023) employed meta- paths to guide information propagation, selecting highly attentive path instances for detailed explanations. Zhao et al. (2022) introduced state elements and syndrome types to model complex relationships between symptoms and TCM prescriptions more effectively.

However, these approaches generally treat all nodes uniformly, reducing the significance of less common nodes. They focus on aggregating information from neighboring nodes and often neglect the graph's global structure, which may limit their capacity to discern high- order relationships and intricate patterns among non- adjacent nodes. Therefore, compared to the simple neighbor aggregation in SMGCN (Jin et al., 2020) and KDHR (Yang et al., 2022), we incorporate heterogeneous and homogeneous similarities of entity nodes to enhance node features with finer granularity selectively.

# 2.3. Graph contrastive learning and recommendation system

Graph contrastive learning (GCL) has recently sparked significant interest in the recommendation system domain Yu, Yin et al. (2024) due to its ability to generate self- supervised signals from raw data, addressing the challenge of data sparsity in recommendation systems (Ahmadian et al., 2023; Kuo & Li, 2023). GCL has been successfully applied to various recommendation tasks, including product recommendations (Ji et al., 2023; Ma et al., 2023; Yu, Xia et al., 2024), conversation recommendations (Zhang, Ma et al., 2023), and sequence recommendations (Zhang, Yin et al., 2023). Specifically, in the context of TCM recommendation systems, one study (Yin et al., 2022) introduced a heterogeneous graph contrastive learning model that integrates node- level and semantic- level analysis. This approach aims to reduce the dominance of high- frequency nodes and improve model accuracy by adjusting the graph's structure.

However, the absence of explicit syndrome differentiation constrains the performance and general applicability of current TCM prescription recommendation models, indicating a need for further investigation to refine these models. Therefore, we obtain prior knowledge through contrastive pre- training and combine it with a hierarchical structure network to cluster multiple symptoms into syndromes, supplementing the syndrome differentiation and treatment process in TCM diagnosis, thereby enabling more accurate TCM recommendations.

# 3.Methods

3. MethodsIn this section, we first outline the problem definition of TCM prescription recommendation in Section 3.1. Next, in Section 3.2, we present the overarching framework of TCMRGCL. Section 3.3 is dedicated to explaining the CpreT module. Section 3.4 discusses the HANL module. The construction of the HSNL module is addressed in Section 3.5, and the recommendation process for TCM prescriptions is elaborated in Section 3.6.

# 3.1.Problem definition

3.1. Problem definitionUpon gathering a patient's symptoms through the Four Diagnosis Methods, Chinese physicians synthesize these symptoms into syndromes, subsequently prescribing a set of herbs. Therefore, the prescription dataset is expressed as  $\mathrm{P} = \langle s_{- set},h_{- set}\rangle$ ,  $|\mathrm{P}|$  indicating the number of prescriptions. Each prescription contains a symptom set  $s_{- set} = \{s_1,s_2,\ldots ,s_i\}$  and a herb set  $h_{- set} = \{h_1,h_2,\ldots ,h_j\}$ , where  $|s_{- set}| > 0$ ,  $|h_{- set}| > 0$ . In the given TCM prescriptions data, there are a total of  $M$  symptoms in  $N$  herbs, all symptoms form a total symptom set  $V_{S} = \{s_{1},s_{2},\ldots ,s_{M}\}$ , and all herbs form a total herb set  $V_{H} = \{h_{1},h_{2},\ldots ,h_{N}\}$ . The objective of TCM prescription data is to identify and suggest the most suitable herb combination for a patient's specific symptomatology. To achieve this, we propose TCMRGCL, a model designed to establish an effective prediction function that calculates the likelihood of all herbs treating all symptoms. Using the symptom set provided by the patient, the model is capable of accurately selecting the top  $K$  herbs with the highest probability, which constitute the final prescription recommendation.

# 3.2.Overview of TCMRGCL

The TCMRGCL framework is depicted in Fig. 1 and the workflow of the entire model is shown in Algorithm 1. TCMRGCL comprises four principal modules: (1) In the CPreT module, we generate two distinct enhanced views via TSVD matrix decomposition and edge perturbation, followed by the acquisition of general node representations through contrastive pre- training. These representations are subsequently utilized in both the HANL and HSNL modules. (2) The HANL module investigates dependence relationships within homogeneous graphs by creating and analyzing symptom- symptom and herb- herb association graphs. (3) The HSNL module categorizes symptom and herb nodes into three tiers and conducts targeted neighbor aggregation operations on nodes at these various levels to enrich node representation. (4) Finally, TCMRGCL amalgamates the symptom and herb representations derived from HANL and HSNL to forecast the symptoms' preference for herbs, thereby recommending the most fitting herbs.

# Algorithm 1 The overall framework for TCMRGCL.

Algorithm 1 The overall framework for TCMRGCL.Require: Prescription dataset P, SH- graph, SS- graph, HH- graphEnsure: Prediction function  $y_{pre} = g(s_{- set},V_{H}|\theta)$ 1: Randomly initialize model parameters2: get  $Z_{s}$  and  $Z_{h}$  from the CPreT module3: while iter  $<$  epoch do4: Get the initial embedding of SS- graph and HH- graph through the embedding layer5: for prescription  $p$  in P do6: get  $G_{s}^{ss}$  and  $G_{h}^{hh}$  from the HANL module7: get  $G_{s}$  and  $G_{h}$  from the HSNL module8: Connect  $G_{s}^{ss}$  and  $G_{s}$  to get  $Q_{s}$  via Eq. (23), and connect  $G_{h}^{hh}$  and  $G_{h}$  to get  $Q_{h}$  via Eq. (24)9: Perform one- hot encoding on the symptom set in  $p = < s_{- set},h_{- set}>$ , and then interact with  $Q_{s}$  according to Eq. (25) to obtain the comprehensive embedding of symptom set  $Z_{s_{- set}}$ 10: Get  $g(s_{- set},V_{H}|\theta)$  via Eq. (26)11: Update parameters through gradient descent12: end for13: end while14: return  $y_{pre} = g(s_{- set},V_{H}|\theta)$

# 3.3. Contrastive pre-training

3.3. Contrastive pre- trainingThe CPreT module characterizes two different enhanced views from the original SH- graph. TSVD generates one enhanced view, and the other is generated by edge dropout. These two views are extracted and fused using GCN. The views are then contrastively learned to maximize the consistency of the representation of the same node, thereby generating entity- level representations  $Z_{s}$  and  $Z_{h}$ , which are used as the initial representations for the HSNL and HANL modules. The algorithm flow of the CPreT module is shown in Algorithm 2.

# 3.3.1. Construction of SH-graph

Each prescription is characterized by a set of symptoms and herbs. Accordingly, the symptoms and herbs within a prescription can be interconnected to form a symptom- herb graph (SH- graph), with the edges signifying therapeutic connections. For the prescription  $p_1 =$ $\langle s_{- set},h_{- set}\rangle$  , where  $s_{- set} = \{s_1,s_2,s_3\}$  and  $A_{s,set} = \{h_1,h_2,h_3\}$  , the edge set  $E_{p_1}^{SH} = \{(s_1,h_1),(s_1,h_2),(s_1,h_3),\ldots ,(s_3,h_3)\}$  of the SH- graph can be obtained. The SH- graph can be defined as shown in Eq. (1). If there is an edge between two nodes, then  $A_{SH}(s_i,h_j)$  is set to 1; otherwise, it is 0.

$$
A_{SH}(s_i,h_j) = \left\{ \begin{array}{ll}1, & if(s_i,h_j)\in E_P^{SH}\\ 0, & otherwise \end{array} \right. \tag{1}
$$

# 3.3.2. Construction of augmented views

Given that the original view contains much redundant information and noise, which can negatively impact the model's learning by reducing efficiency and accuracy, TCMRGCL introduces two enhanced views for contrastive pre- training to achieve a universal node representation. The first augmented view,  $G_{SH}^{1}$  , is constructed based on TSVD matrix factorization, and the second augmented view,  $G_{SH}^{2}$  , is constructed through edge perturbation.

(1) Construction of augmented view  $G_{SH}^{1}$

According to the principle of singular value decomposition (SVD), a given matrix  $\mathcal{A}$  can be decomposed into the product of three matrices:

$A = USV^{T}$  ,where  $U\in \mathbb{R}^{m\times n}$  and  $V\in \mathbb{R}^{n\times n}$  are standard orthogonal matrices, and  $S\in \mathbb{R}^{m\times n}$  is a non- negative diagonal matrix with singular values arranged in descending order on its diagonal. The use of truncated singular value decomposition (TSVD) (Yan et al., 2021) instead of standard SVD addresses the practical limitations of handling large datasets, particularly the excessive memory and computational resource requirements associated with processing redundant noise. By focusing on the matrix's principal components, TSVD facilitates the construction of an augmented view that efficiently captures the essential features necessary for the model's learning process, as exemplified in Eq. (2).

$$
\tilde{A} = U_rS_rV_r^T \tag{2}
$$

In this context,  $r$  represents the top  $r$  singular values in the SVD, the matrix  $U_{r}\in \mathbb{R}^{M\times r}$  consists of the top  $r$  columns from  $U$  in the SVD, the matrix  $V_{r}\in \mathbb{R}^{r\times N}$  consists of the top  $r$  columns from  $V$  and  $S_{r}\in \mathbb{R}^{r\times r}$  consists of the top  $r$  diagonal elements of  $S$  .By decomposing the matrix  $A_{SH}$  using this method, we obtain its approximate reconstruction matrix  $\tilde{A}_{SH}$  , and the corresponding augmented view is  $G_{SH}^{1}$

(2) Construction of augmented view  $G_{SH}^{2}$

Construct the augmented view  $G_{SH}^{2}$  by performing dropout operations on edges. Dropout serves as a data augmentation technique that produces an augmented perturbation view by randomly discarding some edges, thereby increasing the model's resilience to various data variations and improving its ability to develop more robust feature representations.

# 3.3.3. Neighbor information aggregation of augmented views

For the symptom- herb augmented views  $G_{SH}^{1}$  and  $G_{SH}^{2}$  we use the GCN to learn the embeddings of symptoms and herbs, respectively. Specifically, taking the symptoms as an example, we first initialize all symptoms and herbs, and then aggregate all neighbor information into the current symptoms' embedding through neighbor aggregation. From this, we obtain the symptoms' embedding after aggregating neighbor information. Symptom- based neighbor aggregation and herb- based neighbor aggregation are shown in Eqs. (3) and (4), respectively.

$$
\begin{array}{rl} & {e_s^{l + 1} = \sum_{j\in N_s}\frac{1}{\sqrt{|N_s||N_h|}} e_j^l}\\ & {e_h^{l + 1} = \sum_{i\in N_h}\frac{1}{\sqrt{|N_s||N_h|}} e_i^l} \end{array} \tag{3}
$$

Here,  $e_s^{l + 1}$  and  $e_h^{l + 1}$  represent the features of symptom  $s$  and herb  $h$  respectively, after aggregating the information from the lth layer neighbors.  $N_{s}$  represents the set of herbs interacting with symptom  $s$  in the SH- graph, and  $N_{h}$  represents the set of symptoms interacting with herb  $h$  in the SH- graph.

After completing the aggregation of neighbor information, the target node information is updated, as shown in Eqs. (5) and (6).

$$
\begin{array}{l}{D_s = \frac{1}{L + 1}\sum_{l = 0}^{L}e_s^l}\\ {D_h = \frac{1}{L + 1}\sum_{l = 0}^{L}e_h^l} \end{array} \tag{6}
$$

Here,  $D_{s}$  and  $D_{h}$  represent the features of symptoms and herbs, respectively, after multi- layer information fusion, and  $L$  is the number of aggregation layers.

# 3.3.4. Construction of contrastive loss function

After generating two augmented views that encapsulate multi- layer neighbor information, we apply contrastive learning between these views. The aim is to obtain the contrastive loss  $L_{s}$  by minimizing the distance between positive sample pairs while maximizing the distance between negative pairs. For example, for symptoms, the formula is shown in Eq. (7).

$$
L_{s} = \sum_{i = 0}^{M}\sum_{l = 0}^{L} - \log \frac{\exp(f(D_{i,l}^{Net1},D_{i,l}^{Net2}) / \tau)}{\sum_{i^{\prime} = 0}^{M}\exp(f(D_{i,l}^{Net1},D_{i^{\prime}l}^{Net2}) / \tau)} \tag{7}
$$

Here,  $D_{i,l}^{Net1}$  and  $D_{i,l}^{Net2}$  represent the features of symptoms  $i$  in the lth layer of  $G_{SH}^{1}$  and  $G_{SH}^{2}$  respectively. The function  $f(\cdot)$  is the similarity function. In the experiment, we use the Hadamard product to calculate the similarity between two symptoms, with  $\tau$  as the temperature coefficient. A similar contrastive loss function for herbs can be obtained. Consequently, the total pre- training contrast loss function is presented in Eq. (8).

$$
L_{cl} = L_{s} + L_{h} \tag{2}
$$

As the training progresses,  $L_{cl}$  will gradually decrease. When the loss reaches convergence, the pre- training is completed, and the final symptom embedding  $Z_{s}$  and herb embedding  $Z_{h}$  for downstream recommendation tasks are obtained. These embeddings are  $e_s^0$  and  $e_h^0$  after several training updates.

Algorithm 2 The workflow for CPreT module.

Require: SH- graph

Ensure: The embedding of symptoms  $Z_{s}$  and the embedding of herbs  $Z_{h}$

1: Randomly initialize model parameters 2: Construct augmented view  $G_{SH}^{1}$  through Eq. (2), and construct augmented view  $G_{SH}^{2}$  through edge dropout 3: Get the initial embedding of SH- graph through the embedding layer 4: Obtain the embedding of symptoms and herbs from SH- graph, obtain the neighbor information of symptoms and herbs through Eqs. (3) and (4) respectively 5: Fuse the neighbor information into the current symptoms and herbs through Eqs. (5) and (6) to obtain  $D_{s}$  and  $D_{h}$  6: Train the neural network by using the loss obtained from Eqs. (7) and (8), obtain features of downstream tasks  $Z_{s}$  and  $Z_{h}$  7: return  $Z_{s}$  and  $Z_{h}$

# 3.4. Homogeneous association network learning

In the HANL module, we process symptoms and herbs by using homogeneous networks to capture the associations between similar nodes, obtain the potential structural information of symptoms and herbs, and then generate high- dimensional features  $G_{s}^{ss}$  and  $G_{h}^{hh}$  through an MLP layer. The algorithm flow of the HANL module is shown in Algorithm 3.

# 3.4.1. Construction of SS-graph and HH-graph

The prescription  $p$  represents a set of symptoms that may appear simultaneously in a patient, and there may be interaction or co- occurrence relationships between these symptoms. For the prescription  $p_1$  the edge set of the SS- graph can be obtained as  $E_{p_1}^{SS} = \{(s_1,s_2),(s_1,s_3),(s_2,s_3)\}$ . The SS- graph can be defined as shown in Eq. (9). If there is an edge between two nodes,  $A_{ss}(s_i,s_k)$  is set to 1; otherwise, it is set to 0.

$$
A_{SS}(s_i,s_k) = \left\{ \begin{array}{ll}1,if(s_i,s_k)\in E_P^{SS}\\ 0,otherwise \end{array} \right. \tag{9}
$$

Similarly, the set of herbs in the prescription  $p$  suggests that the combination of herbs can enhance the therapeutic effect, indicating potential interaction or co- occurrence relationships between the herbs. Therefore, for the prescription  $p_1$ , the edge set of the HH- graph can be obtained as  $E_{p_1}^{HH} = \{(h_1,h_2),(h_1,h_3),(h_2,h_3)\}$ . The HH- graph can be defined as shown in Eq. (10). If there is an edge between two nodes,

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-16/8e35f7e7-3d91-4213-bd78-444a97ab04b2/7b3fbc65ec13a70f3eff6f507272620e8b987c51409a1f731ea5a001974a5a94.jpg)  
Fig.1. (2) The HANL module explores relationships within homogeneous graphs by analyzing symptom-symptom and herb-erb associations. (3) The HSNL module categorizes nodes into three levels and performs neighbor aggregation to enhance node representation. (4) TCMRGCL combines these representations to predict and recommend the most suitable herbs.

$A_{hh}(h_j, h_q)$  is set to 1; otherwise, it is set to 0.

$$
A_{HH}(h_i,h_k) = \left\{ \begin{array}{ll}1,if(h_i,h_k)\in E_P^{HH}\\ 0,otherwise \end{array} \right. \tag{10}
$$

# 3.4.2. Construction of similarity matrix

In the realm of TCM, prescriptions crafted by Chinese doctors consider the synergistic effects and interactions of different herbs. Therefore, we highlight the importance of understanding similarities among symptoms and herbs, as this understanding is crucial for unraveling the intricate relationships between symptoms and herbs, ultimately refining treatment strategies. For instance, focusing on herbs, we construct similarity matrices based on both the SH- graph and the HH- graph, and illustrate this through a similarity heatmap in Fig. 2.

(1) Construction of heterogeneous similarity matrix

The heterogeneous similarity matrix, derived from the SH- graph, operates on the premise that if two herbs treat the same symptom, they exhibit a degree of similarity. Such a matrix not only reveals the correlations between various herbs but also aids in the more informed selection and combination of herbs, thereby facilitating more efficient and accurate diagnostic and treatment recommendations. Therefore, the heterogeneous similarity  $HeS(h_j, h_q)$  between herb  $j$  and herb  $q$  can be expressed as shown in Eq. (11).

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-16/8e35f7e7-3d91-4213-bd78-444a97ab04b2/634126aa6e718d396668523a1925a03a2fd4d3d0c785c76829a8ee782e11fda1.jpg)  
Fig. 2. Similarity heatmap of herbs. The horizontal and vertical axes represent the top 10 herbs, with color indicating similarity—darker colors mean higher similarity. (a) shows heterogeneous similarity from the SH-graph, suggesting herbs with similar effects for similar symptoms. (b) shows homogeneous similarity from the HH-graph, indicating herbs with similar pharmacological effects or ingredients.

$$
H e S(h_{j},h_{q}) = \frac{|R x(h_{j})\cap R x(h_{q})|}{M} \tag{11}
$$

Here,  $Rx(h_j)$  represents the set of symptoms related to herb  $j$  in the SH- graph,  $Rx(h_q)$  represents the set of symptoms related to herb  $q$ ,

and  $M$  represents the number of symptoms. The symptom- symptom heterogeneous similarity  $HeS(s_i,s_k)$  is calculated in a manner similar to  $HeS(h_j,h_q)$ . This equation calculates the overlap between the sets of symptoms treated by the two herbs and normalizes it by the total number of symptoms. The similarity score ranges from 0 to 1, where 1 indicates that the two herbs treat the same symptoms, and 0 indicates that they treat entirely different symptoms.

(2) Construction of homogeneous similarity matrix

The homogeneous similarity matrix, built from the HH- graph, defines two herbs as similar if they are related to the same herb. The Jaccard similarity coefficient is used to quantify this similarity. Therefore, the homogeneous similarity  $HoS(h_j,h_q)$  between herb  $j$  and herb  $q$  can be expressed as shown in Eq. (12).

$$
H o S(h_{j},h_{q}) = \frac{|T x(h_{j})\cap T x(h_{j})|}{|T x(h_{j})\cup T x(h_{q})|} \tag{12}
$$

Here,  $Tx(h_j)$  represents the set of herbs related to herb  $j$  in the HH- graph, and  $Tx(h_q)$  represents the set of herbs related to herb  $q$ . The symptom- symptom homogeneous similarity  $HoS(s_i,s_k)$  is calculated in the same way as  $HoS(h_j,h_q)$ . This equation measures the similarity between two herbs by comparing the sets of herbs they are associated with. The Jaccard index is between 0 and 1, where 1 indicates that the two herbs co- occur with the same herbs, and 0 indicates that they do not share any co- occurring herbs.

The herb similarity matrix and the herb- herb adjacency matrix are combined using the Hadamard product. This fusion introduces additional information into the edge data, helping to differentiate between connected and unconnected edges, as detailed in Eqs. (13) and (14).

$$
\begin{array}{rl} & A_{SS}^{\prime} = A_{ss}\odot Sim_{-ss}\\ & A_{HH}^{\prime} = A_{hh}\odot Sim_{-hh}\\ & Sim_{-ss} = \lambda HeS(s_i,s_k) + (1 - \lambda)HoS(s_i,s_k)\\ & Sim_{-hh} = \lambda HeS(s_j,h_q) + (1 - \lambda)HoS(h_j,h_q) \end{array} \tag{13}
$$

Here,  $Sim_{s}s$  and  $Sim_{h}h$  are the comprehensive similarity matrices generated by Eqs. (15) and (16).  $\lambda \in [0,1]$  is a hyperparameter, and  $\odot$  denotes the Hadamard product.

# 3.4.3. Neighbor information aggregation of SS-graph and HH-graph

We use  $Z_{s}$  and  $Z_{h}$  as the initial feature representations of symptoms and herbs, respectively, with  $A_{SS}^{\prime}$  and  $A_{HH}^{\prime}$  as the input to the model. Utilizing the GCN model, we aggregate a layer of neighbor information for both the SS- graph and HH- graph. The aggregation process, outlined in Eqs. (17) and (18), focuses on consolidating neighbor information, while Eqs. (19) and (20) detail the methodology for updating the target node information.

$$
\begin{array}{l}{e_s^1 = \sum_{i\in N_s}\frac{1}{\sqrt{|N_s|}} z_{s_i}}\\ {e_h^1 = \sum_{j\in N_h}\frac{1}{\sqrt{|N_h|}} z_{h_j}} \end{array} \tag{17}
$$

Here,  $e_s^1$  and  $e_h^1$  represent the features after aggregating the 1st- order neighbors of the symptom homogeneous graph and the herb homogeneous graph, respectively.  $z_{s_i}$  and  $z_{h_j}$  represent symptom  $i$  and herb  $j$  in  $Z_{s}$  and  $Z_{h}$ , respectively.

$$
\begin{array}{l}{E_s = \frac{1}{2}\Big(Z_s + e_s^1\Big)}\\ {E_h = \frac{1}{2}\Big(Z_h + e_h^1\Big)} \end{array} \tag{19}
$$

The embedding representations of aggregated neighbor information,  $E_{s}$  and  $E_{h}$ , are used as inputs to the MLP. In the MLP, the data passes through two layers of neurons for dimension transformation, after which the linear input is converted into nonlinear output via an activation function. Finally, the symptom embedding  $G_{s}^{ss}$  and the herb embedding  $G_{h}^{hh}$  are obtained, as shown in Eqs. (21) and (22).

$$
G_{s}^{ss} = \sigma (W^{l}\cdot \sigma (W^{l - 1}E_{s} + b^{l - 1}) - b^{l}) \tag{21}
$$

$$
G_{h}^{hh} = \sigma (W^{l}\cdot \sigma (W^{l - 1}E_{h} + b^{l - 1}) + b^{l}) \tag{22}
$$

Here,  $W$  is the weight matrix,  $b$  is the bias vector, and  $\sigma (\cdot)$  is the nonlinear activation function (sigmoid) used to enhance the learning capability of the model.

# Algorithm 3 The workflow for HANL module.

Require: SS- graph, HH- graph

Ensure: The embedding of symptoms  $G_{s}^{ss}$  and the embedding of herbs  $G_{h}^{hh}$  1: Obtain  $HeS$  of herbs from HH- graph according to Eq. (11), and obtain  $HoS$  from SH- graph according to Eq. (12). Obtain  $HeS$  and  $HoS$  of symptoms in the same way 2:Fuse  $HeS$  and  $HoS$  of herbs to get  $Sim_{- hh}$  according to Eq. (15), and fusion of the  $HeS$  and  $HoS$  of symptoms to get  $Sim_{- ss}$  according to Eq. (16) 3: Use Eq. (13) to merge  $Sim_{- ss}$  and  $A_{ss}$  to obtain  $A_{SS}^{\prime}$  4: Use Eq. (14) to merge  $Sim_{- hh}$  and  $A_{hh}$  to obtain  $A_{HH}^{\prime}$  5: Use  $Z_{s}$  and  $Z_{h}$  as the initial embeddings in the SS- graph and HH- graph, and aggregate neighbor information using Eqs. (17) and (18), respectively 6: Obtain  $E_{s}$  via Eq. (19) and obtain  $E_{h}$  via Eq. (20) 7: Apply Eqs. (21) and (22) to pass  $E_{s}$  and  $E_{h}$  through a MLP to obtain  $G_{s}^{ss}$  and  $G_{h}^{hh}$ , respectively 8: return  $G_{s}^{ss}$  and  $G_{h}^{hh}$

# 3.5. Hierarchical structure network learning

In the HSNL module, herbs (symptoms) are divided into three levels by using soft K- Means, and different levels are processed by GCN of varying depths. Since high- frequency nodes appear more frequently in the data, the model uses a shallower network structure to capture their features quickly. In comparison, low- frequency nodes require a deeper network structure to extract potential feature information due to sparse data fully. These features are then integrated to generate the final herb (symptom) representations  $G_{h}(G_{s})$ . The algorithm flow of the HSNL module is shown in Algorithm 4.

# 3.5.1. Division of node levels

Analysis of the training set reveals a long- tail distribution in the occurrence frequencies of both symptoms and herbs, as depicted in Fig. 3. Herbs with higher popularity tend to be preferentially selected during the model training phase, potentially skewing recommendations towards these more commonly used herbs and reducing the diversity of the suggested prescriptions. To counteract this trend and promote a broader spectrum of recommendations, we focus on increasing the representation of rarer nodes. Using soft K- Means clustering (Waqas et al., 2023), we categorize the nodes into three distinct levels based on their frequency. This stratification allows for a hierarchical network structure where nodes at different levels undergo varied neighbor aggregation processes, specifically aimed at enriching the representation of less common nodes.

# 3.5.2. Neighbor information aggregation at different levels

Upon stratification, it becomes evident that the nodes exhibit a pyramid- shaped distribution, with the apex representing the most popular nodes and the base comprising the less popular ones. This hierarchical arrangement necessitates tailored aggregation strategies for each level. Top- level nodes undergo aggregation of one layer of neighbors, middle- level nodes two layers, and bottom- level nodes three layers. Through this hierarchical aggregation strategy, we extract node features  $G_{s}$  and  $G_{h}$  from the hierarchical structure network.

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-16/8e35f7e7-3d91-4213-bd78-444a97ab04b2/4ac9aa0061d2222a50202d747a3ebb26aa5d28e1c18c40c5f9d508637da7a64a.jpg)  
Fig. 3. Herbs and symptoms frequency distribution. The vertical axes show the frequency of herbs (symptoms) in the datasets, and the horizontal axes show their IDs. (a) displays herb frequency, while (b) shows symptom frequency.

# 3.6. Prescriptions recommendation

In the PR phase, the features output by the HSNL module are further integrated with the features from the previous HANL module to generate unified herbal features  $Q_{h}$  and symptom features  $Q_{s}$ . The one- hot encoded patients are then mapped to the herbal space to generate a patient- level representation  $y_{pre}$ , which is compared with the actual prescription  $y$  to obtain the prediction loss.

All features of symptoms and herbs are fused separately to obtain the final symptom embedding  $Q_{s}$  and herb embedding  $Q_{h}$ , as shown in Eqs. (23) and (24).

$$
\begin{array}{l}{Q_s = \frac{1}{2} (G^{ss} + G_s)}\\ {Q_h = \frac{1}{2} (G_h^{hh} + G_h)} \end{array} \tag{23}
$$

We use one- hot encoding to represent the interaction relationship between prescriptions and symptoms to obtain  $P_{s} \in \mathbb{R}^{I \times M}$ . Then, we obtain the embedding  $Z_{s_{- set}}$  of prescriptions as shown in Eq. (25). Next, we interact with the final embedding  $Q_{h}$  of herbs and pass it through a sigmoid activation function to get the final prediction results  $y_{pre} \in \mathbb{R}^{I \times n}$ , as shown in Eq. (26).

Table 1 Basic data for both datasets.  

<table><tr><td></td><td>All</td><td>Symptoms</td><td>Herbs</td><td>Train</td><td>Test</td></tr><tr><td>Dataset1</td><td>33 765</td><td>390</td><td>805</td><td>20 259</td><td>13 056</td></tr><tr><td>Dataset2</td><td>26 360</td><td>360</td><td>753</td><td>22 917</td><td>3443</td></tr></table>

is quantified using BCEWithLogitsLoss, as detailed in Eq. (27). To improve the model's training efficiency and facilitate convergence, the Adam optimizer is used.

$$
L_{pre} = L(y_{pre},y) \tag{27}
$$

Here,  $L(\cdot)$  is the BCEWithLogitsLoss function. By integrating Sigmoid and Binary- Cross Entropy (BCF), BCEWithLogitsLoss directly processes the original model outputs during the training phase, thereby leading to more stable calculation results.  $y$  represents the actual prescriptions.

# 3.7.2. Loss function based on BPR

Reflecting the specific demands of the recommendation task, the BPR loss is utilized to further optimize the model. This loss function is outlined in Eq. (28).

$$
L_{BPR} = -\sum_{i \in M, j, q \in N} \log \sigma (y_{i, j} - y_{i, q}) \tag{23}
$$

Here,  $y_{i, j}$  represents the similarity score between the positive samples of symptoms  $s_{i}$  and herbs  $h_{j}$ , while  $y_{i, q}$  represents the similarity score between the negative samples of symptoms  $s_{i}$  and herbs  $h_{q}$ . The function  $\sigma (\cdot)$  is the sigmoid activation function. Based on this, the fusion of the two losses forms the final loss, as shown in Eq. (29).

$$
Loss = L_{BPR} + L_{pre} \tag{29}
$$

# 4. Experimental results and analyses

In Section 4.1, we introduce the experimental dataset. Next, in Section 4.2, we compare our method with existing models. We describe the evaluation metrics in Section 4.3, while Section 4.4 details the experimental setup. The experimental results are presented in Section 4.5. Section 4.6 conducts correlation coefficients and significance test for all models, followed by an analysis of the effects of each module in Section 4.7. In Section 4.8, we discuss the impact of hyperparameters. Finally, in Section 4.9, we analyze three cases.

# 4.1. Datasets

We use two public datasets from KDHR (Yang et al., 2022) and SMGCN (Jin et al., 2020), referred to as Dataset1 and Dataset2, respectively, in Table 1. In the two datasets, a piece of medical record can contain up to 33 types of herbs, corresponding to 33 labels.

# 4.2. Baselines

We compare TCMRGCL with the following baselines:

SMGCN (Jin et al., 2020) constructed SH- graph, SS- graph, and HH- graph based on prescription data characteristics and used a 2- layer GCN for feature extraction. It integrated the symptom set into a syndrome for symptom- aware in vivo TCM prescription recommendations.

KDHR (Yang et al., 2022) acknowledged herbs' unique properties. While performing convolution operations on SH- graph, SS- graph, and HH- graph with GCN, it integrated property information as features, obtaining symptom and herb features separately for prescription recommendations.

Algorithm 4 The workflow for HSNL module.  

<table><tr><td>Require:</td><td>Prescription dataset P, SH-graph</td></tr><tr><td>Ensure:</td><td>The embedding of symptoms Gs and the embedding of herbs Gh</td></tr><tr><td>1:</td><td>Use soft K-means clustering to divide symptoms into three layers based on their frequency of occurrence in prescription P.</td></tr><tr><td>2:</td><td>Use soft K-means clustering to divide herbs into three layers based on their frequency of occurrence in prescription P.</td></tr><tr><td>3:</td><td>Perform one-layer neighbor aggregation for high-prevalence nodes, two-layer neighbor aggregation for medium-prevalence nodes, and three-layer neighbor aggregation for low-prevalence nodes in SH-graph to obtain Gs and Gh</td></tr><tr><td>4:</td><td>return Gs and Gh</td></tr></table>

# 3.7. Construct loss function

# 3.7.1. Loss function based on recommendation results

After obtaining the herbal score matrix  $y_{pre}$ , the discrepancy between the model's predicted prescriptions and the actual prescriptions SMRGAT (Yang & Ding, 2023) employed a multi- head attention mechanism to differentiate the effects of herbs on symptoms and used a residual network to expand entity features. It integrated 23- dimensional herbal attributes to derive final symptom and herb embeddings.

Table 2 The main features and results of TCMRGCL compared to baseline models.  

<table><tr><td>Methods</td><td>Advantages</td><td>Disadvantages</td><td>Precision</td><td>Recall</td><td>F1-score</td></tr><tr><td>SMGCN</td><td>Capture complex relationships between symptoms and herbs by using a multi-graph convolutional network.</td><td>Leave the paired compatibility of herbs out of consideration.</td><td>Dataset1:0.1551
Dataset2:0.2276</td><td>Dataset1:0.1483
Dataset2:0.2321</td><td>Dataset1:0.1656
Dataset2:0.2512</td></tr><tr><td>KDHR</td><td>Leverage the knowledge graph to capture and integrate the rich relationships between symptoms and herbs.</td><td>Take no account of the interactions between herbs.</td><td>Dataset1:0.1683
Dataset2:0.2113</td><td>Dataset1:0.1682
Dataset2:0.2145</td><td>Dataset1:0.1841
Dataset2:0.2325</td></tr><tr><td>SMRGAT</td><td>Leverage the multi-graph residual attention network to capture the nonlinear relationships between symptoms and herbs effectively.</td><td>Lack of consideration for herbal dosage in the recommendations.</td><td>Dataset1:0.1892
Dataset2:0.2147</td><td>Dataset1:0.1980
Dataset2:0.2158</td><td>Dataset1:0.2112
Dataset2:0.2350</td></tr><tr><td>SCEIKG</td><td>Use the sequential condition evolution and interaction knowledge graph to improve the accuracy of herbal recommendations.</td><td>Lack of explicit consideration for herbal dosage and the implicit nature of patient state transitions.</td><td>Dataset1:0.1957
Dataset2:0.2182</td><td>Dataset1:0.1953
Dataset2:0.2203</td><td>Dataset1:0.2136
Dataset2:0.2398</td></tr><tr><td>PresRecST</td><td>Utilize a structured approach to enhance the accuracy and relevance of herbal prescriptions by aligning with clinical diagnostics.</td><td>Neglect to consider the comprehensive syndrome differentiation and treatment.</td><td>Dataset1:0.1759
Dataset2:0.2237</td><td>Dataset1:0.1713
Dataset2:0.2272</td><td>Dataset1:0.1895
Dataset2:0.2466</td></tr><tr><td>TCMRGCL</td><td>Obtain prior knowledge through contrastive pre-training. Strengthen node features by combining homogeneous networks with dual similarity features and a hierarchical structure network.</td><td>Overlook the herbal dosage and compatibility.</td><td>Dataset1:0.2001
Dataset2:0.2294</td><td>Dataset1:0.2057
Dataset2:0.2346</td><td>Dataset1:0.2221
Dataset2:0.2537</td></tr></table>

SCEIKG (Liu et al., 2023) posited that TCM diagnosis and treatment consider the patient's status and symptoms over time. As such, SCEIKG made TCM prescription recommendations by comprehensively sensing the evolving status of the patient's condition.

PresRecST (Dong et al., 2024) implemented a progressive diagnosis and treatment workflow by using residual neural networks and a TCM knowledge graph.

TCMRGCL constructs two augmented views for contrastive pretraining to obtain initial symptom and herb embeddings. We explore hidden association information in the SS- graph and HH- graph by fusing node similarity information. A hierarchical structure network categorizes nodes into different levels and implements differentiated neighbor aggregation operations. These measures aim to improve the performance and accuracy of the recommendation system, thereby generating more comprehensive prescriptions.

In Table 2, we provide a detailed comparison of the baseline models' strengths, weaknesses, and metric results with TCMRGCL. Precision is calculated as the average of R@5,  $\mathrm{P@10}$  and  $\mathrm{P@20}$  Recall is the average of  $\mathbb{R}(\mathbb{Q}5,\mathbb{R}(\mathbb{Q}10,$  and  $\mathbb{R}(\mathbb{Q}20;$  and the F1- score is the average of  $\mathrm{F1@5}$ $\mathrm{F1@10}$  and  $\mathrm{F1@20}$  Although TCMRGCL does not incorporate auxiliary knowledge or focus on the patients' state evolution like other models, it still achieves commendable results. This may be because TCMRGCL leverages prior knowledge obtained through contrastive pre- training and enhances node features by integrating homogeneous networks with a hierarchical structure network. Unlike other models, TCMRGCL focuses on the similarity between symptoms and herbs and captures more comprehensive entity features through its multi- layered network structure. This multi- faceted feature representation enables TCMRGCL to reflect the complex relationships between herbs and symptoms more accurately, thereby improving predictive performance.

# 4.3. Evaluation metrics

In this paper, we consider Precision@K, Recall@K and F1 - score@K as the evaluation indicators for the TCMRGCL model. The formulas are given in Eqs. (30), (31) and (32) respectively, with  $K$  representing the number of recommended herbs. In the experiment, we set  $K$  to5,10,20.

$$
\begin{array}{rl} & {Precision@K = \frac{|Top(V_{S_{set}},K)\cap V_{H_{set}}|}{K}}\\ & {Recall@K = \frac{|Top(V_{S_{set}},K)\cap V_{H_{set}}|}{|V_{H_{set}}|}}\\ & {F1 - score@K = 2\times \frac{precision@K\times recall@K}{(precision@K + recall@K)}} \end{array} \tag{31}
$$

Here, Precision@K, Recall@K, and F1- score@K represent the precision, recall, and F1- score of the recommended  $K$  herbs.  $Top(V_{S_{set}},K)$  represents the set of the top  $K$  herbs predicted for a given symptom set  $V_{S_{set}}$  and  $V_{H_{set}}$  represents the actual TCM prescriptions. The improvement in precision indicates a higher proportion of truly effective herbs among those recommended by the model. This is crucial for clinical use, as it allows doctors to more confidently adopt the recommendations, reducing the risk of using ineffective or inappropriate herbs and minimizing potential adverse effects. Moreover, the increase in precision suggests the model is better at identifying relevant herbs, showing enhanced robustness against noisy data.

The improvement in recall means the model identifies more effective herbs, ensuring that essential herbs are not overlooked, which is vital for treating complex diseases. This broader identification provides a more comprehensive treatment plan and demonstrates the model's broad applicability.

The F1- score, the harmonic mean of precision and recall, reflects a balance. A high F1- score indicates that the model maintains high precision while also achieving high recall, ensuring optimal recommendation accuracy and comprehensiveness performance, thereby maximizing the effectiveness of prescription recommendations.

# 4.4. Experimental settings

We set the learning rate to 2e- 3, the embedding dimension to {64, 128, 256, 512}, the number of model layers in the CPreT module to 2, and the epochs to {100, 200, 300, 400, 500}. The batch size is set to 128, the dropout rate to 0.1, the truncation value  $r$  to 5, and the temperature coefficient to 0.2.

Table 3 Performance on Dataset1 and Dataset2 compared with the baseline models.  

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td colspan="3">Precision</td><td colspan="3">Recall</td><td colspan="3">F1 score</td></tr><tr><td>P@5</td><td>P@10</td><td>P@20</td><td>R@5</td><td>R@10</td><td>R@20</td><td>F1@5</td><td>F1@10</td><td>F1@20</td></tr><tr><td rowspan="7">Dataset1</td><td>SMGCN</td><td>0.1933</td><td>0.1568</td><td>0.1153</td><td>0.1265</td><td>0.2031</td><td>0.3027</td><td>0.1529</td><td>0.1770</td><td>0.1670</td></tr><tr><td>KDHR</td><td>0.2138</td><td>0.1660</td><td>0.1251</td><td>0.1510</td><td>0.2284</td><td>0.3414</td><td>0.1770</td><td>0.1922</td><td>0.1832</td></tr><tr><td>SMRGAT</td><td>0.2461</td><td>0.1873</td><td>0.1343</td><td>0.1821</td><td>0.2777</td><td>0.3970</td><td>0.2093</td><td>0.2237</td><td>0.2007</td></tr><tr><td>SCEIKG</td><td>0.2482</td><td>0.1942</td><td>0.1446</td><td>0.1722</td><td>0.2691</td><td>0.3973</td><td>0.1035</td><td>0.2256</td><td>0.2119</td></tr><tr><td>PresRecST</td><td>0.2238</td><td>0.1749</td><td>0.1290</td><td>0.1512</td><td>0.2338</td><td>0.3465</td><td>0.2803</td><td>0.2001</td><td>0.1879</td></tr><tr><td>TCMRGCL</td><td>0.2543</td><td>0.1984</td><td>0.1477</td><td>0.1852</td><td>0.2841</td><td>0.4192</td><td>0.2143</td><td>0.2337</td><td>0.2184</td></tr><tr><td>Improve</td><td>2.16%</td><td>2.16%</td><td>2.14%</td><td>1.79%</td><td>2.30%</td><td>5.51%</td><td>2.39%</td><td>3.59%</td><td>3.07%</td></tr><tr><td rowspan="7">Dataset2</td><td>SMGCN</td><td>0.2878</td><td>0.2287</td><td>0.1664</td><td>0.2070</td><td>0.3229</td><td>0.4631</td><td>0.2409</td><td>0.2677</td><td>0.2449</td></tr><tr><td>KDHR</td><td>0.2698</td><td>0.2103</td><td>0.1537</td><td>0.1932</td><td>0.2967</td><td>0.4299</td><td>0.2251</td><td>0.2461</td><td>0.2264</td></tr><tr><td>SMRGAT</td><td>0.2710</td><td>0.2157</td><td>0.1574</td><td>0.1892</td><td>0.3009</td><td>0.4334</td><td>0.2228</td><td>0.2512</td><td>0.2309</td></tr><tr><td>SCEIKG</td><td>0.2735</td><td>0.2189</td><td>0.1621</td><td>0.1921</td><td>0.3066</td><td>0.4482</td><td>0.2257</td><td>0.2555</td><td>0.2381</td></tr><tr><td>PresRecST</td><td>0.2807</td><td>0.2248</td><td>0.1656</td><td>0.1988</td><td>0.3173</td><td>0.4618</td><td>0.2327</td><td>0.2632</td><td>0.2438</td></tr><tr><td>TCMRGCL</td><td>0.2884</td><td>0.2310</td><td>0.1689</td><td>0.2082</td><td>0.3266</td><td>0.4707</td><td>0.2418</td><td>0.2706</td><td>0.2486</td></tr><tr><td>Improve</td><td>0.21%</td><td>1.01%</td><td>1.50%</td><td>0.57%</td><td>1.15%</td><td>1.64%</td><td>0.37%</td><td>1.08%</td><td>1.51%</td></tr></table>

Table 4 Result of ablation experiments on Dataset1 and Dataset2.  

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td colspan="3">Precision</td><td colspan="3">Recall</td><td colspan="3">F1-score</td></tr><tr><td>P@5</td><td>P@10</td><td>P@20</td><td>R@5</td><td>R@10</td><td>R@20</td><td>F1@5</td><td>F1@10</td><td>F1@20</td></tr><tr><td rowspan="4">Dataset1</td><td>TCMRGCL-NoCpreT</td><td>0.2517</td><td>0.1975</td><td>0.1448</td><td>0.1799</td><td>0.2804</td><td>0.4075</td><td>0.2098</td><td>0.2317</td><td>0.2136</td></tr><tr><td>TCMRGCL-NoHANL</td><td>0.2310</td><td>0.1827</td><td>0.1387</td><td>0.1613</td><td>0.2528</td><td>0.3870</td><td>0.1899</td><td>0.2121</td><td>0.2042</td></tr><tr><td>TCMRGCL-NoHSNL</td><td>0.2508</td><td>0.1979</td><td>0.1468</td><td>0.1852</td><td>0.2870</td><td>0.4209</td><td>0.2131</td><td>0.2343</td><td>0.2177</td></tr><tr><td>TCMRGCL</td><td>0.2543</td><td>0.1984</td><td>0.1477</td><td>0.1852</td><td>0.2841</td><td>0.4192</td><td>0.2143</td><td>0.2337</td><td>0.2184</td></tr><tr><td rowspan="4">Dataset2</td><td>TCMRGCL-NoCpreT</td><td>0.2801</td><td>0.2199</td><td>0.1624</td><td>0.1981</td><td>0.3100</td><td>0.4492</td><td>0.2320</td><td>0.2572</td><td>0.2386</td></tr><tr><td>TCMRGCL-NoHANL</td><td>0.2461</td><td>0.2001</td><td>0.1458</td><td>0.1689</td><td>0.2751</td><td>0.4007</td><td>0.2003</td><td>0.2316</td><td>0.2139</td></tr><tr><td>TCMRGCL</td><td>0.2847</td><td>0.2263</td><td>0.1649</td><td>0.2017</td><td>0.3204</td><td>0.4585</td><td>0.2361</td><td>0.2652</td><td>0.2426</td></tr><tr><td>TCMRGCL</td><td>0.2884</td><td>0.2310</td><td>0.1689</td><td>0.2082</td><td>0.3266</td><td>0.4707</td><td>0.2418</td><td>0.2706</td><td>0.2486</td></tr></table>

# 4.5. Performance comparison of the two datasets

The TCMRGCL is evaluated using three metrics, based on tests conducted on the two datasets, as shown in Table 3. On Dataset1, TCMRGCL achieves a significant improvement in F1- score compared to the existing best model, SCEIKG, with increases of  $2.39\%$  on F1- score@5,  $3.59\%$  on F1- score@10, and  $3.07\%$  on F1- score@20. This may be because contrastive pre- training on enhanced views with TSVD and edge dropout allows the model to better distinguish between relevant and irrelevant features, resulting in high- quality features that are both robust and informed by prior knowledge. On Dataset2, TCMRGCL also achieves a slight improvement in F1- score compared to SCEIKG, with increases of  $0.37\%$  on F1- score@5,  $1.08\%$  on F1- score@10, and  $1.51\%$  on F1- score@20. One possible reason is that the focus on dual similarity in the homogeneous network takes into account multiple aspects of entity relationships, while the hierarchical structure network effectively addresses the issue of data scarcity and variability. The combination of these two elements captures both local and global patterns. The improvement on Dataset2 is not as pronounced as on Dataset1, which may be attributed to differences in the distribution of the two datasets. However, overall, the TCMRGCL model proves to be effective.

It is worth noting that, all indexes such as precision, recall and F1score of the existing TCM prescription recommendation methods are suboptimal and can be attributed to several key factors. Firstly, the complexities of the TCM recommendation task. Unlike conventional recommendation systems that typically deal with single items, TCM prescriptions involve multiple herbs that can be combined in myriad ways. Second is the sparse and imbalanced nature of the symptoms and herbs data; Third, most current methods do not account for herb pairs' synergistic or counteracting effects.

# 4.6. Correlation coefficients and significance test

To support our claims of superiority, we perform a statistical significance test on Dataset1. Fig. 4(a) illustrates the correlation coefficients between each method's final predicted score matrices and the original herb- symptom matrix. Using a significance threshold of  $|r| > 0.10$  (with  $n = 390$  ), our analysis concentrates on identifying linear correlations. Notably, the TCMRGCL method achieve the highest correlation value of 0.4226, outperforming the baseline methods.

We validate the robustness of these correlations by conducting a significance analysis using a t- test. This test is applied specifically to correlation coefficients greater than 0.10, as our dataset of 390 entries results in 388 degrees of freedom. A t- value exceeding 1.90 indicates a probability of less than 0.05 that the results occurred by chance. Fig. 4(b) shows the corresponding t- values for r- values above 0.1, surpassing the 1.90 threshold. This confirms that the correlations are statistically significant, ruling out the likelihood of coincidental occurrence.

# 4.7. Ablation analysis

Based on our TCMRGCL model, we conducted an ablation analysis on three important modules across two datasets, as detailed in Table 4, to determine the impact of these modules on the final model performance. (1) The TCMRGCL- NoCPreT model, which lacks the contrastive pre- training module, shows a decline in prediction results across both datasets. This suggests that node embeddings derived from contrastive pre- training effectively capture the structural information within the graph, aiding in the accurate modeling of the nonlinear relationships between symptoms and herbs, thereby improving the model's overall performance. (2) The TCMRGCL- NoHANL model, which omits the homogeneous association network, also performs less effectively than TCMRGCL on both datasets. This may be attributed to the sparse interaction data between symptoms and herbs, where the two homogeneous graphs provide additional information, enriching the network. By incorporating the similarity matrix, the model can bring semantically similar yet initially distant nodes closer in the feature space, further explaining the suboptimal performance of the TCMRGCL- NoHANL variant. (3) Similarly, the TCMRGCL- NoHSNL model, excluding the hierarchical structure network module, experiences a drop in performance. The hierarchical structure network is instrumental in managing nodes of

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-16/8e35f7e7-3d91-4213-bd78-444a97ab04b2/cf46f876f999fe15151fc92a157930b673ffd067c29fcbd522aa69a13ff34ea0.jpg)  
Fig. 4. The correlation coefficients (a) and t-test values (b) compared for the Dataset1.

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-16/8e35f7e7-3d91-4213-bd78-444a97ab04b2/86b21a9dcb493247bc90ad40bd70d2338e2cd8e28b7639bdfe3a5df21ef49302.jpg)  
Fig. 5. Visual representation of symptom and herb embeddings: (a) and (b) represent the herb and symptom embeddings of TCMRGCL, respectively. (c) and (d) represent the herb and symptom embeddings of TCMRGCL-NoCpreT, respectively. (e) and (f) represent the herb and symptom embeddings of TCMRGCL-NoHANL, respectively. (g) and (h) represent the herb and symptom embeddings of TCMRGCL-NoLSNL, respectively.

varying popularity levels and in the comprehensive learning of deeper features. These findings validate the significance of each TCMRGCL component and the model's overall efficacy.

Furthermore, to evaluate the model's recommendation performance more thoroughly, for Dataset2, we visualize the symptom and herb embeddings using t- SNE and gaussian kernel density estimation (KDE) as illustrated in Fig. 5. We map symptoms and herbs of different models into two- dimensional space. The embedding distributions of herbs (a) and symptoms (b) generated by the TCMRGCL model are relatively compact, forming clear clusters, and there are also obvious distinctions between different clusters, which shows that the TCMRGCL model can cluster similar herbs and symptoms together while distinguishing different characteristics of herbs and symptoms, which helps to recommend prescriptions more accurately. The embedding distributions of herbs (c) and symptoms (d) of the TCMRGCL- NoCpreT model are more scattered than those of (a) and (b), and the compactness of the clusters is poor, resulting in reduced accuracy of recommendations in practical applications. The embedding distributions of herbs (e) and symptoms (f) of the TCMRGCL- NoHANL model presents an uncommon shape, which may be because the removal of the HANL module causes the model to perform poorly when dealing with complex relationships. Similarly, the embedding distributions (g) and (h) generated by the TCMRGCL- NoLSNL model also has an imbalance problem, indicating that the lack of the LSNL module will also affect the capture of features, thereby affecting the quality of recommendations. The comparison of the various modules in Fig. 5 highlights the superiority of the TCMRGCL model.

# 4.8. Effect of hyperparameters

In Figs. 6 and 7, the left vertical axis represents the precision rate, which is illustrated by a bar graph. The bars in different colors represent various precision indicators (such as  $\mathrm{P@5}$ $\mathrm{P@10}$  and  $\mathrm{P@20}$  The right vertical axis represents the recall rate, depicted by a line graph. The lines in different colors correspond to different recall indicators (such as  $\mathrm{R@5}$ $\mathrm{R@10}$  and  $\mathrm{R@20}$

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-16/8e35f7e7-3d91-4213-bd78-444a97ab04b2/ebb792bfc929025221de2e30aeba85e56e850d6657966a42b2fed93c55f35eca.jpg)  
Fig. 6. Analysis of epoch. Epoch represents the number of model training iterations. The two figures compare the performance of different datasets across various epoch values.

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-16/8e35f7e7-3d91-4213-bd78-444a97ab04b2/9779b905349ad2982c27948031b9335537b63bc51f04361f0f2dc871b3f32f12.jpg)  
Fig. 7. Analysis of dimension. Dim represents the vector dimension used to represent the entity. The two figures compare the performance of different datasets across various dimensions.

# 4.8.1. Effect of epoch

As depicted in Fig. 6, we vary the epoch values in the set 100, 200, 300, 400, 500. It is observed that for Dataset1, the optimal results across various metrics are achieved at an epoch value of 200, while for Dataset2, the highest evaluation metrics are observed at an epoch value of 300.

# 4.8.2. Effect of dimension

Referencing Fig. 7, the embedding dimensions are set to 64, 128, 256, 512. The figure indicates that at a dimension of 512, the model's evaluation metrics reach their peak for both Dataset1 and Dataset2.

# 4.9. Case studies

To examine explicitly the recommendation results of TCMRGCL, we display the real prescriptions corresponding to the symptom set and the recommended prescriptions given by the model based on the test set of Dataset2. The following three case studies of herbal formulae generated are provided as examples. The recommended prescription, targets and GO enrichment with which the three case studies are analyzed.

# 4.9.1. Case 1

In Case 1, we recommended 18 herbs for three symptoms and found that 77.8% of them matched the actual prescription in Table 5. The herbs highlighted in red in the table represent those correctly predicted. To further investigate the differences between the actual prescription and the recommended prescription in treating the symptoms, we extracted all targets related to the symptoms, the actual prescription, and the recommended prescription from SymMap (Wu et al., 2018). By merging and deduplicating the targets of the symptoms and the actual prescription, we obtained the Symptoms- Ground truth shown in

Table 5 Comparison between TCMRGCL recommended prescription and Ground truth for Case 1.  

<table><tr><td>Symptom</td><td>Ground truth</td><td>TCMRGCL</td></tr><tr><td>109(Red-white vaginal discharge)[赤白带]</td><td>32(Ligusticum chuanxiong)[川芎]</td><td>16(Angelica sinensis)[当归]</td></tr><tr><td>27(Leucorrhea)[白带]</td><td>116(Atractylodes macrocephala)[白术]</td><td>32(Ligusticum chuanxiong)[川芎]</td></tr><tr><td>30(Cold abdomen)[腹冷]</td><td>282(Foeniculum vulgare)[小茴香]</td><td>83(Aconitum carmichaelii)[附子]</td></tr><tr><td></td><td>108(Cinnamomum cassia)[肉桂]</td><td>68(Paeonia lactiflora)[白芍]</td></tr><tr><td></td><td>52(Artemisia argy)[艾叶]</td><td>118(Zingiber officinale)[干姜]</td></tr><tr><td></td><td>15(Glycyrrhiza uralensis)[甘草]</td><td>27(Wolfiporia externa)[茯苓]</td></tr><tr><td></td><td>118(Zingiber officinale)[干姜]</td><td>87(Cyperus rotundus)[香附]</td></tr><tr><td></td><td>16(Angelica sinensis)[当归]</td><td>78(Saussurea costus)[木香]</td></tr><tr><td></td><td>68(Paeonia lactiflora)[白芍]</td><td>116(Atractylodes macrocephala)[白术]</td></tr><tr><td></td><td>280(Piper nigrum)[胡椒]</td><td>15(Glycyrrhiza uralensis)[甘草]</td></tr><tr><td></td><td>41(Panax ginseng)[人参]</td><td>41(Panax ginseng)[人参]</td></tr><tr><td></td><td>187(Anethum graveolens)[茴香]</td><td>89(Paeonia veitchii)[芍药]</td></tr><tr><td></td><td>27(Wolfitopia externa)[茯苓]</td><td>187(Anotum graveolens)[茴香]</td></tr><tr><td></td><td>83(Aconitum carmichaelii)[附子]</td><td>52(Artemisia argy)[艾叶]</td></tr><tr><td></td><td>28(Atractylodes lancea)[苍术]</td><td>11(Ostrea gigas)[杜蛎]</td></tr><tr><td></td><td>204(Eindera aggregata)[乌药]</td><td>11(Evodia rutaecarpa)[吴茱萸]</td></tr><tr><td></td><td>111(Evodia rutaecarpa)[吴茱萸]</td><td>108(Cinnamomum cassia)[肉桂]</td></tr><tr><td></td><td>87(Cyperus rotundus)[香附]</td><td>107(Os Draconis)[龙骨]</td></tr></table>

Table 6 Symptoms-Ground truth-TCMRGCL target comparison for Case 1.  

<table><tr><td></td><td>Symptoms-Ground truth</td><td>Symptoms-TCMRGCL</td></tr><tr><td>Target</td><td>ABCC2/ALAD/APC/CTNNB1/IL10/TCF4/
IL6/JAK2/MYC/PIK3CA/PPOX/PRS1/
STAT4/TLR4/F5/FUS/MIF/SLC12A3/
CPOX/LACC1/MVK/SDHB/SDHC/GHSR/
SEMA3C/SEMA1/EDNC5/NABG8/FSHR/
GDNF/SLCO1B3/ELANE/NME1/BRCA1/
IL12A/POLG/GPR35/HMBS/CCR1/EDNR/
DDIT3/EWSR1/GTF2IRD1/HLA-B/SECG3/
ABCA1/RBPG3/BRCA2/C4A/C4L/CFTD/
CPA1/ECE1/FAS/GLA/KIT/ND5/SMAD4/
TGFBR2/TNFRSF1A/MLH3/CASR/CAAT1/</td><td>ABCC2/ALAD/APC/CTNNB1/IL10/
IL6/JAK2/MYC/PIK3CA/PPOX/PRS1/
STAT4/TLR4/F5/FUS/MIF/SLC12A3/
LACC1/MVK/SDHB/SDHC/SEMA3C/SLCO1B1/
SEMA3D/ELANE/NME1/BRCA1/IL12A/
POLG/GPR35/HMBS/SLC12A3/CCR1/
EDNRB/GHSR/TCF4/DDIT3/EWSR1/SAA1/
GTF2IRD1/HLA-B/SECG3/ABCA1/CASR/
ABCG8/RBPG3/BRCA2/C4A/CLMEL1B1/
CFTR/CPA1/ECE1/FAS/FSHR/TNFRSF1A/
GDNF/GLA/KIT/ND5/SMAD4/TGFBR2/</td></tr></table>

Table 6. Similarly, we obtained the Symptoms- TCMRGCL by merging the targets of the symptoms and the recommended prescription. A comparison of the targets revealed that all targets were completely consistent, with no differential targets, indicating that our recommended prescription is highly effective. Subsequently, we conducted a GO enrichment analysis on these targets to reveal the biological processes through which they exert therapeutic effects on the symptoms. As shown in Fig. 8, IL10 is involved in 11 biological processes, and 13 genes are enriched in the "positive regulation of transcription by RNA polymerase II" process. The GO enrichment analysis chart clearly illustrates the pathways through which these genes function.

To validate the authenticity of the recommended prescription, we reviewed literature (Bian, 2022) and found that in 45 cases of treating leukorrhea, the most frequently used herbs were Atractylodes macrocephala (17 times), Angelica sinensis (17 times), Wolfiporia extensa (17 times), Ostrea gigas (13 times), Paeonia lactiflora (12 times), and Os Draconis (8 times). Similarly, in 31 cases of treating both red and white leukorrhea, the most frequently used herbs were Angelica sinensis (17 times), Wolfiporia extensa (13 times), Paeonia lactiflora (13 times), Atractylodes macrocephala (7 times), and Ostrea gigas (5 times). Additionally, for syndromes induced by cold in the abdomen leading to Leukorrhea and Red- white vaginal discharge, the above herbs were commonly used in treatment.

# 4.9.2. Case 2

Case 2 includes 4 symptoms, 8 real herbs, and 8 herbs recommended by the model, as shown in Table 7. The prescription recommendation accuracy reached  $75\%$ . We also identified 36 intersection genes between the symptoms- ground truth and 30 intersection genes between the symptoms- TCMRGCL in Table 8. There were only 7 differential genes between the two sets, indicating that the two prescriptions may have therapeutic differences in certain biological processes. We conducted a GO analysis on the common genes between the symptoms and the recommended prescriptions, with the enrichment analysis results shown in Fig. 9. The results indicate that this prescription group can treat the symptoms through biological processes such as "negative regulation of neuron apoptotic process", "response to toxic substance" and "female pregnancy".

Based on the fundamental principles of TCM diagnosis and treatment, the causes of night sweats and spontaneous sweating are attributed to deficiencies in both Qi and Yin. The proposed TCM treatment involves using herbs that tonify  $\mathrm{Qi}^{+ + }$  (T), nourish Yin (阴), clear heat, and dissolve phlegm, including Panax ginseng, Largehead atractylodes, Wolfiporia extensa, and Glycyrrhiza uralensis (Zhu, 2022). According to the literature (Wang et al., 2020), Angelica sinensis ranks 11th in frequency of external use for treating scrofula in ancient texts, with 12 recorded instances, accounting for  $11.21\%$ . In the distribution table of the four Qi and five Flavors (味) of external medicines used for Scrofula, the flavors pungent, bitter and sweet are the top three, making up  $84.52\%$  of the total, which aligns with the flavors of the recommended herbs. Additionally, in the meridian distribution table for external medicines used in treating scrofula, the liver, spleen, and heart meridians are the top three, accounting for  $52.94\%$  of the total, consistent with the meridian associations of our recommended herbal formula. Overall, this indicates that the recommended herbal formula is effective.

# 4.9.3. Case 3

In this case, we conducted an analysis similar to the previous two cases. A comparison between the predicted and actual prescriptions is

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-16/8e35f7e7-3d91-4213-bd78-444a97ab04b2/d9c4113c4ff167a5b41574ecdab3e67177550f3580406ebc3f1bf229244cc515.jpg)  
Fig.8. 1.  t  t  t t  t t  t t  t t  t t  t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t h t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t t

Table 7 Comparison between TCMRGCL recommended prescription and Ground truth for Case 2.  

<table><tr><td>Symptom</td><td>Ground truth</td><td>TCMRGCL</td></tr><tr><td>44(Night sweats)[盗汗]</td><td>116(Largehead atractylodes)[白术]</td><td>15(Glycyrrhiza uralensis)[甘草]</td></tr><tr><td>25(Scrofula)[瘰]</td><td>15(Glycyrrhiza uralensis)[甘草]</td><td>16(Angelica sinensis)[当归]</td></tr><tr><td>45(Emaciation)[消瘦]</td><td>16(Angelica sinensis)[当归]</td><td>41(Panax ginseng)[人参]</td></tr><tr><td>21(Spontaneous sweating)[目汗]</td><td>68(Paeonia lactiflora)[白芍]</td><td>27(Wolfiporia externa)[茯苓]</td></tr><tr><td></td><td>4(Honeysuckle flower)[金银花]</td><td>116(Largehead atractylodes)[白术]</td></tr><tr><td></td><td>41(Panax ginseng)[人参]</td><td>11(Ostrea gigas)[牡蛎]</td></tr><tr><td></td><td>85(Ternate pinellia)[丰夏]</td><td>71(Chinese thunowax)[柴胡]</td></tr><tr><td></td><td>71(Chinese thorowax)[柴胡]</td><td>68(Paeonia lactiflora)[白芍]</td></tr></table>

Table 8 Symptoms-Ground truth-TCMRGCL target comparison for Case 2.  

<table><tr><td></td><td>Symptoms-Ground truth</td><td>Symptoms-TCMRGCL</td></tr><tr><td>Target</td><td>BCL2/NHP2/PPARG/RTEL1/UCF2/
SDHB/SDHC/RGS13/CAV1/FOS/DDC/
JAK2/JUP/BDNF/HINT1/IGHMIP2/
IL12B/FGFR3/IKBKONT/ISCN11A/
TRPV3/BMX/BIRC3/CLCF1/TINF2/
GDNF/HLA-B/HPGD/KRT5/MAP2K1/
WNT10A/HNF4A/HNF1A/ZCCHC7/</td><td>BCL2/NHP2/PPARG/RTEL1/UCP2/
SDHB/SDHC/RGS13/CAV1/HINT1/
BDNF/HMBS/BIRC3/IL12B/HPGD/
DDC/GDNF/JUP/IGHMBP2/TRPV3/
HLA-B/MAP2K1/TINF2/SCN11A/
WNT10A/ZCCHC7/FOS/JAK2/KRT5/
CLCF1/</td></tr></table>

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-16/8e35f7e7-3d91-4213-bd78-444a97ab04b2/e4f826cf4fe9935917ba932d523479edfd28a2102b0fec2e8b74d4057a45e501.jpg)  
Fig. 9. Go enrichment analysis sankey plot for Case 2.

Table 9 Comparison between TCMRGCL recommended prescription and Ground truth for Case 3.  

<table><tr><td>Symptom</td><td>Ground truth</td><td>TCMRGCL</td></tr><tr><td>15(Vomiting)[呕吐]</td><td>15(Glycyrrhiza uralensis)[甘草]</td><td>15(Glycyrrhiza uralensis)[甘草]</td></tr><tr><td>41(Acid regurgitation)[否酸]</td><td>24(Citrus reticulata)[陈皮]</td><td>116(Largehead attractylodes)[白木]</td></tr><tr><td>104(Borborygmus)[肠鸣]</td><td>27(Wolfiporia extensa)[茯苓]</td><td>118(Zingiber officinale)[干姜]</td></tr><tr><td></td><td>30(Magnolia Bar)[厚朴]</td><td>30(Magnolia Bar)[厚朴]</td></tr><tr><td></td><td>41(Panax ginseng)[人参]</td><td>78(Saussurea ostus)[木香]</td></tr><tr><td></td><td>81(Patchouli)[霍香]</td><td>119(Cardamomol)[豆蔻]</td></tr><tr><td></td><td>85(Ternate pinella)[半夏]</td><td>75(Clove)[丁香]</td></tr><tr><td></td><td>116(Largehead attractylodes)[白木]</td><td>24(Citrus reticulata)[陈皮]</td></tr><tr><td></td><td>118(Zingiber officinale)[干姜]</td><td>84(Grains of Paradise)[砂仁]</td></tr><tr><td></td><td>119(Cardamomol)[豆蔻]</td><td>41(Panax ginseng)[人参]</td></tr><tr><td></td><td>164(Terminalia chebula)[诃子]</td><td>27(Wolfiporia extensa)[茯苓]</td></tr><tr><td></td><td>233(Alpinia katsumadai)[草豆蔻]</td><td>85(Ternate pinella)[半夏]</td></tr></table>

Table 10 Symptoms-Ground truth-TCMRGCL target comparison for Case 3.  

<table><tr><td></td><td>Symptoms-Ground truth</td><td>Symptoms-TCMRGCL</td></tr><tr><td>Target</td><td>ACSF3/ALDH18A1/ARG1/BCKDHA/CPOX/ CYP11B2/DHCR7/EGFR-F12/GALE/GLA/ HLCS/HMBS/HNF4A/HSD3B2/LPL/MVK NAGS/NDUFAF5/NR3C2/OTC/PMM2/SAA/ SCN1A/SCNN1B/SCNN1C/SLC12A1/SLC25A/ SLC7A7/SSR4/TNFRSF1A/TP53/UCP2</td><td>ACSF3/ARG1/CPOX/GALE/HSD3B2/PCGA/ MVK/NAGS/NDDUFAF5/NR3C2/OTC/PMM2/ SLC22A5/BCDHA/EGFR/HNF1A/HNF4A/ LPL/SCNN1B/SCNN1G/SLC12A1/SLC7A7/ TP53/UCP2/A1AD/A4/TP13/GANF/PCDHFS1/ PPOX/DHCR7/HMBS/CYP11B2/HKCS/SSR4/ ALDH18A1/SAA1/GLA/SCNN1A/TNFRSF1A/ ACADM/ACADVL/ACAT1/CPT2/ESP1/ETFB/ MMAB/NUBPL/STK11/TMEM12B/TRMU/</td></tr></table>

shown in Table 9, with a prediction accuracy of  $75\%$ . During further validation, we could not identify targets related to Acid regurgitation and Borborygmus in SymMap. However, the database indicates that Saussurea costus is a Qi- regulating herb used to treat these conditions, which is encouraging. We then obtained the targets for Vomiting, the actual prescription, and the recommended prescription, extracting the intersection targets between symptoms and the actual prescription, as well as between symptoms and the recommended prescription. The comparison of these targets is shown in Table 10, revealing that our recommended prescription contains more genes (marked in red) involved in treating the symptoms. Subsequently, we performed a GO enrichment analysis on the 51 targets from the symptoms- recommended prescription. The enrichment results, displayed in Fig. 10, show that "SCNN1B", "SCNN1G" and "SCNN1A" are involved in multiple biological processes. Additionally, "phosphorylation", "liver development", "heme biosynthetic process" and "sodium ion homeostasis" are each

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-16/8e35f7e7-3d91-4213-bd78-444a97ab04b2/edd139c7c723e09f51ee80c487df338de6d2f02eff292200a527b63ab8e7dab0.jpg)  
Fig. 10. Go enrichment analysis sankey plot for Case 3.

enriched with five genes, clearly demonstrating the enrichment effects. These findings are of significant value for further research into treatment methods for related symptoms.

Xu (2023) suggested that a weak spleen and stomach are the causes of Vomiting, Acid reflux, and Borborygmus, requiring herbs with an intense and warm nature, such as Zingiber officinale, Evodia rutaecarpa, Cardamom, Grains of paradise, and Clove. Our recommended prescription includes Zingiber officinale, Cardamom, Clove and Grains of Paradise, which were not part of the original prescription. Additionally, SymMap indicates that Saussurea costus can treat Borborygmus and Acid reflux. These elements collectively demonstrate the effectiveness of the recommended prescription.

# 5. Discussion

The herbal prescription recommendation model offers significant advantages but needs addressing limitations. Currently, TCMRGCL does not consider herb dosages, a critical aspect of TCM affecting prescription efficacy and safety. Additionally, it overlooks herb compatibility, which can result in suboptimal or harmful recommendations. Future improvements could involve using deep neural networks to predict optimal dosages based on patient specific features and integrating knowledge graphs to account for herb compatibility. Reinforcement learning can optimize dosage recommendations based on real- time data and patient outcomes.

# 5.1. Potential impacts

TCMRGCL offers significant benefits for TCM practice by enhancing the efficiency of diagnosis and treatment through automation and intelligent recommendations. It provides personalized prescription suggestions, improving treatment effectiveness and patient trust. Additionally, the model standardizes prescriptions, reducing reliance on individual physician experience and serving as a valuable decision- support tool, especially for less experienced practitioners.

# 5.2. Practical applications

From a broader perspective, the herbal prescription recommendation model is likely to impact TCM practice profoundly. It can help researchers analyze large volumes of TCM and symptom data to discover new drug combinations and therapeutic effects, providing fresh ideas for drug development. Additionally, the model can analyze clinical data through AI and machine learning to uncover patterns and connections within TCM theory. This data- driven approach helps to verify and refine traditional TCM theory, promoting its modernization.

# 5.3. Limitations and future work

The herbal prescription recommendation model offers significant advantages but has limitations that need addressing. Currently, the model does not consider herb dosages, a critical aspect of TCM, which

affects the efficacy and safety of prescriptions. Additionally, it overlooks herb compatibility, which can result in suboptimal or harmful recommendations. Future improvements could involve using deep neural networks to predict optimal dosages based on patient- specific features and integrating knowledge graphs to account for herb compatibility. Reinforcement learning could further optimize dosage recommendations based on real- time data and patient outcomes.

# 6. Conclusion

This study introduces a novel TCM prescription recommendation model based on contrastive pre- training and a hierarchical structure network. TCMRGCL uses contrastive pre- training on two augmented views to create a robust and generalizable feature representation, forming a solid foundation for recommendation tasks. The model accurately captures correlations between nodes by integrating similarity information within the SS- graph and HH- graph, enhancing prescription precision. The hierarchical structure network also addresses data scarcity, particularly for rare nodes, by considering node popularity variance. Experimental analyses on two public datasets demonstrate the model's superior performance and validate its effectiveness in generating rational TCM prescriptions. Additionally, network pharmacology exploration supports the scientific validity of the recommendations.

However, the model has some limitations. For instance, it does not sufficiently consider aspects such as herb compatibility, the progression of a patient's condition, and the dosage of herbs. The current TCMRGCL does not account for herbal dosages, While we consider the compatibility of traditional Chinese medicines through the acquisition of isomorphic similarity, this approach may not be entirely comprehensive. Future research will focus on utilizing AI technologies to address these issues, further enhancing the capabilities of TCM through modern technological advancements. By integrating advanced algorithms and data analysis techniques, including the introduction of herbal compatibility rules and dosage information through reinforcement learning, we aim to improve the precision and efficacy of TCM treatments, thereby promoting its modernization and standardization.

# CRediT authorship contribution statement

Hailong Hu: Methodology, Conceptualization, Investigation, Writing - review & editing. Yaqian Li: Methodology, Conceptualization, Software Writing - original draft. Zeyu Zheng: Visualization, Formal analysis, Writing - review & editing. Wenjun Hu: Writing - review & editing, Funding acquisition, Supervision. Riyang Lin: Validation, Visualization. Yanlei Kang: Supervision, Projectadministration.

# Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

# Acknowledgments

This work was supported by Natural Science Foundation of Huzhou city, China (2022YZ15), Huzhou University Excellent Graduate Course Project (YJGX24003) and Huzhou Key Laboratory of Waters Robotics Technology (2022- 3).

# Data availability

Data will be made available on request.

# References

Ahmadian, M., Ahmadian, S., & Ahmadi, M. (2023). RDERL: Reliable deep ensemble reinforcement learning- based recommender system. Knowledge- Based Systems, 263, Article 110289, https://doi.org/10.1016/j.knosys.2023.110289. Bian, S. (2022). The study on the diagnosis and treatment patterns of leucorrhea cases in the Ming and Qing dynasties. Xinjiang Medical University, https://doi.org/10.27433/d.cnki.gxyku.2022.000935 (in Chinese). Chen, Z., Zhang, D., Liu, C., Wang, H., Jin, X., Yang, F., & Zhang, J. (2024). Traditional Chinese medicine diagnostic prediction model for holistic syndrome differentiation based on deep learning. Integrative Medicine Research, 13(1), Article 101019, https://doi.org/10.1016/j.imr.2023.101019. Chu, X., Sun, B., Huang, Q., Peng, S., Zhou, Y., & Zhang, Y. (2020). Quantitative knowledge presentation models of traditional Chinese medicine (TCM): A review. Artificial Intelligence in Medicine, 103, Article 101810, https://doi.org/10.1016/j.artmed.2020.101810. Dong, X., Zhao, C., Song, X., Zhang, L., Liu, Y., Wu, J., Xu, Y., Xu, N., Liu, J., Yu, H., Yang, K., & Zhou, X. (2024). PresRecST: A novel herbal prescription recommendation algorithm for real- world patients with integration of syndrome differentiation and treatment planning. Journal of the American Medical Informatics Association, 31(6), 1268- 1279, https://doi.org/10.1093/jamia/ocae066. Gao, L., Jia, C., & Wang, W. (2020). Recent advances in the study of ancient books on traditional Chinese medicine. World Journal of Traditional Chinese Medicine, 6(1), 61- 66, https://doi.org/10.4103/wjtcm.wjtcm_3.20. Gao, Z., Jiang, C., Zhang, J., Jiang, X., Li, L., Zhao, P., Yang, H., Huang, Y., & Li, J. (2023). Hierarchical graph learning for protein- protein interaction. Nature Communications, 14(1), 1093, https://doi.org/10.1038/s41647- 023- 36736- 1. Gao, C., Zheng, Y., Li, N., Li, Y., Qin, Y., Piao, J., Quan, Y., Chang, J., Jin, D., He, X., & Li, Y. (2023). A survey of graph neural networks for recommender systems: Challenges, methods, and directions. ACM Transactions on Recommender Systems, 1(1), 1- 51, https://doi.org/10.1145/3568022. Guo, Z., Yu, K., Jolfaei, A., Li, G., Ding, F., & Beheshti, A. (2023). Mixed graph neural network- based fake news detection for sustainable vehicular social networks. IEEE Transactions on Intelligent Transportation Systems, 24(12), 15486- 15498, https://doi.org/10.1109/TITS.2022.3185013. Hua, R., Dong, X., Wei, L., Shu, Z., Yang, P., Hu, Y., Zhou, S., Sun, H., Yan, K., Yan, X., Chang, K., Li, X., Bai, Y., Zhang, R., Wang, W., & Zhou, X. (2024). Lingdan: Enhancing encoding of traditional Chinese medicine knowledge for clinical reasoning tasks with large language models. Journal of the American Medical Information Association, 31(6), 2019- 2020, https://doi.org/10.1093/jamia/ncs- 2027. Huang, K., Zhang, P., Zhang, Z., Youn, J. Y., Wang, C., Zhang, H., & Cai, H. (2021). Traditional Chinese medicine (TCM) in the treatment of COVID- 19 and other viral infections: Efficiencies and mechanisms. Pharmacology & Therapeutics, 225, Article 107843, https://doi.org/10.1016/j.pharmthera.2021.107843. Ji, J., Zhang, B., Yu, J., Zhang, X., Qiu, D., & Zhang, B. (2023). Relationship- aware contrastive learning for social recommendations. Information Sciences, 629, 778- 797, https://doi.org/10.1016/j.ins.2023.02.011. Jin, Y., Ji, W., Shi, Y., Wang, X., & Yang, X. (2023). Meta- path guided graph attention network for explainable herb recommendation. Health Information Science and Systems, 11(1), 5, https://doi.org/10.1007/s13755- 022- 00207- 6. Jin, Y., Ji, W., Zhang, W., He, X., Wang, X., & Wang, X. (2022). A kg- enhanced multi- graph neural network for attentive herb recommendation. IEEE/ACM transactions on computational biology and bioinformatics, 19(5), 2560- 2571, https://doi.org/10.1109/TCBB.2021.3115489. Jin, Y., Zhang, W., He, X., Wang, X., & Wang, X. (2020). Syndrome- aware herb recommendation with multi- graph convolution network. In 2020 IEEE 36th international conference on data engineering (pp. 145- 156). https://doi.org/10.1109/ICDE48307.2020.00020. Jin, Z., Zhang, Y., Miao, J., Yang, Y., Zhuang, Y., & Pan, Y. (2023). A knowledge- guided and traditional Chinese medicine informed approach for herb recommendation. Frontiers of Information Technology & Electronic Engineering, 24(10), 1416- 1429, https://doi.org/10.1631/FTIEE.2200662. Kumar, S., Mallik, A., Khetarpal, A., & Panda, B. (2022). Influence maximization in social networks using graph embedding and graph neural network. Information Sciences, 607, 1617- 1636, https://doi.org/10.1016/j.ins.2022.06.075. Kuo, R., & Li, S. (2023). Applying particle swarm optimization algorithm- based collaborative filtering recommender system considering rating and review. Applied Soft Computing, 135, Article 110038, https://doi.org/10.1016/j.asoc.2023.110038. Li, C., Liu, D., Yang, K., Huang, X., & Lv, J. (2020). Herb- Know: Knowledge enhanced prescription generation for traditional Chinese medicine. In 2020 IEEE international conference on bioinformatics and biomedicine (pp. 1560- 1567). https://doi.org/10.1109/BIBM49941.2020.9313476. Li, R., Wu, S., Tu, J., Peng, L., & Ma, L. (2024). An enhanced graph convolutional network with property fusion for acupoint recommendation. Applied Intelligence: The International Journal of Artificial Intelligence, Neural Networks, and Complex Problem- Solving Technologies, 1- 11, https://doi.org/10.1007/s10489- 024- 05792- 5. Liu, Z., Luo, C., Fu, D., Gui, J., Zheng, Z., Qi, L., & Guo, H. (2022). A novel transfer learning model for traditional herbal medicine prescription generation from unstructured resources and knowledge. Artificial Intelligence in Medicine, 124, Article 102232, https://doi.org/10.1016/j.artmed.2021.102232.

Liu, Z., Yang, J., Chen, K., Yang, T., Li, X., Lu, B., Fu, D., Zheng, Z., & Luo, C. (2024). TCM- KDIF: An information interaction framework, driven knowledge- data and its clinical application in traditional Chinese medicine. IEEE Internet of Things Journal, 1- 15, https://doi.org/10.1109/JIGT.2024.3368029. Liu, J., Zhuo, H. H., Jin, K., Yuan, J., Yang, Z., & Yao, Z. (2023). Sequential condition evolved interaction knowledge graph for traditional Chinese medicine recommendation. arXiv preprint arXiv:2305.17866, https://doi.org/10.48550/arXiv.2305.17866. Lv, Q., Chen, G., He, H., Yang, Z., Zhao, L., Zhang, K., & Chen, C. Y.- C. (2023). TCMBank- the largest TCM database provides deep learning- based Chinese- Western medicine exclusion prediction. Signal Transduction and Targeted Therapy, 8(1), 127, https://doi.org/10.1038/s41392- 023- 1399- 1. Ma, Y., Zhang, X., Gao, C., Tang, Y., Li, L., Zhu, R., & Yin, C. (2023). Enhancing recommendations with contrastive learning from collaborative knowledge graph. Neurocomputing, 523, 103- 115, https://doi.org/10.1016/j.neucom.2022.12.032. Min, S., Gao, Z., Peng, J., Wang, L., Qin, K., & Fang, B. (2021). STGSN- A spatial- temporal graph novel research for active cooking social networks. Knowledge- Based Systems, 214, Article 106746, https://doi.org/10.1016/j.knosys.2021.106746. Niu, Q., Li, H., Tong, L., Liu, S., Zeng, W., Zhang, S., Tian, S., Wang, J., Liu, J., Li, B., Wang, Z., & Zhang, H. (2023). TCMFP: A novel herbal formula prediction method based on network target's score integrated with semi- supervised learning genetic algorithms. Briefings in Bioinformatics, 24(3), bbad102, https://doi.org/10.1093/bib/bbad102. Peng, B., & Lu, M. (2020). From religious manual to herbal pharmacopoeia: a textual study of the formation and transformation of Shennong's Classic Materia Medica. Traditional Medicine Research, 5, 368- 376, https://doi.org/10.53388/TMR20200428177. Tan, Y., Zhang, Z., Li, M., Pan, F., Duan, H., Huang, Z., Deng, H., Yu, Z., Yang, C., Shen, G., Qi, P., Yue, C., Liu, Y., Hong, L., Yu, H., Fan, G., & Tang, Y. (2024). MedChatZH: A tuning LLM for traditional Chinese medicine consultations. Computers in Biology and Medicine, 172, Article 108290, https://doi.org/10.1016/j.compbiomed.2024.108290. Teng, S., Ma, J., Li, Z., Zhou, C., & Lu, W. (2024). State- element- aware syndrome classification based on hypergraph convolutional network. Expert Systems with Applications, 248, Article 123369, https://doi.org/10.1016/j.esw.2024.123369. Wang, S., Huang, E. W., Zhang, R., Zhang, X., Liu, B., Zhou, X., & Zhai, C. (2026). A conditional probabilistic model for joint analysis of symptoms, diseases, and herbs in traditional Chinese medicine patient records. In 2016 IEEE international conference on bioinformatics and biomedicine (pp. 411- 418), https://doi.org/10.1109/BIBM.2016.7822553. Wang, Z., Jin, R., & Gao, J. (2020). Analysis of the medication patterns in external formulas for treating scrofula. Guiding Journal of Traditional Chinese Medicine and Pharmacy, 26(11), 164- 168, https://doi.org/10.13862/j.cnki.cn43- 1446/16.2020.11.037 (in Chinese). Waqas, M., Tahir, M. A., & Khan, S. A. (2023). Robust bag classification approach for multi- instance learning via subspace fuzzy clustering. Expert Systems with Applications, 214, Article 119113, https://doi.org/10.1016/j.eswa.2022.119113. Wu, Y., Zhang, F., Yang, K., Fang, S., Bu, D., Li, H., Sun, L., Hu, H., Gao, K., Wang, W., Zhou, X., Zhao, Y., & Chen, J. (2018). SymMap: An integrative database of traditional Chinese medicine enhanced by symptom mapping. Nucleic Acids Research, 47(D1), D1110- D1117, https://doi.org/10.1093/nar/gky1021. Xu, N. (2023). Study on Zhang Sarni's clinical experience and academic thoughts in treating spleen and stomach diseases. Jiangxi University of Chinese Medicine, https://doi.org/10.27180/d.cnki.gjxzc.2023.000013 (in Chinese). Xu, D., Lu, M., Liu, Y., Chen, W., Yang, X., Xu, M., Zhou, H., Wei, X., Zhu, Y., & Song, Q. (2023). An analysis of the clinical medication rules of traditional Chinese medicine for polycystic variant syndrome based on data mining. Evidence- Based Complementary and Alternative Medicine, 2023(1), Article 6198001, https://doi.org/10.1155/2023/6198001.

Yan, C., Zhang, Y., Zhong, W., Zhang, C., & Xin, B. (2021). A truncated SVD- based ARIA model for multiple QoS prediction in mobile edge computing. Tsinghua Science and Technology, 27(2), 315- 324, https://doi.org/10.26599/TST.2021.9010040. Yang, X., & Ding, C. (2023). SMRGAT: A traditional Chinese herb recommendation model based on a multi- graph residual attention network and semantic knowledge fusion. Journal of Ethnopharmacology, 315, Article 116693, https://doi.org/10.1016/j.jep.2023.116693. Yang, G., Liu, X., Shi, J., Wang, Z., & Wang, G. (2024). TCM- GPT: Efficient pretraining of large language models for domain adaptation in traditional Chinese medicine. Computer Methods and Programs in Biomedicine Update, 6, Article 100158, https://doi.org/10.1016/j.cmpbp.2024.100158. Yang, Y., Rao, Y., Yu, M., & Kang, Y. (2022). Multi- layer information fusion based on graph convolutional network for knowledge- driven herb recommendation. Neural Networks, 146, 1- 10, https://doi.org/10.1016/j.neunet.2021.11.010. Yao, L., Zhang, Y., Wei, B., Zhang, W., & Jin, Z. (2018). A topic modeling approach for traditional Chinese medicine association. IEEE Transactions on Knowledge and Data Engineering, 30(6), 1007- 1021, https://doi.org/10.1109/TKDE.2017.2787158. Yin, Z., Wu, Y., & Zhang, Y. (2022). HGCL: Heterogeneous graph contrastive learning for traditional Chinese medicine prescription generation. In International conference on health information science (pp. 88- 99), https://doi.org/10.1007/978- 3- 031- 20627- 6_9. Yu, J., Xia, X., Chen, T., Cui, L., Hung, N. Q. V., & Yin, H. (2024). XSimGCL: Towards extremely simple graph contrastive learning for recommendation. IEEE Transactions on Knowledge and Data Engineering, 36(2), 913- 926, https://doi.org/10.1109/TKDE.2023.3288135. Yu, J., Yin, H., Xia, X., Chen, T., Li, J., & Huang, Z. (2024). Self- supervised learning for recommender systems: A survey. IEEE Transactions on Knowledge and Data Engineering, 36(1), 335- 355, https://doi.org/10.1109/TKDE.2023.3282907. Zeng, X., Meng, F. F., Li, X., Zhong, K. Y., Jiang, B., & Li, Y. (2024). GHGPR- PPIs: A graph convolutional network for identifying protein- protein interaction site using heat kernel with generalized pagerank techniques and edge self- attention feature processing block. Computers in Biology and Medicine, 168, Article 107683, https://doi.org/10.1016/j.compbiomed.2023.107683. Zhang, X., Ma, H., Yang, F., Li, Z., & Chang, L. (2023). KGCL: A knowledge- enhanced graph contrastive learning framework for session- based recommendation. Engineering Applications of Artificial Intelligence, 124, Article 106512, https://doi.org/10.1016/j.engappai.2023.106512. Zhang, Y., Yin, G., Dong, Y., & Zhang, L. (2023). Contrastive learning with frequency domain for sequential recommendation. Applied Soft Computing, 144, Article 110481, https://doi.org/10.1016/j.asoc.2023.110481. Zhang, Q., Zhou, J., & Zhang, B. (2021). Computational traditional Chinese medicine diagnosis: A literature survey. Computers in Biology and Medicine, 133, Article 104358, https://doi.org/10.1016/j.compbiomed.2021.104358. Zhao, W., Lu, W., Li, Z., Zhou, C., Fan, H., Yang, Z., Lin, X., & Li, C. (2022). TCM herbal prescription recommendation model based on multi- graph convolutional network. Journal of Ethnopharmacology, 297, Article 115109, https://doi.org/10.1016/j.jep.2022.115109. Zhao, Z., Qiang, Y., Yang, F., Hou, X., Zhao, J., & Song, K. (2024). Two- stream computer transformer based multi- label recognition for TCM prescriptions construction. Computers in Biology and Medicine, 170, Article 107920, https://doi.org/10.1016/j.compbiomed.2024.107920. Zhao, Z., Ren, X., Song, K., Qiang, Y., Zhao, J., Zhang, J., & Han, P. (2023). PreGenerator: TCM prescription recommendation model based on retrieval and generation method. IEEE Access, 11, 103679- 103692, https://doi.org/10.1109/ACCESS.2023.3316219. Zhu, J. (2022). Clinical analysis of the efficacy of traditional Chinese medicine in treating spontaneous and night sweats in patients with advanced lung cancer and Qi- Yin deficiency syndrome. Journal of Medical Forum, 43(10), 91- 94, (in Chinese).