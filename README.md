# Medical-Vision-and-Language-Tasks-and-Methodologies-A-Survey
Medical Vision-and-Language Tasks and Methodologies: A Survey

# Table of Contents
- [Medical Vision-Language Pre-training](#medical-vision-language-pre-training)
- [Medical Report Generation](#medical-report-generation)
- [Medical Visual Question Answering](#medical-visual-question-answering)
- [Medical Multi-modal Diagnosis and Prognosis](#medical-multi-modal-diagnosis-and-prognosis)
- [Medical Image Segmentation](#medical-image-segmentation)
- [Medical Image-Text Retrieval](#medical-image-text-retrieval)

[[paper]()] [[code]()]

## Medical Vision-Language Pre-training


- **ConVIRT:** "Contrastive Learning of Medical Visual Representations from Paired Images and Text" **PMLR (2022)** [[paper](https://proceedings.mlr.press/v182/zhang22a/zhang22a.pdf)] [[code](https://github.com/yuhaozhang/convirt)]
  
- **CheXzero:**  "Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning" **Nature (2022).** [[paper](https://www.nature.com/articles/s41551-022-00936-9)] [[code](https://github.com/rajpurkarlab/CheXzero)]
- **BiomedCLIP:** "a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs." **arXiv (2023)** [[paper](https://arxiv.org/pdf/2303.00915)] [[huggingface](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)]
- **PLIP:** "A visual–language foundation model for pathology image analysis using medical Twitter" **Nature (2023)** [[paper](https://www.nature.com/articles/s41591-023-02504-3)] [[code](https://github.com/PathologyFoundation/plip)]
- **BioViL:** "Making the Most of Text Semantics to Improve Biomedical Vision–Language Processing" **ECCV (2022)** [[paper](https://link.springer.com/chapter/10.1007/978-3-031-20059-5_1)] [[code](https://github.com/microsoft/hi-ml/tree/main/hi-ml-multimodal)]
- **MGCA:** "Multi-Granularity Cross-modal Alignment for Generalized Medical Visual Representation Learning" "Neurips (2022) [[paper](https://arxiv.org/pdf/2210.06044)] [[code](https://github.com/HKU-MedAI/MGCA)]
- **MedCLIP:** "Medclip: Contrastive learning from unpaired medical images and text. " **EMNLP (2022) [[paper](https://arxiv.org/pdf/2210.10163)] [[code](https://github.com/RyanWangZf/MedCLIP)]
- **M2I2:** "Self-supervised vision-language pretraining for Medical visual question answering" **ISBI (2023)** [[paper](https://arxiv.org/pdf/2211.13594)] [[code](https://github.com/pengfeiliHEU/M2I2)]
- **PMC-CLIP:** "PMC-CLIP: Contrastive Language-Image Pre-training Using Biomedical Documents" **MICCAI (2023)** [[paper]([https://link.springer.com/chapter/10.1007/978-3-031-43993-3_51](https://arxiv.org/pdf/2303.07240))] [[code](https://github.com/WeixiongLin/PMC-CLIP)]
- **Medical X-VL:** "Self-supervised Multi-modal Training from Uncurated Image and Reports Enables Zero-shot Oversight Artificial Intelligence in Radiology" **ArXiv (2023)** [[paper](https://arxiv.org/pdf/2208.05140)] [[code](https://github.com/sangjoon-park/Medical_X-VL)]
- **IMITATE:** "IMITATE: Clinical Prior Guided Hierarchical Vision-Language Pre-training" **ArXiv (2023)** [[paper](https://arxiv.org/pdf/2310.07355)]
- **SAT:** "Improving Medical Vision-Language Contrastive Pretraining with Semantics-aware Triage" **TMI (2023)** [[paper](https://pubmed.ncbi.nlm.nih.gov/37440389/)]
- **KoBo:** "Knowledge Boosting: Rethinking Medical Contrastive Vision-Language Pre-training" **MICCAI (2023)** [[paper](https://arxiv.org/pdf/2307.07246)] [[code](https://github.com/ChenXiaoFei-CS/KoBo)]
- **PTUnifier:** "Towards Unifying Medical Vision-and-Language Pre-training via Soft Prompts" **ICCV (2023)** [[paper](https://arxiv.org/pdf/2302.08958)] [[code](https://github.com/zhjohnchan/PTUnifier)]
- **GLoRIA:** "GLoRIA: A Multimodal Global-Local Representation Learning Framework for Label-efficient Medical Image Recognition" [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_GLoRIA_A_Multimodal_Global-Local_Representation_Learning_Framework_for_Label-Efficient_Medical_ICCV_2021_paper.pdf)] [[code](https://github.com/marshuang80/gloria)]
- **





  
## Medical Report Generation

- Yang, Shuxin and Wu, Xian and Ge, Shen and Zheng, Zhuozhao and Zhou, S Kevin and Xiao, Li.<br> "Radiology report generation with a learned knowledge base and multi-modal alignment" **Medical Image Analysis (2023).** [[paper](https://www.sciencedirect.com/science/article/pii/S1361841523000592)] [[code](https://github.com/LX-doctorAI1/M2KT)]

- Szeskin, Adi and Rochman, Shalom and Weiss, Snir and Lederman, Richard and Sosna, Jacob and Joskowicz, Leo.<br> "Liver lesion changes analysis in longitudinal CECT scans by simultaneous deep learning voxel classification with SimU-Net" **Medical Image Analysis (2023).** [[paper](https://www.sciencedirect.com/science/article/pii/S1361841522003036)]

- Zhu, Qingqing and Mathai, Tejas Sudharshan and Mukherjee, Pritam and Peng, Yifan and Summers, Ronald M and Lu, Zhiyong.<br> "Utilizing longitudinal chest x-rays and reports to pre-fill radiology reports" **MICCAI (2023).** [[paper](https://link.springer.com/chapter/10.1007/978-3-031-43904-9_19)] [[code](https://github.com/CelestialShine/Longitudinal-Chest-X-Ray)]

- Dalla Serra, Francesco and Wang, Chaoyang and Deligianni, Fani and Dalton, Jeffrey and O’Neil, Alison Q.<br> "Finding-aware anatomical tokens for chest X-ray automated reporting" **MICCAI (2023).** [[paper](https://link.springer.com/chapter/10.1007/978-3-031-45673-2_41)]

- **KiUT:** Huang, Zhongzhen and Zhang, Xiaofan and Zhang, Shaoting.<br> "Kiut: Knowledge-injected u-transformer for radiology report generation" **CVPR (2023).** [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_KiUT_Knowledge-Injected_U-Transformer_for_Radiology_Report_Generation_CVPR_2023_paper.pdf)]

- **DCL:** Li, Mingjie and Lin, Bingqian and Chen, Zicong and Lin, Haokun and Liang, Xiaodan and Chang, Xiaojun.<br> "Dynamic graph enhanced contrastive learning for chest x-ray report generation" **CVPR (2023).** [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_Dynamic_Graph_Enhanced_Contrastive_Learning_for_Chest_X-Ray_Report_Generation_CVPR_2023_paper.pdf)] [[code](https://github.com/mlii0117/DCL)]

- **RGRG:** Tanida, Tim and Müller, Philip and Kaissis, Georgios and Rueckert, Daniel.<br> "Interactive and explainable region-guided radiology report generation" **CVPR (2023).** [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Tanida_Interactive_and_Explainable_Region-Guided_Radiology_Report_Generation_CVPR_2023_paper.pdf)] [[code](https://github.com/ttanida/rgrg)]

- **METransformer:** Wang, Zhanyu and Liu, Lingqiao and Wang, Lei and Zhou, Luping.<br> "Metransformer: Radiology report generation by transformer with multiple learnable expert tokens" **CVPR (2023).** [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_METransformer_Radiology_Report_Generation_by_Transformer_With_Multiple_Learnable_Expert_CVPR_2023_paper.pdf)]

- **ICT:** Zhang, Junsan and Shen, Xiuxuan and Wan, Shaohua and Goudos, Sotirios K and Wu, Jie and Cheng, Ming and Zhang, Weishan.<br> "A novel deep learning model for medical report generation by inter-intra information calibration" **JBHI (2023).** [[paper](https://ieeexplore.ieee.org/abstract/document/10016250)]

- Zheng, Ervine and Yu, Qi.<br> "Evidential interactive learning for medical image captioning" **ICML (2023).** [[paper](https://proceedings.mlr.press/v202/zheng23g/zheng23g.pdf)]

- **PRIOR:** Cheng, Pujin and Lin, Li and Lyu, Junyan and Huang, Yijin and Luo, Wenhan and Tang, Xiaoying.<br> "Prior: Prototype representation joint learning from medical images and reports" **ICCV (2023).** [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Cheng_PRIOR_Prototype_Representation_Joint_Learning_from_Medical_Images_and_Reports_ICCV_2023_paper.pdf)] [[code](https://github.com/QtacierP/PRIOR)]

- **MRM:** Zhou, Hong-Yu and Lian, Chenyu and Wang, Liansheng and Yu, Yizhou.<br> "Advancing radiograph representation learning with masked record modeling" **ICLR (2023).** [[paper](https://arxiv.org/pdf/2301.13155)] [[code](https://github.com/RL4M/MRM-pytorch)]

- **MMTN:** Cao, Yiming and Cui, Lizhen and Zhang, Lei and Yu, Fuqiang and Li, Zhen and Xu, Yonghui.<br> "MMTN: multi-modal memory transformer network for image-report consistent medical report generation" **AAAI (2023).** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25100)]

- **ATAG:** Yan, Sixing and Cheung, William K and Chiu, Keith and Tong, Terence M and Cheung, Ka Chun and See, Simon.<br> "Attributed abnormality graph embedding for clinically accurate x-ray report generation" **TMI (2023).** [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10045710)]

- Yang, Shuxin and Wu, Xian and Ge, Shen and Zhou, S Kevin and Xiao, Li.<br> "Knowledge matters: Chest radiology report generation with general and specific knowledge" **Medical Image Analysis (2022).** [[paper](https://www.sciencedirect.com/science/article/pii/S1361841522001578)] [[code](https://github.com/LX-doctorAI1/GSKET)]

- **VTI:** Najdenkoska, Ivona and Zhen, Xiantong and Worring, Marcel and Shao, Ling.<br> "Uncertainty-aware report generation for chest X-rays by variational topic inference" **Medical Image Analysis (2022).** [[paper](https://www.sciencedirect.com/science/article/pii/S1361841522002341)] [[code](https://github.com/ivonajdenkoska/variational-xray-report-gen)]

- **TranSQ:** Kong, Ming and Huang, Zhengxing and Kuang, Kun and Zhu, Qiang and Wu, Fei.<br> "Transq: Transformer-based semantic query for medical report generation" **MICCAI (2022).** [[paper](https://link.springer.com/chapter/10.1007/978-3-031-16452-1_58)] [[code](https://github.com/zjukongming/TranSQ)]

- Sun, Jinghan and Wei, Dong and Wang, Liansheng and Zheng, Yefeng.<br> "Lesion guided explainable few weak-shot medical report generation" **MICCAI (2022).** [[paper](https://arxiv.org/pdf/2211.08732)] [[code](https://github.com/jinghanSunn/Few-weak-shot-RG)]

- **MCGN:** Wang, Zhanyu and Tang, Mingkang and Wang, Lei and Li, Xiu and Zhou, Luping.<br> "A medical semantic-assisted transformer for radiographic report generation" **MICCAI (2022).** [[paper](https://arxiv.org/pdf/2208.10358)]

- **SGF:** Li, Jun and Li, Shibo and Hu, Ying and Tao, Huiren.<br> "A self-guided framework for radiology report generation" **MICCAI (2022).** [[paper](https://arxiv.org/pdf/2206.09378)]

- **SGT:** Lin, Chen and Zheng, Shuai and Liu, Zhizhe and Li, Youru and Zhu, Zhenfeng and Zhao, Yao.<br> "Sgt: Scene graph-guided transformer for surgical report generation" **MICCAI (2022).** [[paper](https://liyouru0228.github.io/HomePage/data/SGT.pdf)] [[code](https://github.com/ccccchenllll/SGT_master)]

- **ITA:** Wang, Lin and Ning, Munan and Lu, Donghuan and Wei, Dong and Zheng, Yefeng and Chen, Jie.<br> "An inclusive task-aware framework for radiology report generation" **MICCAI (2022).** [[paper](https://link.springer.com/chapter/10.1007/978-3-031-16452-1_54)] [[code](https://github.com/Reremee/ITA)]

- **RepsNet:** Tanwani, Ajay K and Barral, Joelle and Freedman, Daniel.<br> "Repsnet: Combining vision with language for automated medical reports" **MICCAI (2022).** [[paper](https://arxiv.org/pdf/2209.13171)]

- **CoPlan:** Nishino, Toru and Miura, Yasuhide and Taniguchi, Tomoki and Ohkuma, Tomoko and Suzuki, Yuki and Kido, Shoji and Tomiyama, Noriyuki.<br> "Factual accuracy is not enough: Planning consistent description order for radiology report generation" **EMNLP (2022).** [[paper](https://aclanthology.org/2022.emnlp-main.480.pdf)]

- Delbrouck, Jean-Benoit and Chambon, Pierre and Bluethgen, Christian and Tsai, Emily and Almusa, Omar and Langlotz, Curtis P.<br> "Improving the factual correctness of radiology report generation with semantic rewards" **EMNLP (2022).** [[paper](https://arxiv.org/pdf/2210.12186)] [[code](https://github.com/jbdel/vilmedic)]

- **CGT:** Li, Mingjie and Cai, Wenjia and Verspoor, Karin and Pan, Shirui and Liang, Xiaodan and Chang, Xiaojun.<br> "Cross-modal clinical graph transformer for ophthalmic report generation" **CVPR (2022).** [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Cross-Modal_Clinical_Graph_Transformer_for_Ophthalmic_Report_Generation_CVPR_2022_paper.pdf)]

- **TransFuser:** Huang, Jia-Hong and Wu, Ting-Wei and Yang, C-H Huck and Shi, Zenglin and Lin, I and Tegner, Jesper and Worring, Marcel and others.<br> "Non-local attention improves description generation for retinal images" **WACV (2022).** [[paper](https://openaccess.thecvf.com/content/WACV2022/papers/Huang_Non-Local_Attention_Improves_Description_Generation_for_Retinal_Images_WACV_2022_paper.pdf)]

- **XPRONET:** Wang, Jun and Bhalerao, Abhir and He, Yulan.<br> "Cross-modal prototype driven network for radiology report generation" **ECCV (2022).** [[paper](https://arxiv.org/pdf/2207.04818)] [[code](https://github.com/Markin-Wang/XProNet)]

- **DCNet(EDC-Net):** Singh, Dilbag and Kaur, Manjit and Alanazi, Jazem Mutared and AlZubi, Ahmad Ali and Lee, Heung-No.<br> "Efficient evolving deep ensemble medical image captioning network" **JBHI (2022).** [[paper](https://heungno.net/wp-content/uploads/2023/07/3.-Efficient-Evolving-deep-1.pdf)] [[code]()]

- Yan, Bin and Pei, Mingtao and Zhao, Meng and Shan, Caifeng and Tian, Zhaoxing.<br> "Prior guided transformer for accurate radiology reports generation" **JBHI (2022).** [[paper](https://ieeexplore.ieee.org/abstract/document/9852309)]

- **NSL:** Han, Zhongyi and Wei, Benzheng and Xi, Xiaoming and Chen, Bo and Yin, Yilong and Li, Shuo.<br> "Unifying neural learning and symbolic reasoning for spinal medical report generation" **Medical Image Analysis (2021).** [[paper](https://arxiv.org/pdf/2004.13577)]

- **AlignTransformer:** You, Di and Liu, Fenglin and Ge, Shen and Xie, Xiaoxia and Zhang, Jing and Wu, Xian.<br> "Aligntransformer: Hierarchical alignment of visual regions and disease tags for medical report generation" **MICCAI (2021).** [[paper](https://arxiv.org/pdf/2203.10095)]

- **VTI:** Najdenkoska, Ivona and Zhen, Xiantong and Worring, Marcel and Shao, Ling.<br> "Variational topic inference for chest x-ray report generation" **MICCAI (2021).** [[paper](https://arxiv.org/pdf/2107.07314)]

- **CNN-TRG:** Pino, Pablo and Parra, Denis and Besa, Cecilia and Lagos, Claudio.<br> "Clinically correct report generation from chest x-rays using templates" **MICCAI (2021).** [[paper](https://dparra.sitios.ing.uc.cl/pdfs/preprint_Pinoetal_MICCAI_2021.pdf)]

- **RATCHET:** Hou, Benjamin and Kaissis, Georgios and Summers, Ronald M and Kainz, Bernhard.<br> "Ratchet: Medical transformer for chest x-ray diagnosis and reporting" **MICCAI (2021).** [[paper](https://arxiv.org/pdf/2107.02104)] [[code](https://github.com/farrell236/RATCHET)]

- **CIDA:** Xu, Mengya and Islam, Mobarakol and Lim, Chwee Ming and Ren, Hongliang.<br> "Class-incremental domain adaptation with smoothing and calibration for surgical report generation" **MICCAI (2021).** [[paper](https://arxiv.org/pdf/2107.11091)] [[code](https://github.com/XuMengyaAmy/CIDACaptioning)]

- **:** Nguyen, Hoang TN and Nie, Dong and Badamdorj, Taivanbat and Liu, Yujie and Zhu, Yingying and Truong, Jason and Cheng, Li.<br> "Automated generation of accurate & fluent medical x-ray reports" **EMNLP (2021).** [[paper](https://arxiv.org/pdf/2108.12126)] [[code](https://github.com/ginobilinie/xray_report_generation)]

- **${M^2}$TR. PROGRESSIVE:** Nooralahzadeh, Farhad and Gonzalez, Nicolas Perez and Frauenfelder, Thomas and Fujimoto, Koji and Krauthammer, Michael.<br> "Progressive transformer-based generation of radiology reports" **EMNLP (2021).** [[paper](https://arxiv.org/pdf/2102.09777)] [[code](https://github.com/uzh-dqbm-cmi/ARGON)]

- **CMCL:** Liu, Fenglin and Ge, Shen and Zou, Yuexian and Wu, Xian.<br> "Competence-based multimodal curriculum learning for medical report generation" **ACL (2021).** [[paper](https://arxiv.org/pdf/2206.14579)]]

- **MedWriter:** Yang, Xingyi and Ye, Muchao and You, Quanzeng and Ma, Fenglong.<br> "Writing by memorizing: Hierarchical retrieval-based medical report generation" **ACL (2021).** [[paper](https://arxiv.org/pdf/2106.06471)]

- **CA:** Liu, Fenglin and Yin, Changchang and Wu, Xian and Ge, Shen and Zou, Yuexian and Zhang, Ping and Sun, Xu.<br> "Contrastive attention for automatic chest x-ray report generation" **ACL (2021).** [[paper](https://arxiv.org/pdf/2106.06965)]

- **CMN:** Chen, Zhihong and Shen, Yaling and Song, Yan and Wan, Xiang.<br> "Cross-modal memory networks for radiology report generation" **ACL (2021).** [[paper](https://arxiv.org/pdf/2204.13258)] [[code](https://github.com/zhjohnchan/R2GenCMN)]

- **KGAE:** Liu, Fenglin and You, Chenyu and Wu, Xian and Ge, Shen and Sun, Xu and others.<br> "Auto-encoding knowledge graph for unsupervised medical report generation" **NIPS (2021).** [[paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/876e1c59023b1a0e95808168e1a8ff89-Paper.pdf)]

- **CXR-RePaiR:** Endo, Mark and Krishnan, Rayan and Krishna, Viswesh and Ng, Andrew Y and Rajpurkar, Pranav.<br> "Retrieval-based chest x-ray report generation using a pre-trained contrastive language-image model" **NIPS (2021).** [[paper](https://proceedings.mlr.press/v158/endo21a/endo21a.pdf)]

- **MEDSKIP:** Pahwa, Esha and Mehta, Dwij and Kapadia, Sanjeet and Jain, Devansh and Luthra, Achleshwar.<br> "Medskip: Medical report generation using skip connections and integrated attention" **ICCV (2021).** [[paper](https://openaccess.thecvf.com/content/ICCV2021W/CVAMD/papers/Pahwa_MedSkip_Medical_Report_Generation_Using_Skip_Connections_and_Integrated_Attention_ICCVW_2021_paper.pdf)]

- Zhou, Yi and Huang, Lei and Zhou, Tao and Fu, Huazhu and Shao, Ling.<br> "Visual-textual attentive semantic consistency for medical report generation" **ICCV (2021).** [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_Visual-Textual_Attentive_Semantic_Consistency_for_Medical_Report_Generation_ICCV_2021_paper.pdf)]

- **PPKED:** Liu, Fenglin and Wu, Xian and Ge, Shen and Fan, Wei and Zou, Yuexian.<br> "Exploring and distilling posterior and prior knowledge for radiology report generation" **CVPR (2021).** [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Exploring_and_Distilling_Posterior_and_Prior_Knowledge_for_Radiology_Report_CVPR_2021_paper.pdf)]

- Wang, Zhanyu and Zhou, Luping and Wang, Lei and Li, Xiu.<br> "A self-boosting framework for automated radiographic report generation" **CVPR (2021).** [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_A_Self-Boosting_Framework_for_Automated_Radiographic_Report_Generation_CVPR_2021_paper.pdf)]

- Huang, Jia-Hong and Yang, C-H Huck and Liu, Fangyu and Tian, Meng and Liu, Yi-Chieh and Wu, Ting-Wei and Lin, I and Wang, Kang and Morikawa, Hiromasa and Chang, Hernghua and others.<br> "Deepopht: medical report generation for retinal images via deep models and visual explanation" **WACV (2021).** [[paper](https://openaccess.thecvf.com/content/WACV2021/papers/Huang_DeepOpht_Medical_Report_Generation_for_Retinal_Images_via_Deep_Models_WACV_2021_paper.pdf)]

- **TriNet:** Yang, Yan and Yu, Jun and Zhang, Jian and Han, Weidong and Jiang, Hanliang and Huang, Qingming.<br> "Joint embedding of deep visual and semantic features for medical image report generation" **TMM (2021).** [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9606584)] [[code](https://github.com/yangyan22/Medical-Report-Generation-TriNet)]

- **TS-MRGen:** Nishino, Toru and Ozaki, Ryota and Momoki, Yohei and Taniguchi, Tomoki and Kano, Ryuji and Nakano, Norihisa and Tagawa, Yuki and Taniguchi, Motoki and Ohkuma, Tomoko and Nakamura, Keigo.<br> "Reinforcement learning with imbalanced dataset for data-to-text medical report generation" **EMNLP (2020).** [[paper](https://aclanthology.org/2020.findings-emnlp.202.pdf)] [[code]()]

- **R2Gen:** Chen, Zhihong and Song, Yan and Chang, Tsung-Hui and Wan, Xiang.<br> "Generating radiology reports via memory-driven transformer" **EMNLP (2020).** [[paper](https://arxiv.org/pdf/2010.16056)] [[code](https://github.com/zhjohnchan/R2Gen)]

- Lovelace, Justin and Mortazavi, Bobak.<br> "Learning to generate clinically coherent chest X-ray reports" **EMNLP (2020).** [[paper](https://aclanthology.org/2020.findings-emnlp.110.pdf)]

- **CVSE:** Ni, Jianmo and Hsu, Chun-Nan and Gentili, Amilcare and McAuley, Julian.<br> "Learning visual-semantic embeddings for reporting abnormal findings on chest X-rays" **EMNLP (2020).** [[paper](https://arxiv.org/pdf/2010.02467)]

- Gasimova, Aydan and Seegoolam, Gavin and Chen, Liang and Bentley, Paul and Rueckert, Daniel.<br> "Spatial semantic-preserving latent space learning for accelerated dwi diagnostic report generation" **MICCAI (2020).** [[paper](https://link.springer.com/chapter/10.1007/978-3-030-59728-3_33)]

- Syeda-Mahmood, Tanveer and Wong, Ken CL and Gur, Yaniv and Wu, Joy T and Jadhav, Ashutosh and Kashyap, Satyananda and Karargyris, Alexandros and Pillai, Anup and Sharma, Arjun and Syed, Ali Bin and others.<br> "Chest x-ray report generation through fine-grained label learning" **MICCAI (2020).** [[paper](https://arxiv.org/pdf/2007.13831)]

- Zhang, Yixiao and Wang, Xiaosong and Xu, Ziyue and Yu, Qihang and Yuille, Alan and Xu, Daguang.<br> "When radiology report generation meets knowledge graph" **AAAI (2020).** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/6989)]

- **KERP:** Li, Christy Y and Liang, Xiaodan and Hu, Zhiting and Xing, Eric P.<br> "Knowledge-driven encode, retrieve, paraphrase for medical image report generation" **MICCAI (2019).** [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/4637)]

- **MvH:** Yuan, Jianbo and Liao, Haofu and Luo, Rui and Luo, Jiebo.<br> "Automatic radiology report generation based on multi-view image fusion and medical concept enrichment" **MICCAI (2019).** [[paper](https://arxiv.org/pdf/1907.09085)]

- **CMAS:** Jing, Baoyu and Wang, Zeya and Xing, Eric.<br> "Show, describe and conclude: On exploiting the structure information of chest x-ray reports" **ACL (2019).** [[paper](https://arxiv.org/pdf/2004.12274)]

- Xue, Yuan and Huang, Xiaolei.<br> "Improved disease classification in chest x-rays with transferred features from report generation" **IPMI (2019).** [[paper](https://faculty.ist.psu.edu/suh972/Xue-IPMI2019.pdf)]

- Daniels, Zachary A and Metaxas, Dimitris N.<br> "Exploiting visual and report-based information for chest x-ray analysis by jointly learning visual classifiers and topic models" **ISBI (2019).** [[paper](https://par.nsf.gov/servlets/purl/10105312)]

- **RGAN:** Han, Zhongyi and Wei, Benzheng and Leung, Stephanie and Chung, Jonathan and Li, Shuo.<br> "Towards automatic report generation in spine radiology using weakly supervised framework" **MICCAI (2018).** [[paper](http://digitalimaginggroup.ca/members/Shuo/miccai2018-hanzhongyi.pdf)]

- Xue, Yuan and Xu, Tao and Rodney Long, L and Xue, Zhiyun and Antani, Sameer and Thoma, George R and Huang, Xiaolei.<br> "Multimodal recurrent model with attention for automated radiology report generation" **MICCAI (2018).** [[paper](https://faculty.ist.psu.edu/suh972/Xue-MICCAI2018.pdf)]

- Jing, Baoyu and Xie, Pengtao and Xing, Eric.<br> "On the automatic generation of medical imaging reports" **ACL (2018).** [[paper](https://arxiv.org/pdf/1711.08195)]

- **HRGR-Agent:** Li, Yuan and Liang, Xiaodan and Hu, Zhiting and Xing, Eric P.<br> "Hybrid retrieval-generation reinforced agent for medical image report generation" **NIPS (2018).** [[paper](https://proceedings.neurips.cc/paper/2018/file/e07413354875be01a996dc560274708e-Paper.pdf)]

- **TieNet:** Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Summers, Ronald M.<br> "Tienet: Text-image embedding network for common thorax disease classification and reporting in chest x-rays" **CVPR (2018).** [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_TieNet_Text-Image_Embedding_CVPR_2018_paper.pdf)]

 Medical Visual Question Answering

- **MUMC:** Li, Pengfei and Liu, Gang and He, Jinlong and Zhao, Zixu and Zhong, Shenjun.<br> "Masked vision and language pre-training with unimodal and multimodal contrastive losses for medical visual question answering" **MICCAI (2023).** [[paper](https://arxiv.org/pdf/2307.05314)] [[code](https://github.com/pengfeiliHEU/MUMC)]

- Van Sonsbeek, Tom and Derakhshani, Mohammad Mahdi and Najdenkoska, Ivona and Snoek, Cees GM and Worring, Marcel.<br> "Open-ended medical visual question answering through prefix tuning of language models" **MICCAI (2023).** [[paper](https://arxiv.org/pdf/2303.05977)]

- Tascon-Morales, Sergio and Márquez-Neila, Pablo and Sznitman, Raphael.<br> "Localized questions in medical visual question answering" **MICCAI (2023).** [[paper](https://arxiv.org/pdf/2307.01067)]

- **CS-VQLA:** Bai, Long and Islam, Mobarakol and Ren, Hongliang.<br> "Revisiting distillation for continual learning on visual question localized-answering in robotic surgery" **MICCAI (2023).** [[paper](https://arxiv.org/pdf/2307.12045)] [[code](https://github.com/longbai1006/CS-VQLA)]

- **CAT-ViL:** Bai, Long and Islam, Mobarakol and Ren, Hongliang.<br> "CAT-ViL: co-attention gated vision-language embedding for visual question localized-answering in robotic surgery" **MICCAI (2023).** [[paper](https://www.researchgate.net/profile/Long-Bai-13/publication/372286558_Co-Attention_Gated_Vision-Language_Embedding_for_Visual_Question_Localized-Answering_in_Robotic_Surgery/links/64d61268b684851d3d9e9320/Co-Attention-Gated-Vision-Language-Embedding-for-Visual-Question-Localized-Answering-in-Robotic-Surgery.pdf)] [[code](https://github.com/longbai1006/CAT-ViL)]

- **hi-VQA:** Pellegrini, Chantal and Keicher, Matthias and Özsoy, Ege and Navab, Nassir.<br> "Rad-restruct: A novel vqa benchmark and method for structured radiology reporting" **MICCAI (2023).** [[paper](https://arxiv.org/pdf/2307.05766)] [[code](https://github.com/ChantalMP/Rad-ReStruct)]

- **DeBCF:** Zhan, Chenlu and Peng, Peng and Zhang, Hanrong and Sun, Haiyue and Shang, Chunnan and Chen, Tao and Wang, Hongsen and Wang, Gaoang and Wang, Hongwei.<br> "Debiasing Medical Visual Question Answering via Counterfactual Training" **MICCAI (2023).** [[paper](https://link.springer.com/chapter/10.1007/978-3-031-43895-0_36)]

- **${MF^2-MVQA}$:** Song, Shanshan and Li, Jiangyun and Wang, Jing and Cai, Yuanxiu and Dong, Wenkai.<br> "\$MF^2-MVQA$: A Multi-Stage Feature Fusion Method for Medical Visual Question Answering" **ISBI (2023).** [[paper](https://arxiv.org/pdf/2211.05991)] [[code]()]

- **M2I2:** Li, Pengfei and Liu, Gang and Tan, Lin and Liao, Jinying and Zhong, Shenjun.<br> "Self-supervised vision-language pretraining for medial visual question answering" **ISBI (2023).** [[paper](https://arxiv.org/pdf/2211.13594)] [[code](https://github.com/pengfeiliHEU/M2I2)]

- **Q2ATransformer:** Liu, Yunyi and Wang, Zhanyu and Xu, Dong and Zhou, Luping.<br> "Q2atransformer: Improving medical vqa via an answer querying decoder" **IPMI (2023).** [[paper](https://arxiv.org/pdf/2304.01611)]

- Tascon-Morales, Sergio and Márquez-Neila, Pablo and Sznitman, Raphael.<br> "Consistency-preserving visual question answering in medical imaging" **MICCAI (2022).** [[paper](https://arxiv.org/pdf/2206.13296)] [[code](https://github.com/sergiotasconmorales/consistency_vqa)]

- **RepsNet:** Tanwani, Ajay K and Barral, Joelle and Freedman, Daniel.<br> "Repsnet: Combining vision with language for automated medical reports" **MICCAI (2022).** [[paper](https://arxiv.org/pdf/2209.13171)]

- Cong, Fuze and Xu, Shibiao and Guo, Li and Tian, Yinbing.<br> "Anomaly matters: An anomaly-oriented model for medical visual question answering" **TMI (2022).** [[paper](https://www.researchgate.net/profile/Fuze-Cong/publication/361648686_Anomaly_Matters_An_Anomaly-Oriented_Model_for_Medical_Visual_Question_Answering/links/62c29197bd55e01e75f94d9e/Anomaly-Matters-An-Anomaly-Oriented-Model-for-Medical-Visual-Question-Answering.pdf)]

- **VQAMix:** Gong, Haifan and Chen, Guanqi and Mao, Mingzhi and Li, Zhen and Li, Guanbin.<br> "Vqamix: Conditional triplet mixup for medical visual question answering" **TMI (2022).** [[paper](https://drive.google.com/file/d/157DtLucUdiACgWiWJTJjkQ9Us1T1X-VU/view)] [[code](https://github.com/haifangong/VQAMix)]

- Liu, Bo and Zhan, Li-Ming and Xu, Li and Wu, Xiao-Ming.<br> "Medical visual question answering via conditional reasoning and contrastive learning" **TMI (2022).** [[paper](https://ieeexplore.ieee.org/abstract/document/9999450)] [[code](https://github.com/Awenbocc/CPCR)]

- **TraP-VQA:** Naseem, Usman and Khushi, Matloob and Kim, Jinman.<br> "Vision-language transformer for interpretable pathology visual question answering" **JBHI (2022).** [[paper](https://ieeexplore.ieee.org/abstract/document/9745795)]

- **MMQ:** Do, Tuong and Nguyen, Binh X and Tjiputra, Erman and Tran, Minh and Tran, Quang D and Nguyen, Anh.<br> "Multiple meta-model quantifying for medical visual question answering" **MICCAI (2021).** [[paper](https://arxiv.org/pdf/2105.08913)] [[code](https://github.com/aioz-ai/MICCAI21_MMQ)]

- **CPRD:** Liu, Bo and Zhan, Li-Ming and Wu, Xiao-Ming.<br> "Contrastive pre-training and representation distillation for medical visual question answering based on radiology images" **MICCAI (2021).** [[paper](https://www4.comp.polyu.edu.hk/~csxmwu/papers/MICCAI-2021-Med_VQA.pdf)] [[code](https://github.com/awenbocc/cprd)]

- **MMBERT:** Khare, Yash and Bagal, Viraj and Mathew, Minesh and Devi, Adithi and Priyakumar, U Deva and Jawahar, CV.<br> "Mmbert: Multimodal bert pretraining for improved medical vqa" **ISBI (2021).** [[paper](https://arxiv.org/pdf/2104.01394)] [[code](https://github.com/VirajBagal/MMBERT)]

- **QC-MLB:** Vu, Minh H and Löfstedt, Tommy and Nyholm, Tufve and Sznitman, Raphael.<br> "A question-centric model for visual question answering in medical imaging" **TMI (2020).** [[paper](https://arxiv.org/pdf/2003.08760)]

- **MEVF:** Nguyen, Binh D and Do, Thanh-Toan and Nguyen, Binh X and Do, Tuong and Tjiputra, Erman and Tran, Quang D.<br> "Overcoming data limitation in medical visual question answering" **MICCAI (2019).** [[paper](https://arxiv.org/pdf/1909.11867)] [[code](https://github.com/aioz-ai/MICCAI19-MedVQA)]

 Medical Multi-modal Diagnosis and Prognosis

- **Xplainer:** Pellegrini, Chantal and Keicher, Matthias and {\"O}zsoy, Ege and Jiraskova, Petra and Braren, Rickmer and Navab, Nassir.<br> "Xplainer: From x-ray observations to explainable zero-shot diagnosis" **MICCAI (2023).** [[paper](https://arxiv.org/pdf/2303.13391)] [[code](https://github.com/ChantalMP/Xplainer)]

- Zhong, Yi and Xu, Mengqiu and Liang, Kongming and Chen, Kaixin and Wu, Ming.<br> "Ariadne's Thread: Using Text Prompts to Improve Segmentation of Infected Areas from Chest X-ray Images" **MICCAI (2023).** [[paper](https://arxiv.org/pdf/2307.03942)] [[code](https://github.com/Junelin2333/LanGuideMedSeg-MICCAI2023)]

- **CLIP-Lung:** Lei, Yiming and Li, Zilong and Shen, Yan and Zhang, Junping and Shan, Hongming.<br> "CLIP-Lung: Textual knowledge-guided lung nodule malignancy prediction" **MICCAI (2023).** [[paper](https://arxiv.org/pdf/2304.08013)] 

- **GSDG:** Chen, Shouyu and Guo, Xin and Zhu, Jianping and Wang, Yin.<br> "GSDG: Exploring a Global Semantic-Guided Dual-Stream Graph Model for Automated Volume Differential Diagnosis and Prognosis" **MICCAI (2023).** [[paper](https://link.springer.com/chapter/10.1007/978-3-031-43904-9_45)]

- Ichinose, Akimichi and Hatsutani, Taro and Nakamura, Keigo and Kitamura, Yoshiro and Iizuka, Satoshi and Simo-Serra, Edgar and Kido, Shoji and Tomiyama, Noriyuki.<br> "Visual grounding of whole radiology reports for 3d ct images" **MICCAI (2023).** [[paper](https://arxiv.org/pdf/2312.04794)]

- Liu, Jiaxiang and Hu, Tianxiang and Zhang, Yan and Gai, Xiaotang and Feng, Yang and Liu, Zuozhu.<br> "A chatgpt aided explainable framework for zero-shot medical image diagnosis" **arXiv (2023).** [[paper](https://arxiv.org/pdf/2307.01981)]

- **WSI-MTMI:** Liu, Jianxin and Ge, Rongjun and Wan, Peng and Zhu, Qi and Zhang, Daoqiang and Shao, Wei.<br> "Multi-task multi-instance learning for jointly diagnosis and prognosis of early-stage breast invasive carcinoma from whole-slide pathological images" **IPMI (2023).** [[paper](https://link.springer.com/chapter/10.1007/978-3-031-34048-2_12)]

- Song, Xuegang and Zhou, Feng and Frangi, Alejandro F and Cao, Jiuwen and Xiao, Xiaohua and Lei, Yi and Wang, Tianfu and Lei, Baiying.<br> "Multicenter and multichannel pooling GCN for early AD diagnosis based on dual-modality fused brain network" **TMI (2022).** [[paper](https://ieeexplore.ieee.org/abstract/document/9810283)] [[code](https://github.com/Xuegang-S)]

- Mehta, Sachin and Lu, Ximing and Wu, Wenjun and Weaver, Donald and Hajishirzi, Hannaneh and Elmore, Joann G and Shapiro, Linda G.<br> "End-to-end diagnosis of breast biopsy images with transformers" **Medical image analysis (2022).** [[paper](https://www.sciencedirect.com/science/article/pii/S136184152200113X)] 

- **${M^2F}$:** Lu, Zilin and Lu, Mengkang and Xia, Yong.<br> "M2F: A Multi-modal and Multi-task Fusion Network for Glioma Diagnosis and Prognosis" **MICCAI (2022).** [[paper](https://link.springer.com/chapter/10.1007/978-3-031-18814-5_1)]

- **BERTHop:** Monajatipoor, Masoud and Rouhsedaghat, Mozhdeh and Li, Liunian Harold and Jay Kuo, C-C and Chien, Aichi and Chang, Kai-Wei.<br> "Berthop: An effective vision-and-language model for chest x-ray disease diagnosis" **MICCAI (2022).** [[paper](https://link.springer.com/chapter/10.1007/978-3-031-16443-9_69)] [[code](https://github.com/masoud-monajati/BERTHop)]

- Kim, Daekyung and Nam, Chang-Mo and Park, Haesol and Jang, Mijung and Lee, Kyong Joon.<br> "Weakly supervised branch network with template mask for classifying masses in 3D automated breast ultrasound" **WACV (2022).** [[paper](https://openaccess.thecvf.com/content/WACV2022/papers/Kim_Weakly_Supervised_Branch_Network_With_Template_Mask_for_Classifying_Masses_WACV_2022_paper.pdf)] 

- Wu, Yujiao and Wang, Yaxiong and Huang, Xiaoshui and Yang, Fan and Ling, Sai Ho and Su, Steven Weidong.<br> "Multimodal Learning for Non-small Cell Lung Cancer Prognosis" **arXiv (2022).** [[paper](https://arxiv.org/pdf/2211.03280)]

- Tan, Kaiwen and Huang, Weixian and Liu, Xiaofeng and Hu, Jinlong and Dong, Shoubin.<br> "A multi-modal fusion framework based on multi-task correlation learning for cancer prognosis prediction" **Artificial Intelligence in Medicine (2022).** [[paper](https://www.sciencedirect.com/science/article/pii/S0933365722000252)]

- Chen, Yifei and Li, Dandan and Zhang, Xin and Jin, Jing and Shen, Yi.<br> "Computer aided diagnosis of thyroid nodules based on the devised small-datasets multi-view ensemble learning" **Medical Image Analysis (2021).** [[paper](https://www.sciencedirect.com/science/article/pii/S1361841520301833)]

- Gündel, Sebastian and Setio, Arnaud AA and Ghesu, Florin C and Grbic, Sasa and Georgescu, Bogdan and Maier, Andreas and Comaniciu, Dorin.<br> "Robust classification from noisy labels: Integrating additional knowledge for chest radiography abnormality assessment" **Medical Image Analysis (2021).** [[paper](https://arxiv.org/pdf/2104.05261)]

- Qiu, Di and Lui, Lok Ming.<br> "Modal Uncertainty Estimation for Medical Imaging Based Diagnosis" **MICCAI (2021).** [[paper](https://link.springer.com/chapter/10.1007/978-3-030-87735-4_1)]

- Bhalodia, Riddhish and Hatamizadeh, Ali and Tam, Leo and Xu, Ziyue and Wang, Xiaosong and Turkbey, Evrim and Xu, Daguang.<br> "Improving pneumonia localization via cross-attention on medical images and reports" **MICCAI (2021).** [[paper](https://arxiv.org/pdf/2110.03094)]

- Sekuboyina, Anjany and Oñoro-Rubio, Daniel and Kleesiek, Jens and Malone, Brandon.<br> "A relational-learning perspective to multi-label chest X-ray classification" **ISBI (2021).** [[paper](https://arxiv.org/pdf/2103.06220)]

- Wu, Joy and Gur, Yaniv and Karargyris, Alexandros and Syed, Ali Bin and Boyko, Orest and Moradi, Mehdi and Syeda-Mahmood, Tanveer.<br> "Automatic bounding box annotation of chest x-ray data for localization of abnormalities" **ISBI (2020).** [[paper](https://ieeexplore.ieee.org/abstract/document/9098482)]

- Chauhan, Geeticka and Liao, Ruizhi and Wells, William and Andreas, Jacob and Wang, Xin and Berkowitz, Seth and Horng, Steven and Szolovits, Peter and Golland, Polina.<br> "Joint modeling of chest radiographs and radiology reports for pulmonary edema assessment" **MICCAI (2020).** [[paper](https://link.springer.com/chapter/10.1007/978-3-030-59713-9_51)] [[code](https://github.com/RayRuizhiLiao/joint_chestxray)]

- van Sonsbeek, Tom and Worring, Marcel.<br> "Towards automated diagnosis with attentive multi-modal learning using electronic health records and chest x-rays" **MICCAI (2020).** [[paper](https://link.springer.com/chapter/10.1007/978-3-030-60946-7_11)]

- Tian, Jiang and Zhong, Cheng and Shi, Zhongchao and Xu, Feiyu.<br> "Towards automatic diagnosis from multi-modal medical data" **MICCAI (2019).** [[paper](https://link.springer.com/chapter/10.1007/978-3-030-33850-3_8)]

- **DGM2FS:** Shao, Wei and Wang, Tongxin and Huang, Zhi and Cheng, Jun and Han, Zhi and Zhang, Daoqiang and Huang, Kun.<br> "Diagnosis-guided multi-modal feature selection for prognosis prediction of lung squamous cell carcinoma" **MICCAI (2019).** [[paper](https://link.springer.com/chapter/10.1007/978-3-030-32251-9_13)]

- Pelka, Obioma and Nensa, Felix and Friedrich, Christoph M.<br> "Branding-fusion of meta data and musculoskeletal radiographs for multi-modal diagnostic recognition" **ICCV (2019).** [[paper](https://openaccess.thecvf.com/content_ICCVW_2019/papers/VRMI/Pelka_Branding_-_Fusion_of_Meta_Data_and_Musculoskeletal_Radiographs_for_ICCVW_2019_paper.pdf)]

 Medical Image Segmentation

- **LViT:** Li, Zihan and Li, Yunxiang and Li, Qingde and Wang, Puyang and Guo, Dazhou and Lu, Le and Jin, Dakai and Zhang, You and Hong, Qingqi.<br> "Lvit: language meets vision transformer in medical image segmentation" **TMI (2024).** [[paper](https://arxiv.org/pdf/2206.14718)] [[code](https://github.com/HUANGLIZI/LViT)]

- **SaLIP:** Aleem, Sidra and Wang, Fangyijie and Maniparambil, Mayug and Arazo, Eric and Dietlmeier, Julia and Curran, Kathleen and Connor, Noel EO' and Little, Suzanne.<br> "Test-Time Adaptation with SaLIP: A Cascade of SAM and CLIP for Zero-shot Medical Image Segmentation" **CVPR (2024).** [[paper](https://openaccess.thecvf.com/content/CVPR2024W/DEF-AI-MIA/papers/Aleem_Test-Time_Adaptation_with_SaLIP_A_Cascade_of_SAM_and_CLIP_CVPRW_2024_paper.pdf)] [[code](https://github.com/aleemsidra/SaLIP)]

- **SegICL:** Shen, Lingdong and Shang, Fangxin and Yang, Yehui and Huang, Xiaoshuang and Xiang, Shining.<br> "SegICL: A Universal In-context Learning Framework for Enhanced Segmentation in Medical Imaging" **arXiv (2024).** [[paper](https://arxiv.org/pdf/2403.16578)]

- **MedCLIP-SAM:** Koleilat, Taha and Asgariandehkordi, Hojat and Rivaz, Hassan and Xiao, Yiming.<br> "MedCLIP-SAM: Bridging text and image towards universal medical image segmentation" **arXiv (2024).** [[paper](https://arxiv.org/pdf/2403.20253)] [[code](https://github.com/HealthX-Lab/MedCLIP-SAM)]

- Kunhimon, Shahina and Naseer, Muzammal and Khan, Salman and Khan, Fahad Shahbaz.<br> "Language Guided Domain Generalized Medical Image Segmentation" **arXiv (2024).** [[paper](https://arxiv.org/pdf/2404.01272)] [[code](https://github.com/ShahinaKK/LG_SDG)]

- **RecLMIS:** Huang, Xiaoshuang and Li, Hongxiang and Cao, Meng and Chen, Long and You, Chenyu and An, Dong.<br> "Cross-Modal Conditioned Reconstruction for Language-guided Medical Image Segmentation" **arXiv (2024).** [[paper](https://arxiv.org/pdf/2404.02845)] [[code](https://github.com/ShawnHuang497/RecLMIS)]

- **${CPAM^{TG}}$:** Lee, Go-Eun and Kim, Seon Ho and Cho, Jungchan and Choi, Sang Tae and Choi, Sang-Il.<br> "Text-guided cross-position attention for segmentation: Case of medical image" **MICCAI (2023).** [[paper](https://link.springer.com/chapter/10.1007/978-3-031-43904-9_52)] [[code]()]

- **TPRO:** Zhang, Shaoteng and Zhang, Jianpeng and Xie, Yutong and Xia, Yong.<br> "TPRO: Text-Prompting-Based weakly supervised histopathology tissue segmentation" **MICCAI (2023).** [[paper](https://link.springer.com/chapter/10.1007/978-3-031-43907-0_11)] [[code](https://github.com/zhangst431/TPRO)]

- Liu, Jie and Zhang, Yixiao and Chen, Jie-Neng and Xiao, Junfei and Lu, Yongyi and A Landman, Bennett and Yuan, Yixuan and Yuille, Alan and Tang, Yucheng and Zhou, Zongwei.<br> "Clip-driven universal model for organ segmentation and tumor detection" **ICCV (2023).** [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_CLIP-Driven_Universal_Model_for_Organ_Segmentation_and_Tumor_Detection_ICCV_2023_paper.pdf)] [[code](https://github.com/ljwztc/CLIP-Driven-Universal-Model)]

- Han, Xianjun and Chen, Qianqian and Xie, Zhaoyang and Li, Xuejun and Yang, Hongyu.<br> "Multiscale progressive text prompt network for medical image segmentation" **Computers & Graphics (2023).** [[paper](https://www.sciencedirect.com/science/article/pii/S0097849323002170)]

- Lu, Yixing and Fan, Zhaoxin and Xu, Min.<br> "Multi-dimensional Fusion and Consistency for Semi-supervised Medical Image Segmentation" **International Conference on Multimedia Modeling (2024).** [[paper](https://arxiv.org/pdf/2309.06618)]

- **EMIT-Diff:** Zhang, Zheyuan and Yao, Lanhong and Wang, Bin and Jha, Debesh and Keles, Elif and Medetalibeyoglu, Alpay and Bagci, Ulas.<br> "Emit-diff: Enhancing medical image segmentation via text-guided diffusion model" **arXiv (2023).** [[paper](https://arxiv.org/pdf/2310.12868)]

- **GTGM:** Chen, Yinda and Liu, Che and Huang, Wei and Cheng, Sibo and Arcucci, Rossella and Xiong, Zhiwei.<br> "Generative text-guided 3d vision-language pretraining for unified medical image segmentation" **arXiv (2023).** [[paper](https://arxiv.org/pdf/2306.04811)]

- **Bi-VLGM:** Wenting, Chen and Jie, Liu and Yixuan, Yuan.<br> "Bi-VLGM: Bi-Level Class-Severity-Aware Vision-Language Graph Matching for Text Guided Medical Image Segmentation" **arXiv (2023).** [[paper](https://arxiv.org/pdf/2305.12231)]

- Segre, Leo and Hirschorn, Or and Ginzburg, Dvir and Raviv, Dan.<br> "Shape-consistent generative adversarial networks for multi-modal medical segmentation maps" **ISBI (2022).** [[paper](https://arxiv.org/pdf/2201.09693)] [[code](https://github.com/orhir/3D-Shape-Consistent-GAN)]

- **DTAN:** Zhao, Yiyang and Li, Jinjiang and Ren, Lu and Chen, Zheng.<br> "DTAN: Diffusion-based Text Attention Network for medical image segmentation" **Computers in Biology and Medicine (2024).** [[paper](https://www.sciencedirect.com/science/article/pii/S0010482523011939)]

- **TGEDiff:** Dong, Zhiwei and Yuan, Genji and Hua, Zhen and Li, Jinjiang.<br> "Diffusion model-based text-guided enhancement network for medical image segmentation" **Expert Systems with Applications (2024).** [[paper](https://www.sciencedirect.com/science/article/pii/S0957417424004147)]




## Medical Image-Text Retrieval
