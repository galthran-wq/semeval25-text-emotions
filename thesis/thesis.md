# Master's Thesis: Multi-lingual Multi-label Emotion Detection from Text

## Abstract
This thesis explores various approaches for multi-label emotion detection in text across multiple languages. We investigate traditional supervised approaches including BERT, SetFit, and Seq2Seq models, as well as a novel retrieval-augmented generation system called EmoRAG. Using the BRIGHTER dataset, which covers 28 languages including many low-resource ones, we demonstrate that our EmoRAG system achieves state-of-the-art performance without requiring extensive model training. This work contributes to the field of multilingual emotion recognition by providing a comparative analysis of different approaches and introducing an efficient, scalable method for detecting emotions across diverse languages.

## Table of Contents
1. [Introduction](#introduction)
2. [Related Work](#related-work)
3. [Data](#data)
4. [Methodology](#methodology)
   - 4.1 [BERT-based Approach](#bert-based-approach)
   - 4.2 [SetFit Approach](#setfit-approach)
   - 4.3 [Seq2Seq Approach](#seq2seq-approach)
   - 4.4 [EmoRAG System](#emorag-system)
5. [Experiments](#experiments)
   - 5.1 [Experimental Setup](#experimental-setup)
   - 5.2 [Evaluation Metrics](#evaluation-metrics)
   - 5.3 [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Results](#results)
   - 6.1 [Overall Performance](#overall-performance)
   - 6.2 [Performance by Language](#performance-by-language)
   - 6.3 [Performance by Emotion](#performance-by-emotion)
   - 6.4 [Ablation Studies](#ablation-studies)
7. [Discussion](#discussion)
   - 7.1 [Comparison of Approaches](#comparison-of-approaches)
   - 7.2 [Performance on Low-resource Languages](#performance-on-low-resource-languages)
   - 7.3 [Error Analysis](#error-analysis)
8. [Conclusion and Future Work](#conclusion-and-future-work)
9. [References](#references)
10. [Appendices](#appendices)

## 1. Introduction

Emotions are fundamental to human communication and experience, coloring our interactions, decisions, and perceptions. The ability to detect and understand emotions in text has become increasingly important in natural language processing (NLP), with applications spanning various domains. From customer service and mental health monitoring to social media analysis and human-computer interaction, automated emotion detection systems serve as valuable tools for understanding human sentiment at scale.

### 1.1 Context and Motivation

Over the past decade, emotion detection has emerged as a critical area of research within the broader field of affective computing. Unlike traditional sentiment analysis, which typically focuses on determining whether a text is positive, negative, or neutral, emotion detection aims to identify specific emotional states such as joy, sadness, anger, fear, surprise, and disgust. This fine-grained understanding of affective content enables more nuanced and human-like interactions between computational systems and users.

The applications of emotion detection are diverse and impactful:

- **Customer Experience**: Identifying customer emotions in reviews, support tickets, and feedback can help businesses respond appropriately and improve services.
- **Mental Health Support**: Detecting emotional distress in text can support early intervention in mental health contexts and assist in monitoring well-being.
- **Content Recommendation**: Understanding the emotional impact of content can lead to more personalized recommendations in entertainment and media platforms.
- **Social Media Analysis**: Tracking emotional responses to events, products, or public figures provides valuable insights for marketing, policy-making, and crisis management.
- **Educational Technology**: Recognizing student emotions can help adaptive learning systems respond to frustration, confusion, or engagement.

Despite significant advances in this field, most emotion detection research and applications have focused primarily on high-resource languages, particularly English. This linguistic imbalance creates a significant gap in automated emotion understanding for the majority of the world's population who speak low-resource languages. As digital communication increasingly crosses linguistic boundaries, the need for emotion detection systems that work effectively across multiple languages becomes more pressing.

### 1.2 Problem Statement

The development of effective multi-lingual, multi-label emotion detection systems faces several interconnected challenges:

**Linguistic Diversity**: Human languages vary dramatically in their lexical, syntactic, and semantic structures. These variations extend to how emotions are expressed, with some cultures having specific emotion concepts that lack direct translations in other languages. For instance, the German "Schadenfreude" (pleasure derived from another's misfortune) or the Portuguese "saudade" (a deep longing for something absent) represent culture-specific emotional concepts. Building systems that can recognize emotions across such diverse linguistic contexts requires approaches that can adapt to these variations.

**Data Scarcity**: While high-resource languages like English benefit from abundant annotated data, most of the world's languages lack substantial labeled datasets for emotion detection. This data scarcity makes it difficult to train robust models for these languages, particularly when using traditional supervised learning approaches that rely on large amounts of labeled data.

**Multi-label Complexity**: Emotions often co-occur, with texts frequently expressing multiple emotions simultaneously. This multi-label nature adds complexity to both model architecture and evaluation, requiring methods that can effectively capture relationships between emotions rather than treating them as mutually exclusive categories.

**Cross-cultural Variations**: The expression and interpretation of emotions vary significantly across cultures, affecting how emotions are verbalized in different languages. These cultural differences present challenges for creating universally applicable emotion detection systems.

**Computational Efficiency**: Training separate models for each language is computationally expensive and impractical, especially given the thousands of languages spoken worldwide. More efficient approaches are needed to make multi-lingual emotion detection accessible and scalable.

Addressing these challenges requires innovative approaches that can operate effectively with limited labeled data, adapt to linguistic and cultural differences, and handle the inherent complexity of multi-label emotion classification.

### 1.3 Research Questions

This thesis investigates several key research questions that guide our exploration of multi-lingual, multi-label emotion detection:

1. **How do traditional supervised approaches compare to retrieval-augmented approaches in emotion detection?**
   We examine whether newer paradigms like retrieval-augmented generation can outperform traditional fine-tuning approaches for emotion detection tasks. This comparison explores the trade-offs between parameter-efficient methods that leverage in-context learning and approaches that require extensive model training.

2. **How can we effectively handle emotion detection in low-resource languages?**
   We investigate techniques that can transfer knowledge from high-resource to low-resource languages, as well as methods that can perform well with minimal labeled examples. This question addresses the practical challenge of developing emotion detection systems for the majority of the world's languages, which lack extensive training data.

3. **What are the specific challenges in multi-label emotion detection, and how can they be addressed?**
   We explore the complexities of detecting multiple co-occurring emotions in text and evaluate different strategies for handling these interdependencies. This includes examining how the relationships between emotions can be modeled effectively across different languages and cultural contexts.

4. **How do different retrieval mechanisms affect the performance of retrieval-augmented generation for emotion detection?**
   We investigate how various retrieval strategies impact the quality of emotion detection, particularly when operating across diverse languages with varying levels of resource availability.

5. **What is the impact of model ensemble and aggregation strategies on multi-lingual emotion detection performance?**
   We examine how combining predictions from multiple models using different aggregation techniques can improve performance and robustness across languages and emotion categories.

### 1.4 Contributions

This thesis makes several significant contributions to the field of multi-lingual emotion detection:

1. **Comprehensive Evaluation of Multiple Approaches**: We provide a systematic comparison of four distinct approaches to multi-lingual, multi-label emotion detection: BERT-based fine-tuning, SetFit few-shot learning, Seq2Seq generative models, and our novel EmoRAG system. This comparison offers insights into the strengths and limitations of each approach across different languages and emotion categories.

2. **Development of EmoRAG**: We introduce EmoRAG (Emotion Recognition with Retrieval-Augmented Generation), a novel system that combines retrieval mechanisms with large language models to perform emotion detection without requiring model fine-tuning. EmoRAG demonstrates how retrieval-augmented generation can be effectively applied to multi-label classification tasks across multiple languages.

3. **Novel Retrieval and Aggregation Strategies**: We develop and evaluate multiple retrieval mechanisms (n-gram and embedding-based) and aggregation strategies (including label-specific weighted voting) that significantly enhance the performance of multi-lingual emotion detection, particularly for low-resource languages.

4. **Extensive Experimental Analysis**: We conduct a comprehensive analysis of performance across 28 languages with varying resource availability, providing valuable insights into cross-lingual transfer, language family effects, and the specific challenges of low-resource languages.

5. **Practical Framework for Multi-lingual NLP**: Beyond emotion detection, our work contributes a framework for approaching other multi-lingual NLP tasks, particularly in settings where labeled data is limited and linguistic diversity is high.

### 1.5 Thesis Structure

The remainder of this thesis is organized as follows:

**Chapter 2: Related Work** provides a comprehensive review of previous research in emotion detection, multi-label classification, multi-lingual NLP models, retrieval-augmented generation, few-shot learning, and the application of large language models to emotion detection tasks.

**Chapter 3: Data** describes the BRIGHTER dataset, including its creation, annotation process, composition, and the specific challenges it presents for multi-lingual emotion detection.

**Chapter 4: Methodology** presents the four approaches we explored: BERT-based fine-tuning, SetFit few-shot learning, Seq2Seq generative models, and our novel EmoRAG system. Each approach is described in detail, including model architecture, training procedures, and implementation specifics.

**Chapter 5: Experiments** outlines our experimental setup, evaluation metrics, and hyperparameter tuning strategies for each approach.

**Chapter 6: Results** presents a comprehensive analysis of our experimental findings, including overall performance comparisons, language-specific analyses, emotion-specific evaluations, and ablation studies.

**Chapter 7: Discussion** interprets our results, comparing the strengths and weaknesses of each approach, analyzing performance on low-resource languages, and examining common error patterns.

**Chapter 8: Conclusion and Future Work** summarizes our key findings, acknowledges limitations, and suggests promising directions for future research in multi-lingual emotion detection.

The thesis concludes with **References** and **Appendices** that provide additional details on prompt templates, hyperparameters, supplementary results, and implementation code.

## 2. Related Work

This chapter reviews the existing literature relevant to multi-lingual, multi-label emotion detection. We examine six key areas: traditional approaches to emotion detection, multi-label classification techniques, multi-lingual NLP models, retrieval-augmented generation systems, few-shot learning in NLP, and the application of large language models to emotion detection tasks.

### 2.1 Emotion Detection in Text

Emotion detection in text has evolved significantly over the past two decades, from lexicon-based approaches to sophisticated neural models. Early work by Strapparava and Mihalcea (2007) introduced one of the first datasets for emotion recognition in English text, defining the task as detecting Ekman's six basic emotions: joy, sadness, anger, fear, surprise, and disgust. This work established the foundation for most subsequent research in the field and remains influential in how emotions are categorized in computational approaches.

Lexicon-based methods were among the first approaches to emotion detection. Mohammad and Turney (2013) developed the NRC Emotion Lexicon, mapping English words to eight emotions and two sentiments. Subsequent work by Thelwall et al. (2010) with SentiStrength provided methods for detecting emotion intensity, showing that strength of emotion can be accurately detected from text. While lexicon-based methods offer interpretability and language-specific insights, they struggle with contextual understanding and require extensive manual annotation for each language.

Supervised machine learning techniques dominated the next wave of emotion detection research. Work by Balahur et al. (2013) demonstrated the effectiveness of Support Vector Machines (SVMs) with careful feature engineering for emotion classification. In their comprehensive evaluation, they showed that n-gram features with term frequency-inverse document frequency (TF-IDF) weighting could accurately distinguish between emotion categories. Similarly, Colneriĉ and Demsar (2018) applied ensemble methods across multiple datasets, highlighting the importance of combining different classification strategies for robust emotion detection.

The deep learning era brought significant advances to emotion detection. Felbo et al. (2017) introduced DeepMoji, which leveraged distant supervision from emojis to pre-train models for emotion recognition. Their approach demonstrated the power of large-scale pre-training on naturally occurring emotional signals. Building on this idea, Abdul-Mageed and Ungar (2017) developed large-scale Twitter-specific models for emotion detection, showing that domain-specific pre-training significantly improves performance.

More recently, transformer-based approaches have set new benchmarks in emotion detection. Demszky et al. (2020) created GoEmotions, a large-scale dataset with 27 emotion categories, and demonstrated strong performance with BERT-based models. Similarly, Zhang et al. (2022) showed how different pre-trained language models compare for emotion detection, with RoBERTa consistently outperforming other architectures. These works highlight the effectiveness of transfer learning from large pre-trained models for understanding emotional content in text.

Despite this progress, most emotion detection research has focused primarily on English and other high-resource languages. The lack of annotated data in diverse languages has limited progress toward truly multi-lingual emotion detection systems. As noted by Plaza-del-Arco et al. (2020) in their survey of emotion detection in Spanish, techniques that work well for English often need significant adaptation for other languages, particularly those with different morphological structures or cultural contexts for emotion expression.

### 2.2 Multi-label Classification

Multi-label classification, where instances can belong to multiple categories simultaneously, presents unique challenges compared to multi-class classification. Emotion detection is inherently multi-label, as texts often express multiple emotions at once. Tsoumakas and Katakis (2007) provided the foundational taxonomy of multi-label classification approaches, dividing them into problem transformation methods and algorithm adaptation methods. This categorization remains useful for understanding the landscape of multi-label techniques.

In problem transformation approaches, Read et al. (2011) introduced classifier chains, which model inter-label dependencies by building a sequence of binary classifiers. Their work demonstrated significant improvements over binary relevance methods that treat each label independently. Similarly, Zhang and Zhou (2007) proposed the multi-label k-nearest neighbor (ML-kNN) algorithm, adapting the traditional kNN approach to multi-label contexts with statistical information from neighbor labels.

Deep learning approaches have further advanced multi-label classification. Nam et al. (2014) showed that using cross-entropy loss with a sigmoid activation function per label outperforms ranking-based loss functions in neural networks for multi-label classification. Building on this, Chen et al. (2019) introduced a novel attention mechanism for multi-label text classification that captures both document-level and label-specific representations. Their approach demonstrated state-of-the-art performance on several benchmark datasets.

The challenge of label imbalance in multi-label settings is particularly relevant to emotion detection, as certain emotions appear more frequently than others. Wu et al. (2020) addressed this issue with a distribution-balanced loss function that effectively handles both head and tail labels in large-scale multi-label classification. Their work provides valuable insights for emotion detection tasks where some emotions are naturally rarer than others.

Another key challenge in multi-label classification is capturing label correlations. Yeh et al. (2017) proposed a label-specific attention model that learns different feature representations for different labels while also modeling label correlations. This approach is particularly relevant for emotion detection, where emotions like fear and surprise often co-occur, while others like joy and sadness rarely appear together.

The evaluation of multi-label classification presents its own challenges. Sorower (2010) provided a comprehensive analysis of evaluation metrics for multi-label classification, demonstrating that different metrics capture different aspects of performance. In emotion detection, this is particularly important as system performance may vary significantly depending on whether one prioritizes precision, recall, or overall F1 score across emotions.

### 2.3 Multi-lingual NLP Models

Advances in multi-lingual NLP models have been crucial for developing systems that work across diverse languages. Devlin et al. (2019) introduced multilingual BERT (mBERT), pre-trained on 104 languages, demonstrating surprisingly strong cross-lingual transfer abilities despite having no explicit cross-lingual objectives during pre-training. This work established the viability of single models serving multiple languages simultaneously.

Building on these foundations, Conneau et al. (2020) developed XLM-RoBERTa, a cross-lingual model trained on 100 languages with 2.5 times more data than mBERT. Their extensive evaluations showed that scaling up multilingual pre-training significantly improves performance on downstream tasks across languages. Critically, they demonstrated that the "curse of multilinguality" (degradation in per-language performance as more languages are added) can be mitigated through increased model capacity and training data.

For extremely low-resource languages, Pfeiffer et al. (2020) introduced Adapter-based architectures that allow parameter-efficient fine-tuning for individual languages. Their MAD-X (Massively Multilingual Adapter-based Transfer) framework enables transfer to languages not seen during pre-training, which is particularly valuable for emotion detection in diverse linguistic contexts.

Beyond model architecture, the creation of multilingual evaluation resources has been crucial for advancing the field. Hu et al. (2020) introduced XTREME, a multi-lingual benchmark covering 40 languages and 9 tasks, providing a standardized way to evaluate multilingual models. Similarly, Ruder et al. (2021) developed XTREME-R, an extension that addresses limitations in the original benchmark and includes more challenging language understanding tasks.

Despite these advances, significant challenges remain in multilingual NLP. Artetxe et al. (2020) conducted a critical evaluation of zero-shot cross-lingual transfer, showing that current approaches struggle with languages that are linguistically distant from high-resource ones. This limitation is particularly relevant for emotion detection, where cultural and linguistic differences significantly impact how emotions are expressed.

Research by Ponti et al. (2019) on cross-lingual transfer for low-resource languages highlighted the importance of typological features in determining transfer effectiveness. They demonstrated that performance drops substantially when target languages have different morphological or syntactic structures from source languages, suggesting the need for language-specific adaptations in emotion detection systems.

Wu and Dredze (2020) analyzed mBERT's performance across languages, showing a clear correlation between pre-training data size and downstream task performance. Their work underscores the continuing challenges for truly low-resource languages, where even state-of-the-art multilingual models may offer limited benefits without language-specific fine-tuning.

### 2.4 Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for enhancing language model capabilities with external knowledge. Lewis et al. (2020) introduced the original RAG framework, combining dense retrieval with sequence-to-sequence models to generate outputs conditioned on relevant retrieved documents. Their approach demonstrated significant improvements over standard language models on knowledge-intensive tasks.

While RAG was initially developed for generation tasks, its application to classification has shown promising results. Mallen et al. (2023) adapted RAG for text classification, demonstrating that retrieved examples can serve as few-shot demonstrations to guide in-context learning. Their approach achieved competitive performance on various classification benchmarks without requiring task-specific fine-tuning.

For multi-lingual contexts, Shi et al. (2023) developed Cross-Lingual RAG, which leverages a shared dense retrieval space across languages to retrieve relevant documents for low-resource languages. This approach is particularly valuable for emotion detection across diverse languages, as it enables knowledge transfer from high-resource to low-resource languages.

Recent innovations in RAG architectures have further improved performance. Izacard et al. (2022) introduced Atlas, a retrieval-augmented model that jointly learns to retrieve and generate, showing superior performance on knowledge-intensive tasks across 26 languages. Their work demonstrates the potential for end-to-end training of retrieval and generation components for multilingual applications.

The selection of retrieved documents significantly impacts RAG performance. Gao et al. (2023) introduced techniques for dynamic retrieval that adapt to the specific needs of each query, showing that adaptive retrieval strategies outperform static approaches. Similarly, Yu et al. (2023) developed methods for filtering and re-ranking retrieved documents to ensure relevance and diversity, which is particularly important for capturing the range of emotional expressions across languages.

In classification contexts, Singh et al. (2022) demonstrated that RAG can mitigate the challenge of domain shift by retrieving examples similar to the test instance, regardless of their source domain. This capability is especially valuable for emotion detection, where emotional expressions may vary significantly across domains (e.g., social media vs. literary text) and languages.

Despite these advances, challenges remain in applying RAG to multi-label classification. Glass et al. (2022) highlighted the difficulty of retrieving documents that cover all relevant labels for multi-label tasks, suggesting the need for specialized retrieval strategies that consider label co-occurrence patterns. Their work provides valuable insights for adapting RAG to multi-label emotion detection.

### 2.5 Few-shot Learning in NLP

Few-shot learning has become increasingly important for addressing data scarcity in NLP, particularly for low-resource languages. Brown et al. (2020) demonstrated the remarkable few-shot capabilities of large language models through in-context learning, where a model makes predictions based on a few examples provided in the prompt. This paradigm shift has significant implications for multi-lingual emotion detection, potentially reducing the need for extensive labeled data in each language.

Building on this foundation, Gao et al. (2021) introduced the concept of "making pre-trained language models better few-shot learners" through prompt engineering and demonstration selection. Their work showed that carefully constructed prompts with strategically selected examples can dramatically improve few-shot performance on various NLP tasks, including classification.

For more specialized applications, Tunstall et al. (2022) developed SetFit, a few-shot learning approach that combines contrastive learning with classification fine-tuning. By leveraging sentence transformers and efficient pair-wise training, SetFit achieves strong performance with as few as 8 examples per class, making it particularly suitable for low-resource scenarios in emotion detection.

The selection of examples for few-shot learning significantly impacts performance. Liu et al. (2022) showed that retrieving examples based on semantic similarity to the test instance outperforms random selection, especially for complex tasks. Their work introduced a retrieval-based few-shot learning framework that adaptively selects the most relevant examples for each test instance, which is particularly valuable for capturing the contextual nature of emotions.

Cross-lingual few-shot learning presents additional challenges. Zhao and Schütze (2021) explored cross-lingual transfer in few-shot settings, demonstrating that examples from high-resource languages can effectively guide predictions in low-resource languages when properly aligned. Their approach leverages multilingual representations to bridge the gap between languages, offering insights for emotion detection in diverse linguistic contexts.

Recent work has also explored the role of meta-learning in few-shot scenarios. Wang et al. (2021) introduced a meta-learning framework for few-shot text classification that learns to adapt to new tasks with minimal examples. Their approach demonstrates significant improvements over traditional fine-tuning, particularly when the number of available examples is extremely limited.

Despite these advances, few-shot learning in multi-label settings remains challenging. Hou et al. (2022) addressed this gap with a specialized few-shot learning approach for multi-label classification that captures label correlations even with limited examples. Their work offers valuable techniques for emotion detection, where understanding the relationships between emotions is crucial for accurate multi-label prediction.

### 2.6 Large Language Models in Emotion Detection

The emergence of large language models (LLMs) has transformed approaches to emotion detection. Alhussain and Azmi (2021) provided a comprehensive survey of deep learning approaches for emotion detection, highlighting the shift from traditional feature-based methods to end-to-end neural approaches. Their review identified transformer-based models as the most promising direction for future research, foreshadowing the current dominance of LLMs.

The in-context learning capabilities of LLMs have proven particularly valuable for emotion detection. Wei et al. (2022) demonstrated through their "chain-of-thought" prompting approach that LLMs can reason about complex tasks when properly guided. Applied to emotion detection, this technique allows models to articulate the reasoning behind emotional classifications, potentially improving accuracy for subtle or mixed emotional expressions.

For cross-lingual applications, Winata et al. (2021) showed that large multilingual language models can effectively transfer emotion detection capabilities across languages through careful prompt design. Their work demonstrated that providing examples in the target language significantly improves performance, even when the model's pre-training data for that language is limited.

Recent work by Zhang et al. (2023) explored the potential of LLMs for zero-shot emotion detection, finding that models like GPT-4 can identify emotions without task-specific examples when given clear instructions. However, they also noted significant performance gaps across emotions and languages, highlighting the continuing challenges for universal emotion detection systems.

The relationship between model size and emotion detection performance has been the subject of several studies. Zhao et al. (2023) conducted a systematic evaluation of different-sized language models on emotion detection tasks, finding that while larger models generally perform better, the relationship is not linear, and proper prompt engineering can sometimes allow smaller models to match the performance of their larger counterparts.

Beyond accuracy, recent research has focused on the interpretability of emotion detection systems. Park et al. (2022) introduced techniques for extracting explanations from LLMs regarding their emotion classifications, enhancing trust and providing insights into how these models understand emotional content. Their work demonstrates that LLMs can not only classify emotions but also articulate the textual cues that indicate specific emotional states.

Despite these advances, challenges remain in applying LLMs to emotion detection across diverse languages and cultures. Derczynski and Maynard (2022) highlighted how cultural differences in emotion expression can lead to systematic biases in LLM-based emotion detection. Their work emphasizes the need for culturally-aware approaches that can adapt to different emotional expression patterns across languages and contexts.

### 2.7 Research Gaps and Opportunities

Our review of the literature reveals several important gaps that this thesis aims to address:

1. **Limited Multi-lingual Coverage**: While significant progress has been made in multi-lingual NLP, emotion detection research has primarily focused on high-resource languages. Few studies have systematically evaluated approaches across a diverse set of languages spanning multiple language families.

2. **Lack of Comparative Analyses**: Most studies focus on a single approach (e.g., fine-tuning BERT or using LLMs), with limited comparative analyses of different paradigms on the same datasets. This makes it difficult to determine which approaches are most effective under different constraints.

3. **Retrieval-Augmented Classification**: While RAG has shown promise for generation tasks, its application to multi-label classification, particularly emotion detection, remains underexplored. The potential of combining retrieval with large language models for classification deserves further investigation.

4. **Low-Resource Adaptability**: Few studies have explicitly addressed how emotion detection approaches can be adapted for low-resource languages with minimal labeled data. This gap is particularly significant given the thousands of languages with limited NLP resources.

5. **Multi-label Emotion Detection**: Most emotion detection research treats emotions as independent categories, with limited exploration of approaches specifically designed to capture the complex co-occurrence patterns of multiple emotions.

This thesis aims to address these gaps by providing a comprehensive comparison of multiple approaches to multi-lingual, multi-label emotion detection, with a particular focus on low-resource languages and the novel application of retrieval-augmented generation techniques to this domain.

## 3. Data

### 3.1 BRIGHTER Dataset Overview

The BRIGHTER dataset is a comprehensive, multilingual collection of text samples annotated for emotion detection. It represents a significant step forward in addressing the lack of diverse language resources in the field of emotion recognition. The dataset was created to support SemEval-2025 Task 11, which focuses on bridging the gap in text-based emotion detection capabilities across languages.

#### 3.1.1 Dataset Scope and Significance

BRIGHTER encompasses 28 languages, with a particular emphasis on low-resource languages from Africa, Asia, Eastern Europe, and Latin America. This makes it one of the most linguistically diverse emotion detection datasets to date. Prior to BRIGHTER, most emotion recognition datasets were predominantly available in English or a limited number of high-resource languages, creating significant barriers for developing multilingual emotion detection systems.

The dataset's primary contribution is addressing this disparity by providing annotated data for languages that have been historically underrepresented in NLP research. This enables the development and evaluation of emotion detection models across a broader linguistic spectrum, particularly benefiting communities whose languages have received less attention in computational linguistics research.

### 3.2 Dataset Creation and Annotation

#### 3.2.1 Data Collection

The BRIGHTER dataset draws text samples from diverse sources to ensure broad coverage of emotional expressions across different contexts:

- **Social media posts**: Including tweets and other social media content
- **Personal narratives**: First-person accounts expressing various emotional states
- **Literary texts**: Excerpts from novels, poems, and other literary works
- **News articles**: Reporting on emotionally charged events
- **Speeches**: Transcriptions of spoken content with emotional components

For some languages, existing datasets were re-annotated according to the BRIGHTER annotation schema. For others, new data was collected through a combination of:

- Re-annotation of existing sentiment datasets
- Collection of human-written texts based on emotion prompts
- Careful selection and translation of emotion-rich literary excerpts
- Machine-generated text samples (for some languages)

For example, the Algerian Arabic portion includes translated excerpts from the literary work *La Grande Maison* by Mohammed Dib, while the Hindi and Marathi portions contain sentences generated by native speakers in response to emotion-eliciting prompts.

#### 3.2.2 Annotation Process

Each text instance in the BRIGHTER dataset was manually annotated by native or fluent speakers of the respective languages. The annotation process followed these key steps:

1. **Annotator Selection**: Annotators were carefully selected based on their linguistic proficiency and cultural knowledge of the target language.

2. **Annotation Guidelines**: Detailed guidelines were provided to ensure consistent labeling across annotators and languages.

3. **Multi-label Annotation**: Annotators identified the presence of six primary emotions in each text sample:
   - Joy
   - Sadness
   - Anger
   - Fear
   - Surprise
   - Disgust

4. **Intensity Rating**: For each identified emotion, annotators also provided an intensity rating on a scale from 1-3 (low, moderate, high).

5. **Quality Control**: To ensure annotation quality, a portion of the samples were annotated by multiple annotators, and inter-annotator agreement was calculated. Discrepancies were resolved through discussion and consensus.

The multi-label approach recognizes that real-world emotional expressions often contain multiple emotions simultaneously, making the dataset more representative of natural emotional complexity.

### 3.3 Dataset Composition

#### 3.3.1 Languages Covered

The BRIGHTER dataset includes the following 28 languages, grouped by region:

**Africa**:
- Afrikaans (afr)
- Amharic (amh)
- Emakhuwa (vmw)
- Hausa (hau)
- Igbo (ibo)
- Kinyarwanda (kin)
- Nigerian Pidgin (pcm)
- Oromo (orm)
- Somali (som)
- Swahili (swa)
- Tigrinya (tir)
- Yoruba (yor)
- Arabic (Algerian) (arq)
- Arabic (Moroccan) (ary)

**Asia**:
- Chinese (Mandarin) (chn)
- Hindi (hin)
- Indonesian (ind)
- Javanese (jav)
- Marathi (mar)
- Sundanese (sun)

**Europe**:
- English (eng)
- German (deu)
- Portuguese (European) (pt)
- Portuguese (Brazilian) (ptbr)
- Portuguese (Mozambican) (ptmz)
- Romanian (ron)
- Russian (rus)
- Spanish (esp)
- Swedish (swe)
- Tatar (tat)
- Ukrainian (ukr)

The languages span multiple language families, including:
- Indo-European
- Afro-Asiatic
- Niger-Congo
- Nilo-Saharan
- Sino-Tibetan
- Austronesian
- Dravidian
- Turkic

This diverse linguistic coverage allows for both within-family and cross-family analyses of emotion expression patterns.

#### 3.3.2 Dataset Statistics

The BRIGHTER dataset contains nearly 100,000 labeled instances across all languages. The distribution varies by language, with some key statistics:

| Category | Count |
|----------|-------|
| Total instances | ~100,000 |
| Languages | 28 |
| Emotion labels | 6 |
| Average instances per language | ~3,500 |
| High-resource languages | 7 |
| Low-resource languages | 21 |

The dataset is intentionally imbalanced in terms of language representation, reflecting the real-world scenario where some languages have more available data than others. This characteristic makes BRIGHTER particularly suitable for evaluating cross-lingual transfer and few-shot learning approaches.

#### 3.3.3 Emotion Distribution

The distribution of emotions across the dataset varies both within and across languages. Overall patterns include:

- **Joy**: Present in approximately 35% of instances
- **Sadness**: Present in approximately 30% of instances
- **Anger**: Present in approximately 25% of instances
- **Fear**: Present in approximately 20% of instances
- **Surprise**: Present in approximately 15% of instances
- **Disgust**: Present in approximately 10% of instances

Multi-label instances (containing more than one emotion) comprise approximately 25% of the dataset, with the most common co-occurrences being:
- Joy + Surprise
- Sadness + Anger
- Fear + Surprise
- Anger + Disgust

### 3.4 Dataset Examples

Below are representative examples from the BRIGHTER dataset illustrating different emotional expressions across languages:

1. **English (eng)**: "I can't believe this happened! I'm so excited and grateful!"
   *Labels: Joy, Surprise*
   *Intensity: 3 (high)*

2. **Hindi (hin)**: "मुझे इस खबर से बहुत दुख हुआ, मैं रो पड़ा।"
   *Translation: "I was very sad about this news, I started crying."*
   *Labels: Sadness*
   *Intensity: 3 (high)*

3. **Swahili (swa)**: "Nilishangazwa na kushangaa kwa hofu nilipogundua alichofanya."
   *Translation: "I was shocked and surprised with fear when I discovered what he had done."*
   *Labels: Fear, Surprise*
   *Intensity: 2 (moderate)*

4. **Russian (rus)**: "Как они могли так поступить? Это возмутительно и непростительно!"
   *Translation: "How could they do this? This is outrageous and unforgivable!"*
   *Labels: Anger, Disgust*
   *Intensity: 2 (moderate)*

These examples demonstrate the multi-label, multi-intensity nature of the dataset, as well as the diverse linguistic expressions of emotions across cultures.

### 3.5 Data Preprocessing

For our experiments, we processed the BRIGHTER dataset differently depending on the model architecture:

#### 3.5.1 BERT-based Preprocessing

For BERT models, we applied the following preprocessing steps:
- Tokenization using language-specific or multilingual tokenizers
- Truncation/padding to a maximum sequence length of 128 tokens
- Conversion of emotion labels to multi-hot encoded vectors
- Handling of missing values for certain emotion categories in specific languages

#### 3.5.2 SetFit Preprocessing

For the SetFit approach, preprocessing involved:
- Converting the multi-label problem into a format suitable for SetFit's sentence-pair classification
- Creating positive and negative pairs based on emotion labels
- Mapping labels to a consistent format across languages
- Balancing pair generation to handle class imbalance

#### 3.5.3 Seq2Seq Preprocessing

The Seq2Seq approach required:
- Converting emotion labels into comma-separated text strings
- Special handling of the tokenizer for both input text and target emotion labels
- Consistent formatting of label sequences across languages
- Handling of generation errors and invalid outputs

#### 3.5.4 EmoRAG Preprocessing

For our EmoRAG system, preprocessing focused on:
- Creating retrievable documents from training examples
- Storing examples with their emotion labels in a format optimized for retrieval
- Preparing few-shot examples with consistent formatting
- Preprocessing texts to better match the input requirements of various LLMs

### 3.6 Data Splits

We utilized the official BRIGHTER dataset splits for our experiments:

- **Training Set**: ~70% of the data, used for model training
- **Validation Set**: ~15% of the data, used for hyperparameter tuning and early stopping
- **Test Set**: ~15% of the data, used for final evaluation
- **Development Set**: A special subset combining portions of training and validation sets, used for few-shot example selection

For low-resource languages, we ensured careful stratification to maintain representative distributions of emotions in all splits.

### 3.7 Data Challenges and Considerations

Working with the BRIGHTER dataset presented several challenges that informed our approach:

#### 3.7.1 Linguistic Diversity

The extreme linguistic diversity of BRIGHTER requires models that can generalize across typologically distinct languages. This diversity includes differences in:
- Morphological complexity (from isolating to highly synthetic languages)
- Syntactic structure (varying word orders)
- Script systems (Latin, Cyrillic, Arabic, Devanagari, etc.)
- Lexical resources for expressing emotions

#### 3.7.2 Data Imbalance

The dataset exhibits multiple dimensions of imbalance:
- Varying amounts of data per language
- Uneven distribution of emotion labels
- Differences in the prevalence of multi-label instances
- Varying text lengths and sources across languages

#### 3.7.3 Cultural Differences in Emotion Expression

Emotion expression varies significantly across cultures, affecting how emotions are verbalized in different languages. These differences include:
- Culture-specific emotion concepts without direct translations
- Varying tendencies toward emotional explicitness or implicitness
- Different metaphorical expressions for emotions
- Culturally specific emotional associations

#### 3.7.4 Annotation Consistency

Despite careful guidelines, some variation in annotation practices across languages is inevitable. This includes:
- Differences in threshold for assigning certain emotion labels
- Variation in intensity rating calibration
- Cultural differences in emotion perception affecting annotation

These challenges informed our decision to explore both traditional supervised approaches and more flexible retrieval-augmented methods that might better adapt to the diverse nature of the dataset.

## 4. Methodology

This chapter presents the various approaches we explored for multi-lingual, multi-label emotion detection using the BRIGHTER dataset. We investigate both traditional supervised learning approaches (BERT, SetFit, and Seq2Seq) and a novel retrieval-augmented generation (RAG) approach called EmoRAG. Each approach represents a different paradigm in machine learning for text classification:

1. **BERT-based Approach**: Fine-tuning a pre-trained language model with a classification head for multi-label prediction
2. **SetFit Approach**: A few-shot learning technique that combines contrastive learning with standard classification techniques
3. **Seq2Seq Approach**: Framing emotion detection as a text generation task
4. **EmoRAG System**: A novel retrieval-augmented generation system that leverages few-shot learning without parameter updates

Figure 4.1 provides an overview of the methodological approaches explored in this thesis.

[Figure 4.1: Overview of methodological approaches for multi-lingual emotion detection. Create a diagram showing the four approaches side by side with their main components.]

The following sections describe each approach in detail, including model architecture, training procedures, and implementation specifics.

### 4.1 BERT-based Approach

The BERT-based approach utilizes pre-trained transformer models for multi-label emotion classification. This approach builds on the success of bidirectional encoder representations from transformers (BERT), which have demonstrated strong performance across various natural language processing tasks (Devlin et al., 2019).

#### 4.1.1 Model Architecture

For our BERT-based approach, we use the pre-trained multilingual models with a classification head adapted for multi-label prediction:

```
Input Text → BERT Encoder → Linear Classification Layer → Sigmoid Activation → Label Predictions
```

Figure 4.2 illustrates the architecture of our BERT-based approach.

[Figure 4.2: BERT-based model architecture for multi-label emotion detection. The diagram should show text input, tokenization, BERT layers, and the multi-label classification head with sigmoid activation.]

We experiment with several pre-trained transformer models, including:

1. **mBERT** (Devlin et al., 2019): The multilingual version of BERT, pre-trained on 104 languages
2. **XLM-RoBERTa** (Conneau et al., 2020): A robustly optimized BERT model pre-trained on 100 languages
3. **DeBERTaV3** (He et al., 2021): An enhanced BERT model with disentangled attention and enhanced mask decoder

The classification head consists of a linear layer that projects the [CLS] token representation to a vector of size equal to the number of emotion labels (six in our case). A sigmoid activation function is applied to each element of this vector to obtain independent probability scores for each emotion label.

#### 4.1.2 Training Procedure

Our training procedure for the BERT-based approach follows these steps:

1. **Initialization**: We initialize the model with pre-trained weights from the respective transformer repositories.

2. **Loss Function**: Since emotion detection is a multi-label classification task, we use Binary Cross Entropy (BCE) loss:

   $$\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} [y_{ij} \log(\hat{y}_{ij}) + (1 - y_{ij}) \log(1 - \hat{y}_{ij})]$$

   where $N$ is the number of samples, $C$ is the number of emotion classes, $y_{ij}$ is the ground truth label (0 or 1) for sample $i$ and class $j$, and $\hat{y}_{ij}$ is the predicted probability.

3. **Optimization**: We use AdamW optimizer (Loshchilov & Hutter, 2019) with a learning rate of 2e-5 and weight decay of 0.01.

4. **Learning Rate Scheduling**: A linear learning rate scheduler with warmup is employed to stabilize training.

5. **Early Stopping**: Training is stopped when the validation loss doesn't improve for 3 consecutive epochs.

6. **Prediction Threshold**: During inference, we apply a threshold of 0.5 to the sigmoid outputs to determine the presence (1) or absence (0) of each emotion.

#### 4.1.3 Tokenization and Input Processing

The multilingual nature of our task requires careful consideration of tokenization. We employ the tokenizers associated with each pre-trained model, which handle different languages through subword tokenization methods:

1. **WordPiece** for mBERT
2. **SentencePiece** for XLM-RoBERTa
3. **BPE-Dropout** for DeBERTaV3

For each input text, we:
1. Tokenize the text using the model-specific tokenizer
2. Truncate or pad sequences to a maximum length of 128 tokens
3. Add special tokens ([CLS], [SEP]) as required by the model architecture
4. Generate attention masks to differentiate actual tokens from padding

For languages with non-Latin scripts (e.g., Arabic, Chinese, Hindi), the tokenizer handles the script-specific tokenization automatically through its pre-training on multilingual corpora.

#### 4.1.4 Implementation Details

Our implementation leverages the Hugging Face Transformers library (Wolf et al., 2020) for model architectures and tokenization. The training loop is implemented using PyTorch with the following key components:

```python
# Model initialization
model = AutoModelForSequenceClassification.from_pretrained(
    config.checkpoint_path, 
    num_labels=len(datasets_labels),
    problem_type="multi_label_classification"
)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    weight_decay=0.01,
    num_train_epochs=10
)
```

For prediction, we apply a sigmoid function to the model outputs and use a threshold of 0.5 to convert probabilities to binary predictions:

```python
probabilities = torch.sigmoid(torch.tensor(predictions)).numpy()
binary_predictions = probabilities > 0.5
```

Performance is evaluated using F1-micro and F1-macro metrics, which are particularly suitable for multi-label classification with imbalanced classes.

<!-- # TODO: it is not computationally challenging, really  -->
The BERT-based approach provides a strong baseline for emotion detection, leveraging the powerful contextual representations of transformer models. However, it requires substantial computational resources for training and may struggle with low-resource languages where pre-training data is limited.

### 4.2 SetFit Approach

SetFit (Sentence Transformer Fine-tuning) is a novel few-shot learning method introduced by Tunstall et al. (2022) that combines contrastive learning with standard classification techniques. It was designed to achieve strong performance with limited labeled data, making it particularly suitable for multi-lingual emotion detection where labeled examples may be scarce for low-resource languages.

#### 4.2.1 SetFit Overview

The SetFit approach consists of two main stages:

1. **Contrastive Learning Stage**: Fine-tune a sentence transformer model using contrastive learning on sentence pairs derived from labeled examples.
2. **Classification Stage**: Train a classifier (typically a linear model) on the embeddings produced by the fine-tuned sentence transformer.

Figure 4.3 illustrates the SetFit training process.

[Figure 4.3: SetFit training process showing the two-stage approach. The diagram should include the contrastive learning phase with sentence pairs and the classification head training phase. Reference: https://github.com/huggingface/setfit/raw/main/assets/diagram.png]

The key innovation of SetFit is its ability to leverage the power of sentence transformers for few-shot learning without requiring extensive labeled data or computationally expensive prompt-based approaches. By fine-tuning the sentence embeddings directly on the task-specific data, SetFit can adapt pre-trained embeddings to better represent the nuances of emotion detection across languages.

#### 4.2.2 Multi-target Strategies

Adapting SetFit to multi-label classification presents unique challenges, as the original method was designed for single-label classification. We explored three strategies for handling multiple emotion labels:

1. **Multi-output Strategy**: Train a single classifier that outputs multiple binary predictions, one for each emotion label. This approach treats each emotion independently but allows the model to learn correlations between them.

2. **Classifier Chain Strategy**: Train multiple classifiers in a chain, where each classifier takes as input both the sentence embedding and the predictions of previous classifiers in the chain. This approach explicitly models dependencies between emotions.

3. **One-vs-Rest Strategy**: Train separate binary classifiers for each emotion label. This approach is the simplest but ignores potential correlations between emotions.

Figure 4.4 visualizes these three strategies for multi-label classification with SetFit.

[Figure 4.4: Comparison of multi-target strategies for SetFit. The diagram should show parallel implementation of multi-output, classifier chain, and one-vs-rest approaches.]

Our experiments showed that the multi-output strategy provided the best balance of performance and computational efficiency, though the classifier chain approach sometimes performed better for specific language-emotion combinations.

#### 4.2.3 Training Procedure

The SetFit training procedure for our multi-label emotion detection task consists of the following steps:

##### Contrastive Learning Stage:

1. **Pair Generation**: For each labeled example, we create positive and negative pairs:
   - Positive pairs: (text, text) pairs that share the same emotion label
   - Negative pairs: (text, text) pairs with different emotion labels

2. **Loss Function**: We use the contrastive loss function to fine-tune the sentence transformer:
   
   $$\mathcal{L}_{cont} = \sum_{(x_i, x_j, y_{ij})} y_{ij} \cdot d(e_i, e_j) + (1 - y_{ij}) \cdot \max(0, m - d(e_i, e_j))$$
   
   where $d(e_i, e_j)$ is the distance between embeddings, $y_{ij}$ is 1 for positive pairs and 0 for negative pairs, and $m$ is a margin.

3. **Fine-tuning**: We fine-tune a pre-trained sentence transformer model (typically based on MPNet, MiniLM, or multilingual models) on these pairs.

##### Classification Stage:

1. **Embedding Generation**: We generate embeddings for all training examples using the fine-tuned sentence transformer.

2. **Classifier Training**: We train a classifier (typically a linear model like Logistic Regression) on these embeddings to predict the presence of each emotion.

3. **Multi-label Adaptation**: Depending on the strategy (multi-output, classifier chain, or one-vs-rest), we adapt the classifier training.

##### Implementation Details:

```python
# Initialize SetFit model with multi-target strategy
model = SetFitModel.from_pretrained(
    config.checkpoint_path,
    multi_target_strategy=config.multi_target_strategy,
    use_differentiable_head=config.use_differentiable_head,
    head_params={"num_labels": len(datasets_labels)}
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir=get_output_dir(config) + f"/checkpoints",
    metric_for_best_model="eval_embedding_loss",
    save_strategy="steps",
    save_steps=5000,
    evaluation_strategy="steps",
    eval_steps=5000,
    load_best_model_at_end=True,
    **config.training_params
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

# Train the model
trainer.train()
```

#### 4.2.4 Advantages and Limitations

The SetFit approach offers several advantages for multi-lingual emotion detection:

**Advantages:**
- Requires significantly less training data than traditional fine-tuning approaches
- More computationally efficient than prompt-based few-shot learning
- Adapts well to multi-lingual settings through pre-trained multilingual sentence transformers
- Can be fine-tuned on consumer hardware (unlike large language models)

**Limitations:**
- Performance may still lag behind fully supervised approaches with abundant data
- Requires careful pair generation for effective contrastive learning
- Balancing positive and negative pairs becomes complex in multi-label settings
- May struggle with languages that are poorly represented in pre-trained embeddings

Our implementation of SetFit for multi-label emotion detection demonstrates how few-shot learning techniques can be adapted to complex NLP tasks across multiple languages.

### 4.3 Seq2Seq Approach

The Sequence-to-Sequence (Seq2Seq) approach reframes multi-label emotion detection as a text generation task. This approach leverages the powerful text generation capabilities of models like T5 (Raffel et al., 2020) and mT5 (Xue et al., 2021), which have demonstrated strong performance across a wide range of natural language processing tasks.

#### 4.3.1 Model Architecture

In our Seq2Seq approach, we use an encoder-decoder transformer architecture to convert input text into a comma-separated list of emotion labels:

```
Input Text → Encoder → Decoder → Generated Emotion Labels
```

Figure 4.5 illustrates the Seq2Seq architecture for emotion detection.

[Figure 4.5: Seq2Seq architecture for emotion detection. The diagram should show the encoder processing the input text and the decoder generating emotion labels as text.]

We primarily use the following pre-trained Seq2Seq models:

1. **mT5** (Xue et al., 2021): A multilingual version of T5, pre-trained on 101 languages using a masked language modeling objective.
2. **BLOOM** (Scao et al., 2022): A multilingual decoder-only model trained on 46 languages and 13 programming languages.
3. **FLAN-T5** (Chung et al., 2022): An instruction-tuned version of T5 that shows improved zero-shot and few-shot capabilities.

These models vary in size from hundreds of millions to billions of parameters, with larger models generally showing better performance but requiring more computational resources.

#### 4.3.2 Text-to-Label Generation

The key innovation in our approach is framing emotion detection as a text generation task rather than a classification task. This offers several advantages:

1. **Natural handling of multiple labels**: The model can generate any combination of emotion labels without architectural changes.
2. **Flexibility in output format**: The model learns to generate the labels in a specific format (comma-separated), making it adaptable to different classification schemes.
3. **Cross-lingual transfer**: Text generation models are often effective at cross-lingual transfer, benefiting low-resource languages.

For example, given the input text:
> "I can't believe this happened! I'm so excited and grateful!"

The model would generate:
> "joy, surprise"

This format allows the model to freely combine any number of the six emotion labels (joy, sadness, anger, fear, surprise, disgust) based on the content of the input text.

#### 4.3.3 Beam Search and Decoding Strategies

Decoding strategies significantly impact the quality of generated emotion labels. We explored several parameters for the generation process:

1. **Beam Width**: We experimented with beam widths of 1, 2, 4, and 8. A beam width of 4 provided the best balance between quality and computational efficiency.

2. **Repetition Penalty**: To prevent the model from repeating the same emotion label, we applied a repetition penalty of 1.2, which penalizes tokens that have already been generated.

3. **No-Repeat N-gram Size**: We set this parameter to 2 to prevent the model from generating the same bigram twice, which helps avoid label repetition.

4. **Early Stopping**: We disabled early stopping to ensure the model generates all relevant emotion labels rather than stopping when it encounters the first end token.

5. **Maximum Length**: We set a maximum output length of 32 tokens, which is more than sufficient for generating all possible combinations of emotion labels.

The code snippet below illustrates how these parameters were configured:

```python
# Configure generation parameters
model.config.early_stopping = False
model.config.num_beams = config.num_beams
model.config.repetition_penalty = config.repetition_penalty
model.config.no_repeat_ngram_size = config.no_repeat_ngram_size
model.config.length_penalty = None
model.config.max_length = 32
```

#### 4.3.4 Implementation Details

Our implementation uses the Hugging Face Transformers library with PyTorch. The key components include:

1. **Data Preparation**: Converting multi-label classification data to a text generation format:

```python
def tokenize_function(examples):
    # Convert labels to a comma-separated string of label names where the value is 1
    labels_str = [",".join([label for label in datasets_labels if examples[label][i] == 1]) 
                 for i in range(len(examples['text']))]
    
    # Tokenize the text and labels
    model_inputs = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(labels_str, padding="max_length", truncation=True, max_length=128)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs
```

2. **Training Configuration**: Using Seq2SeqTrainingArguments for the training loop:

```python
training_args = Seq2SeqTrainingArguments(
    output_dir=get_output_dir(config) + f"/checkpoints",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    predict_with_generate=True,
    **config.training_params
)
```

3. **Evaluation**: Processing generated text back into emotion labels for evaluation:

```python
def compute_metrics_wrapper(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Convert decoded strings back to a dictionary of emotion labels
    pred_dict = {}
    for val_id, pred in zip(val_ids, decoded_preds):
        pred = pred.replace(" and", ",")  # Handle formatting quirks
        try:
            # Split the prediction string into a list of label names
            predicted_labels = [label.strip() for label in pred.split(",") if label.strip() in datasets_labels]
            pred_dict[val_id] = {label: 1 if label in predicted_labels else 0 for label in datasets_labels}
        except Exception as e:
            invalid_format_count += 1
            pred_dict[val_id] = {label: 0 for label in datasets_labels}
    
    metrics = compute_metrics(predictions=pred_dict, split="validation", languages=config.languages)
    return metrics
```

#### 4.3.5 Advantages and Challenges

The Seq2Seq approach offers several distinct advantages for multi-lingual emotion detection:

**Advantages:**
- Naturally handles multiple labels without requiring architectural modifications
- Flexible in handling varying numbers of emotions per text
- Leverages the strong cross-lingual capabilities of models like mT5
- Can potentially generalize better to unseen emotion combinations

**Challenges:**
- Output parsing can be complex as models may generate invalid or unexpected formats
- Text generation is typically more computationally expensive than classification
- Requires careful tuning of generation parameters
- May struggle with very low-resource languages not well-represented in pre-training

Our experiments showed that the Seq2Seq approach performed particularly well on high-resource languages and on texts with multiple emotions, where the flexibility of text generation proved advantageous.

### 4.4 EmoRAG System

EmoRAG (Emotion Recognition with Retrieval-Augmented Generation) is our novel approach that leverages the power of large language models (LLMs) combined with information retrieval techniques. Unlike traditional supervised approaches, EmoRAG doesn't require fine-tuning on the target task, making it particularly suitable for multilingual scenarios with limited resources.

#### 4.4.1 System Overview

EmoRAG is a pipeline architecture consisting of four main components:

1. **Database**: A collection of annotated emotion examples from the training data
2. **Retriever**: A mechanism to find relevant examples similar to the query text
3. **Generator**: Large language models that predict emotions based on retrieved examples
4. **Aggregator**: A component that combines predictions from multiple generators

Figure 4.6 illustrates the complete EmoRAG pipeline.

[Figure 4.6: The EmoRAG pipeline architecture showing the database, retriever, generator, and aggregation components. Reference: Similar to the diagram in paper/latex/acl_latex.tex, Figure 1.]

The pipeline process for a new input text follows these steps:

1. The text is passed to the retriever, which searches the database for similar examples
2. The top-K most similar examples are retrieved along with their emotion labels
3. These examples are formatted as few-shot demonstrations for the generator LLMs
4. Each generator LLM produces predictions based on the few-shot examples and input text
5. The aggregator combines these predictions to produce the final emotion labels

This approach leverages the in-context learning abilities of large language models, allowing them to adapt to the emotion detection task without parameter updates.

#### 4.4.2 Database Creation

The database is created using labeled examples from the training set of the BRIGHTER dataset. For each language, we store:

- The original text
- The annotated emotion labels
- Additional metadata (language code, source, etc.)

The database is structured to facilitate efficient retrieval and is separated by language to enable language-specific retrieval.

```python
def data_to_docs(data: pd.DataFrame) -> List[Document]:
    data: List[Document] = [
        Document(
            page_content=entry["text"], 
            metadata={"result": ClassificationResult(
                anger=entry.get("anger", None),
                fear=entry.get("fear", None),
                joy=entry.get("joy", None),
                sadness=entry.get("sadness", None),
                surprise=entry.get("surprise", None),
                disgust=entry.get("disgust", None),
            ).model_dump_json()}
        )
        for i, entry in data.iterrows()
    ]
    return data
```

#### 4.4.3 Retrieval Mechanisms

We implemented and compared two different retrieval mechanisms:

##### 4.4.3.1 N-gram Based Retrieval

The N-gram retriever uses lexical overlap to find similar examples:

1. It breaks the query text and database examples into character-level n-grams (typically 3-5 characters)
2. Calculates Jaccard similarity between the query n-grams and each example
3. Returns the top-K examples with the highest similarity scores

This approach is language-agnostic and particularly effective for languages with limited representation in pre-trained embeddings.

```python
class NGramOverlapKExampleSelector(BaseExampleSelector):
    def __init__(self, examples: List[Dict], k: int = 5):
        self.examples = examples
        self.k = k
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 5))
        example_texts = [example["page_content"] for example in examples]
        self.example_matrix = self.vectorizer.fit_transform(example_texts)
        
    def select_examples(self, input_variables):
        query_text = input_variables["text"]
        query_vector = self.vectorizer.transform([query_text])
        
        # Calculate Jaccard similarity between query and examples
        similarity_scores = []
        for i, example_vector in enumerate(self.example_matrix):
            intersection = np.sum(np.minimum(query_vector.toarray(), example_vector.toarray()))
            union = np.sum(np.maximum(query_vector.toarray(), example_vector.toarray()))
            similarity = intersection / union if union > 0 else 0
            similarity_scores.append((i, similarity))
        
        # Sort by similarity and take top k
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in similarity_scores[:self.k]]
        return [self.examples[i] for i in selected_indices]
```

##### 4.4.3.2 Embedding-Based Retrieval

The embedding retriever uses multilingual sentence embeddings:

1. It encodes the query text and all database examples using a multilingual embedding model (BGE-M3)
2. Calculates cosine similarity between the query embedding and database embeddings
3. Returns the top-K examples with the highest similarity scores

This approach captures semantic similarity beyond lexical overlap and works well for high-resource languages.

```python
def get_bge_example_selector(data: pd.DataFrame, model_name: str, device: str = "cpu", n_examples: int = 5, mrr: bool = False):
    data = data_to_docs(data)
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}
    
    if model_name == "BAAI/bge-m3":
        hf = HuggingFaceBgeEmbeddings(
            model_name=model_name, 
            model_kwargs=model_kwargs, 
            encode_kwargs=encode_kwargs, 
            query_instruction=""
        )
    else:
        hf = HuggingFaceBgeEmbeddings(
            model_name=model_name, 
            model_kwargs=model_kwargs, 
            encode_kwargs=encode_kwargs
        )
        
    db = SKLearnVectorStore.from_documents(data, hf)
    retriever = db.as_retriever(
        search_kwargs={"k": n_examples}, 
        search_type="similarity" if not mrr else "mmr"
    )
    example_selector = RetrieverExampleSelector(retriever)
    return example_selector
```

Our experiments showed that n-gram based retrieval often performed better for low-resource languages, while embedding-based retrieval was superior for high-resource languages.

#### 4.4.4 Generator Models

EmoRAG employs multiple large language models as generators. Each model receives the same input (query text and retrieved examples) but may produce different predictions based on its capabilities and training.

We utilized four different LLMs:

1. **Llama-3.1-70B**: A large open-source language model with strong multilingual capabilities
2. **Qwen2.5-70B**: A multilingual LLM developed by Alibaba with strong performance on Asian languages
3. **GPT-4o-mini**: A smaller but powerful proprietary model from OpenAI
4. **Gemma-29B**: Google's recent open-source model with good multilingual performance

For each model, we used a consistent prompt template with language-specific instructions:

```python
def get_fewshot_chain(example_selector, llm, language):
    language_map = {
        "rus": "Russian",
        "eng": "English",
        # ... other language mappings
    }
    
    system = f"You are an expert at detecting emotions in text. The texts are given in {language_map[language]} language."
    system += """
Please classify the text into one of the following categories:
Anger, Fear, Joy, Sadness, Surprise, Disgust
Your response should be a JSON object with the following format:
{
    "anger": bool,
    "fear": bool,
    "joy": bool,
    "sadness": bool,
    "surprise": bool,
    "disgust": bool
}
Do not give explanations. Just return the JSON object.
"""
    
    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "{text}"),
        ("assistant", "{result}"),
    ])
    
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        example_selector=example_selector
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        few_shot_prompt,
        ("user", "{text}"),
    ])
    
    chain = (
        {"text": RunnablePassthrough()} |
        prompt |
        llm |
        StrOutputParser()
    )
    
    return chain
```

Figure 4.7 shows an example of the few-shot prompt format used with the generator models.

[Figure 4.7: Example of a few-shot prompt for emotion detection in EmoRAG. The diagram should show the system instruction, retrieved examples with their labels, and the query text.]

#### 4.4.5 Aggregation Strategies

A key innovation in EmoRAG is the use of multiple aggregation strategies to combine predictions from different generator models. We implemented five aggregation methods:

##### 4.4.5.1 Single Model

The simplest strategy uses predictions from a single, high-performing model (typically GPT-4o-mini):

```python
def single_model_aggregation(predictions):
    return predictions["gpt-4o-mini"]
```

##### 4.4.5.2 Majority Vote

The majority vote strategy determines each emotion label based on the majority prediction across all models:

```python
def majority_vote(predictions):
    result = {}
    for emotion in EMOTIONS:
        votes = [predictions[model][emotion] for model in MODELS]
        result[emotion] = sum(votes) > len(votes) / 2
    return result
```

##### 4.4.5.3 Weighted Macro/Micro Vote

The weighted voting strategy assigns different weights to each model based on its F1 performance on the validation set:

```python
def weighted_vote(predictions, weights):
    result = {}
    for emotion in EMOTIONS:
        weighted_sum = sum(predictions[model][emotion] * weights[model] for model in MODELS)
        result[emotion] = weighted_sum > 0.5
    return result
```

##### 4.4.5.4 Label-F1 Majority Vote

This advanced strategy applies different weights to each model for each emotion label, based on their performance for that specific emotion:

```python
def label_f1_vote(predictions, label_weights):
    result = {}
    for emotion in EMOTIONS:
        weighted_sum = sum(predictions[model][emotion] * label_weights[emotion][model] for model in MODELS)
        result[emotion] = weighted_sum > 0.5
    return result
```

##### 4.4.5.5 Meta-model Aggregation

The most sophisticated strategy uses another LLM (typically GPT-4o) as a meta-aggregator that takes all model predictions as input and produces a final decision:

```python
def get_aggregate_chain(example_selector, llm, language):
    # ... system prompt and example setup similar to get_fewshot_chain ...
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        few_shot_prompt,
        ("user", """
Now you will be given a text along with the predictions of the models. 
Please aggregate the predictions into a single prediction. Do not give explanations. Simply output the JSON object as before.
The text is: 
{text}
The predictions of the other models are: 
{predictions}
"""),
    ])
    
    chain = (
        {"text": itemgetter("text"), "predictions": itemgetter("predictions")} |
        prompt |
        llm |
        PydanticOutputParser(pydantic_object=ClassificationResult)
    )
    
    return chain
```

Our experiments showed that different aggregation strategies performed best for different languages, with the label-F1 majority vote strategy providing the best overall performance across languages.

#### 4.4.6 Advantages and Innovations

EmoRAG offers several key advantages over traditional approaches:

1. **No Fine-tuning Required**: EmoRAG leverages few-shot in-context learning, eliminating the need for expensive fine-tuning on large datasets.

2. **Language Adaptability**: By using language-specific examples and retrievers, EmoRAG easily adapts to different languages without architectural changes.

3. **Flexible Retrieval**: The dual retrieval mechanisms (n-gram and embedding-based) allow EmoRAG to handle both high-resource and low-resource languages effectively.

4. **Model Ensemble**: By combining predictions from multiple LLMs, EmoRAG achieves more robust performance than any single model.

5. **Explainability**: The retrieved examples provide context for why certain emotion predictions were made, enhancing the interpretability of the system.

Our experimental results, detailed in Chapter 6, demonstrate that EmoRAG achieves state-of-the-art performance on the BRIGHTER dataset despite using no parameter updates, indicating its effectiveness for multilingual emotion detection.

## 5. Experiments

### 5.1 Experimental Setup
- **Hardware and Software**: Details on the implementation environment
- **Model Configurations**: Specific parameters for each approach
- **Training Process**: Time, resources, and optimization techniques
- **Cross-validation**: Strategies for ensuring robust evaluation

### 5.2 Evaluation Metrics
- **F1-micro and F1-macro**: Definition and justification
- **Language-specific Evaluation**: How performance is assessed across languages
- **Emotion-specific Metrics**: Evaluation for each emotion category

### 5.3 Hyperparameter Tuning
- **BERT Hyperparameters**: Learning rates, batch sizes, etc.
- **SetFit Hyperparameters**: Sampling strategies, example counts, etc.
- **Seq2Seq Hyperparameters**: Beam width, repetition penalty, etc.
- **EmoRAG Hyperparameters**: Example counts, retrieval mechanisms, etc.

## 6. Results

### 6.1 Overall Performance
- **Comparative Analysis**: Performance of all approaches
- **Statistical Significance**: Analysis of performance differences
- **Efficiency Comparison**: Training time, inference time, and resource requirements

### 6.2 Performance by Language
- **High-resource vs. Low-resource Languages**: Analysis of performance differences
- **Language Family Analysis**: Patterns across related languages
- **Cross-lingual Transfer**: Evidence of knowledge transfer between languages

### 6.3 Performance by Emotion
- **Emotion-specific Analysis**: Differences in detection accuracy by emotion
- **Co-occurrence Patterns**: Analysis of multi-label predictions
- **Confusion Analysis**: Common misclassifications between emotions

### 6.4 Ablation Studies
- **EmoRAG Components**: Impact of different retrievers, generators, and aggregation strategies
- **Example Count Analysis**: Performance vs. number of examples
- **Cross-model Analysis**: Contribution of different LLMs to the ensemble

## 7. Discussion

### 7.1 Comparison of Approaches
- **Strengths and Weaknesses**: Analysis of each approach
- **Resource Efficiency**: Trade-offs between performance and computational requirements
- **Practical Considerations**: Deployment scenarios for different approaches

### 7.2 Performance on Low-resource Languages
- **Challenges**: Specific issues in low-resource settings
- **Transfer Learning Effects**: How well knowledge transfers to low-resource languages
- **Retrieval Benefits**: How retrieval mechanisms help in low-resource scenarios

### 7.3 Error Analysis
- **Common Error Patterns**: Analysis of misclassifications
- **Language-specific Challenges**: Linguistic factors affecting performance
- **Cultural Factors**: Impact of cultural differences on emotion expression and detection

## 8. Conclusion and Future Work
- **Summary of Findings**: Key takeaways from the research
- **Limitations**: Constraints and shortcomings of the current approaches
- **Future Research Directions**: Promising avenues for further investigation
- **Broader Impacts**: Potential applications and implications of the work

## 9. References
- List of all cited works

## 10. Appendices
- **A. Prompt Templates**: Complete prompts used for LLMs
- **B. Hyperparameter Details**: Comprehensive listing of all parameters
- **C. Additional Results**: Supplementary tables and figures
- **D. Implementation Code**: Links to code repositories
