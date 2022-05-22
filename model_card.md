# HaRT model card

Last updated: May 2022

HaRT model's release information. Inspired by [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993).

## Model Details.

This model was developed as a first solution to the task of Human Language Modeling (HuLM) which further trains large language models to account for statistical dependence between documents (i.e. multiple documents written by the same human). This version of HaRT builds upon Open AI GPT-2 ([Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)) small and is developed with a base code from [HuggingFace](https://github.com/huggingface/transformers).

### Model date

May 2022, trained on social media (Twitter) dataset spanning tweets posted from 2009 through 2020.

### Model type

Human language model (can use anywhere language models are used). 

### Model version

hart-gpt2sml-twt-v1 (153M parameters)

### Paper or other resource for more information
[Website](https://nikita-soni-nlp.netlify.app/) and [paper](https://aclanthology.org/2022.findings-acl.52/)

### Where to send questions or comments about the model
[Contact us](https://nikita-soni-nlp.netlify.app/contact-us)

## Intended Uses:

### Primary intended uses

The primary intended use of this model is to make AI more human-centered. We encourage using HaRT wherever you would use a traditional large language model.
We also envision HaRT's user states to present oppotunites in models designed to enable bias correction and fairness techniques.

### Primary intended users

The primary intended users of the model are _AI and Psychology researchers and practioners_

### Out-of-scope use cases

At this point, our model is not intended to be used in practice for mental health care nor labeling of individuals publicly with personality or age scores.

## Factors
PEW statistics on Twitter users (skews young; everything else is fairly representative): <br/>

Gender: 50% female, 50% male <br/>
Age groups & percentage: 18-29 (29%), 30-49 (44%), 50-64 (19%), 65+ (8%) <br/>
Race/Ethnicity: 11% black, 60% white, 17% hispanic <br/>
Income range & perecntage: <$30,000 (23%), $30-$74,999 (36%), $75,000+ (41%) <br/>
Education groups &  percentage: Less than high school (4%), High school graduate (54%), College graduate+ (42%) <br/>

Reference: [Pew Research Center](https://www.pewresearch.org/internet/2019/04/24/sizing-up-twitter-users/pdl_04-24-19_twitter_users-00-04/)

## Training Data

The released model version trains only on the paper's Twitter dataset, which is a subset of the County Tweet Lexical Bank [Giorgi et al., 2018](https://aclanthology.org/D18-1148/) appended with newer 2019 and 2020 tweets, in total spanning 2009 through 2020.

_Note: The model in our paper also trained on Facebook data and thus achieved slightly better evalution results. We are not able to release the Facebook data but we will soon be releasing a model that includes other public social media data sources for training._

### Preprocessing

We filter the datasets to only include tweets marked as English from users who have at least 50 total posts and at least 1000 words in total, ensuring moderate language history for each user.

## Metrics

We use:
1. Perplexity for language modeling evaluation.
2. Weighted F1 for document-level classification tasks (fine-tuning).
3. Pearson r and Disattenuated pearson-r for user-level regression tasks (fine-tuning).

## Evaluation Data

We evaluate the utility of fine-tuning our pre-trained HaRT for document- and user-level tasks.
Document-level tasks include stance detection, sentiment classification.
User-level tasks include age estimation, and personality (openness) assessment tasks.

### Datasets

Language modeling evaluation: The pre-training dataset is split into a test set of new users (not seen during training), as described in the [paper](https://aclanthology.org/2022.findings-acl.52/).

Stance Detection: SemEval2016 dataset ([Mohammad et al., 2016] (https://aclanthology.org/S16-1003/) with the historical context as a subset of the extended dataset from [Lynn et al. (2019)] (https://aclanthology.org/W19-2103/)

Sentiment Analysis: SemEval-2013 dataset ([Nakov et al., 2013] (https://aclanthology.org/S13-2052/) with the historical context as a subset of the extended dataset from [Lynn et al. (2019)] (https://aclanthology.org/W19-2103/)

Twitter, Stance and Sentiments datasets are released and can be found on our [website](https://nikita-soni-nlp.netlify.app/) and on our [github](data/datasets)

We can not release the Facebook user-level task datasets for privacy considerations: participants did not consent to have their data shared openly. 

### Motivation

The motivation behind using Twitter is that it is a public dataset of human language (documents connected to users) that is prevalently used for language reserach. 

### Preprocessing

We filter the extended context datasets from Lynn et al. (2019) to include the only the historically created posts with respect to the labeled dataset from SemEval tasks.


## Ethical Considerations

Unlike other human-centered approaches, HaRT is not directly fed user attributes as part of the pre-training thus the model parameters do not directly encode user attributes.
While the multi-level human-document-word structure within HuLM can enable bias correcting and fairness techniques, the ability to better model language in its human context also presents opportunities for unintended harms or nefarious exploitation.
For example, models that improve psychological assessment are not only useful for research and clinical applications, but could be used to target content for individuals without their awareness or consent. In the context of use for psychological research, such models may risk release of private research participant information if trained on private data without checks for exposure of private information. To negate this potential, at this time we are only releasing a version of HaRT that is not trained on the consented-use private Facebook data. 
We will consider release of a model trained on the private posts as well if we can verify it meets differential privacy standards. 

## Caveats and Recommendations

While modeling the human state presents opportunities for reducing AI bias, prior to clinical or applied use, such models should be evaluated for failure modes such as error across target populations for error or outcome disparities ([Shah et al., 2020](https://aclanthology.org/2020.acl-main.468/)).

