# An Inferential Commonsense-Driven Framework for Predicting Political Bias in News Headlines

## Abstract

>Identifying political bias in news headlines is important as it impacts the selection and distribution of news articles. However, it is difficult because the short headline text lacks sufficient syntactic and semantic context. To compensate for this lack of information, inferential commonsense knowledge can be employed in a manner similar to how people use it in their daily lives. However, without proper emphasis, the additional inferential context is prone to introduce unnecessary noise. This noise can prevent models from fully exploiting the acquired knowledge. To address this, we propose a novel framework, IC-BAIT short for Inferential Commonsense aware BiAs IdenTifier. We also present two bias-annotated datasets: MediaBias and GoodNews. The experimental results demonstrate that IC-BAIT significantly enhances the performance of the baseline models.<br/>

<p align="center">
  <img src="https://github.com/Swati17293/IC-BAIT/blob/main/framework.png" alt="Framework" width="70%" height="70%"></br>
  Abstract representation of the bias prediction framework. 
</p>

For More details refer our paper (Coming Soon!!)

## Requirements

>We recommend Conda with Python3. 

Create the environment from the environment.yml file:
```
conda env create -f environment.yml
```
Activate the new environment:
```
conda activate envicbait
```

## Datasets: GoodNews and MediaBias
>GoodNews and MediaBias are available in their respective directories. <br/>
<br/>

To replicate the dataset, navigate to the appropriate dataset directory and execute the following commands.
<br/>

To collect annotated headlines using goodnews.tsv downloaded from https://www.ims.unistuttgart.de/en/research/resources/corpora/goodnewseveryone/ for goodnews and to scrape annotated headlines from allsides.com for mediabias:
```
python3 build_dataset.py
```

To preprocess the headlines:
```
python3 preprocess_headline.py
```

To generate IC_Knwl for the headlines:
```
python3 commet_commonsense.py
```

To split the dataset:
```
python3 split.py
```

## Evaluation 
>Navigate to the appropriate dataset directory and execute the following commands

To evaluate the baseline models:
```
python3 predict_bias.py 
```

To evaluate IC-BAIT:
```
python3 IC-BAIT.py 
```

## License
MIT License
