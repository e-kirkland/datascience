# NLP Assessment

The purpose of this project is to analyze a dataset of news articles in two ways:
1. Parsing names and dates from each article in a useable format.
2. Applying text analytics to identify 5 news articles from the dataset that are most similar to a given sample article.

This project is an assessment given by Deloitte for potential Natural Language Processing work with the Centers for Disease Control and Prevention.

# Summary of Results

After analyzing and processing the data, I determined that a Sentence Embedding model was the most accurate at predicting document similarity. A TF-IDF model and a model utilizing Spacy's internal similarity tools were far less performant.

For the final output of this project, I have created two files:
* 'texts_processed.csv': a final csv file of all relevant information, including the columns of names, distinct names, dates and datetime objects.
* 'closest_matches.csv': a sampling of the top five articles in similarity to the sample article, using the Sentence Embedding method.

# Future Recommendations

With more time to devote to this project, there are several changes that could improve accuracy:
* Experimenting with various Spacy models, especially for vectorization/similarity testing.
* Training a custom model/class to detect document similarity utilizing the spacy or flair library.
* Utilizing Spark to increase compute power and decrease run time for text/date classification.

## Ingesting and Analyzing Dataset

The source dataset was provided by Deloitte, as a csv file of unstructured text data from several news articles.


```python
# Importing basic libraries
import pandas as pd
import numpy as np
import datetime

# Importing flair library for name recognition
from flair.data import Sentence, build_spacy_tokenizer
from flair.models import SequenceTagger

# Importing segtok segmenter
from segtok.segmenter import split_single

# Importing spacy for date recognition
import spacy
from spacy.tokenizer import Tokenizer

# Importing ctparse and timefhuman for converting date text into datetime objects
from ctparse import ctparse
from timefhuman import timefhuman

# Importing stopword, tokenizer and stemmer from nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

# Importing tensorflow hub for Sentence Encoder
import tensorflow as tf
import tensorflow_hub as hub
```

    /Users/eddiekirkland/anaconda3/lib/python3.7/site-packages/sklearn/base.py:318: UserWarning: Trying to unpickle estimator CountVectorizer from version 0.20.4 when using version 0.22.1. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)
    /Users/eddiekirkland/anaconda3/lib/python3.7/site-packages/sklearn/base.py:318: UserWarning: Trying to unpickle estimator MultinomialNB from version 0.20.4 when using version 0.22.1. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)
    /Users/eddiekirkland/anaconda3/lib/python3.7/site-packages/sklearn/base.py:318: UserWarning: Trying to unpickle estimator Pipeline from version 0.20.4 when using version 0.22.1. This might lead to breaking code or invalid results. Use at your own risk.
      UserWarning)



```python
# Increasing max column width to display text
pd.set_option('max_colwidth', 10000)
```


```python
# Reading data into dataframe
text_df = pd.read_csv('News-article-wikipedia-DFE.csv')
```


```python
text_df.info(verbose=True)
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3000 entries, 0 to 2999
    Data columns (total 14 columns):
     #   Column                                                                                                                                      Non-Null Count  Dtype  
    ---  ------                                                                                                                                      --------------  -----  
     0   _unit_id                                                                                                                                    3000 non-null   int64  
     1   _unit_state                                                                                                                                 3000 non-null   object 
     2   _trusted_judgments                                                                                                                          3000 non-null   int64  
     3   _last_judgment_at                                                                                                                           3000 non-null   object 
     4   given_the_news_article_headline_and_description_above_please_select_the_most_relevant_wikipedia_page_from_the_following_options             3000 non-null   object 
     5   given_the_news_article_headline_and_description_above_please_select_the_most_relevant_wikipedia_page_from_the_following_options:confidence  3000 non-null   float64
     6   wikipedia_page2__                                                                                                                           3000 non-null   object 
     7   article                                                                                                                                     3000 non-null   object 
     8   gurl                                                                                                                                        3000 non-null   object 
     9   id                                                                                                                                          3000 non-null   int64  
     10  newdescp                                                                                                                                    3000 non-null   object 
     11  nil                                                                                                                                         1 non-null      object 
     12  option3                                                                                                                                     3000 non-null   object 
     13  oururl                                                                                                                                      3000 non-null   object 
    dtypes: float64(1), int64(3), object(10)
    memory usage: 328.2+ KB


It looks as though there are no null values in the dataset, and 3000 entries for each feature.


```python
# Exploring dataframe
text_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>_unit_id</th>
      <th>_unit_state</th>
      <th>_trusted_judgments</th>
      <th>_last_judgment_at</th>
      <th>given_the_news_article_headline_and_description_above_please_select_the_most_relevant_wikipedia_page_from_the_following_options</th>
      <th>given_the_news_article_headline_and_description_above_please_select_the_most_relevant_wikipedia_page_from_the_following_options:confidence</th>
      <th>wikipedia_page2__</th>
      <th>article</th>
      <th>gurl</th>
      <th>id</th>
      <th>newdescp</th>
      <th>nil</th>
      <th>option3</th>
      <th>oururl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>691201838</td>
      <td>finalized</td>
      <td>5</td>
      <td>3/19/2015 19:59</td>
      <td>Wikipedia Page1 :</td>
      <td>1.0000</td>
      <td>Wikipedia Page2 :\n\n\nWikipedia Page2 :\nWikipedia Page2 :</td>
      <td>Gaza aid ship to dock in Egypt after Israel pressure</td>
      <td>http://en.wikipedia.org/wiki/Gaza_Strip</td>
      <td>1</td>
      <td>A ship with supplies for Gaza will dock at el-Arish in Egypt, officials say, after Israeli pressure to stop the vessel breaking its Gaza blockade. The Moldovan-flagged ship chartered by a charity run by the son of Libyan leader Col Muammar Gaddafi, left a Greek port on Saturday. Israel asked for help from the UN, and had talks with Greece and Moldova. But organisers insist they will go to Gaza. An Israeli raid on a Gaza-bound ship in May killed nine Turkish activists. Israel insisted its troops were defending themselves but the raid sparked international condemnation. Israel recently eased its blockade, allowing in almost all consumer goods but maintaining a "blacklist" of some items. Israel says its blockade of the Palestinian territory is needed to prevent the supply of weapons to the Hamas militant group which controls Gaza. The Amalthea, renamed Hope for the mission, set off from the Greek port of Lavrio, loaded with about 2,000 tonnes of food, cooking oil, medicines and pre-fabricated houses. It has been chartered by the Gaddafi International Charity and Development Foundation. Its chairman is Saif al-Islam Gaddafi. The organisation said the 92m (302ft) vessel would also carry "a number of supporters who are keen on expressing solidarity with the Palestinian people".</td>
      <td>NaN</td>
      <td>None</td>
      <td>http://en.wikipedia.org/wiki/Gaza_flotilla_raid</td>
    </tr>
    <tr>
      <th>1</th>
      <td>691201839</td>
      <td>finalized</td>
      <td>5</td>
      <td>3/19/2015 20:34</td>
      <td>Wikipedia Page1 :</td>
      <td>1.0000</td>
      <td>Wikipedia Page2 :\n\nWikipedia Page2 :\n\n</td>
      <td>Mel Gibson</td>
      <td>http://en.wikipedia.org/wiki/Mel_Gibson</td>
      <td>2</td>
      <td>Often acts and directs stories involving an individual who is persecuted, and fights for justice Has often portrayed a widower, in films such as Mad Max (the sequels), Lethal Weapon film series, Braveheart, The Patriot, Signs, and Edge of Darkness. Often portrays men who seek revenge for the murder of family or friends Ranked #12 in Empire (UK) magazine's "The Top 100 Movie Stars of All Time" list. [October 1997] Chosen by People (USA) magazine as one of the "50 Most Beautiful People" in the world. Educated at University of New South Wales, Australia. Chosen by People magazine as one of the "50 Most Beautiful People" in the world. Chosen by People magazine as one of the "50 Most Beautiful People" in the world. Awarded the AO (Officer of the Order of Australia), Australia's highest honor, in mid-1997. He took up acting only because his sister submitted an application behind his back. The night before an audition, he got into a fight, and his face was badly beaten, an accident that won him the role.</td>
      <td>NaN</td>
      <td>None</td>
      <td>http://en.wikipedia.org/wiki/Mel_Gibson_filmography</td>
    </tr>
    <tr>
      <th>2</th>
      <td>691201840</td>
      <td>finalized</td>
      <td>5</td>
      <td>3/19/2015 3:01</td>
      <td>Wikipedia Page1 :</td>
      <td>1.0000</td>
      <td>\n\n\n\n</td>
      <td>Talent Agency WME drops Mel Gibson</td>
      <td>http://en.wikipedia.org/wiki/Mel_Gibson</td>
      <td>3</td>
      <td>Cast member Mel Gibson (R) and Oksana Grigorieva attend the premiere of the film ''Edge of Darkness'' in Los Angeles January 26, 2010. Earlier this week, the agency's Patrick Whitesell informed the actors' representatives that he would no longer be represented by the agency. Gibson's longtime agent, Ed Limato, died July 3, and a funeral will take place in New York next week. William Morris Endeavor (WME) partner Ari Emanuel had previously expressed hostility toward Gibson after the actor made anti-Semitic remarks and made remarks implying skepticism about the Holocaust. An agency source said the only reason the agency had represented Gibson in the first place was his association with Limato. "Mel was really important to Ed," an agency source said. "He was with him for 32 years and I think Ed saw him as a son." But he added, "The world knows how Ari feels and he has never changed that opinion." Gibson's troubles have only increased in recent weeks with allegations of bigoted tirades and reports that he is under investigation for assaulting his ex-girlfriend. Several studio executives have said in the wake of these disclosures that they consider the troubled actor too untouchable in the industry. "I'd rather get engaged to Lindsay Lohan than have anything to do with him," one studio chief said. A spokesman for Gibson could not be reached for comment.</td>
      <td>NaN</td>
      <td>None</td>
      <td>http://en.wikipedia.org/wiki/Lethal_Weapon_(film_series)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>691201841</td>
      <td>finalized</td>
      <td>5</td>
      <td>3/20/2015 6:27</td>
      <td>Wikipedia Page1 :</td>
      <td>0.8282</td>
      <td>\n\n\nWikipedia Page2 :\n</td>
      <td>Suicide bomber killed in Tehran-Fars</td>
      <td>http://en.wikipedia.org/wiki/Iran</td>
      <td>4</td>
      <td>(Adds details)  TEHRAN, June 20 (Reuters) - A suicide bomber was killed and two people were wounded in Tehran on Saturday, near the shrine of Iran's revolutionary founder, Ayatollah Ruhollah Khomeini, Iran's semi-official Fars news agency reported.  "A suicide bomber was killed at the northern wing of Imam Khomeini's shrine. Two people were injured," Fars said.  It did not explain the exact circumstances.  Iranian riot police used teargas elsewhere in Tehran to disperse demonstrators protesting against a disputed presidential election, a witness said. (Editing by Jon Boyle)</td>
      <td>NaN</td>
      <td>None</td>
      <td>http://en.wikipedia.org/wiki/Timeline_of_the_2009_Iranian_election_protests</td>
    </tr>
    <tr>
      <th>4</th>
      <td>691201842</td>
      <td>finalized</td>
      <td>5</td>
      <td>3/20/2015 6:51</td>
      <td>Wikipedia Page1 :</td>
      <td>0.5258</td>
      <td>Wikipedia Page2 :\nWikipedia Page2 :\n\n\n</td>
      <td>Iran's 10% ballot boxes to be recounted</td>
      <td>http://en.wikipedia.org/wiki/Iran</td>
      <td>5</td>
      <td>Tehran - Iran's Guardian Council is ready to recount up to 10 percent of the ballot boxes randomly in last week's presidential election, state television reported on Saturday. "The Guardian Council is ready to recount randomly up to 10 percent of ballot boxes in last week's disputed presidential election," the council's spokesman Abbas Ali Kadkhodai was quoted as saying. "The Guardian Council is not legally obliged," Kadkhodai said, "we will recount the votes in the presence of the three (defeated) candidates." Whenever the examination and the recount is finished the council will announce its final decision, he added. He also said that Mir-Hossein Mousavi and Mehdi Karroubi still have time to express their opinions until Wednesday. Only Iran's former Revolutionary Guards Chief Mohsen Rezaei attended a special meeting of the Guardians Council with presidential candidates on Saturday, the official IRNA news agency reported. Iran's former Prime Minister Mir-Hossein Mousavi and former Parliament Speaker Mehdi Karroubi failed to attend the meeting without giving any reason. The Spokesman of the Guardian Council Abbas-Ali Kadkhodaei said on Wednesday that candidates of Iran's recent presidential election were invited to its upcoming meeting session which is to be held within the next few days. Iran's Supreme Leader Ayatollah Seyyed Ali Khamenei has ordered Iran's Guardian Council, the top legislative body, to investigate the claims of "fraud" in the recent presidential election.</td>
      <td>NaN</td>
      <td>None</td>
      <td>http://en.wikipedia.org/wiki/Timeline_of_the_2009_Iranian_election_protests</td>
    </tr>
  </tbody>
</table>
</div>



It seems as though much of the information in the dataset is not useful for the purpose of this project. I will simplify the dataframe through selecting relevant columns:
* 'article': the title of each article
* 'oururl': the url of the article's text
* 'newdescp': the basic text of each article


```python
# Keeping only relevant columns
keepcols = ['article', 'oururl', 'newdescp']
text_simple = text_df[keepcols]
```

# Part 1: Name and Date Recognition

## Name Recognition

To identify names within the text of each article, I will utilize two external libraries:
* Spacy: Used to tokenize the data into sentences. Selected for fast and accurate performance.
* Flair: Used to tokenize spans within each sentence. Selected for its highly intuitive understanding of the context of tokens and token spans.

After identifying a list of names, I will create an additional column containing only distinct names.

**Note:** I considered removing all single-word names such as "Ghadafi," since many are duplicates of the first/last names located within the article. However, simply removing these names could eliminate important information such as people referred to only by last name. For this reason, I left the list as-is.


```python
# Downloading named entity recognition model
tagger = SequenceTagger.load('ner')

# Creating spacy tokenizer for parsing sentences
nlp = spacy.load('en')
tokenizer = build_spacy_tokenizer(nlp)

# Defining function for tagging named entities in text
def find_names(text):
    # Creating empty list for name tokens
    names = []
    sentences = [Sentence(sent, use_tokenizer=tokenizer) for sent in split_single(text)]
    tagger.predict(sentences)
    for sent in sentences:
        for entity in sent.get_spans('ner'):
            if entity.tag=='PER':
                names.append(entity.text)
            else:
                pass
    return names
```

    2020-03-20 15:52:39,619 loading file /Users/eddiekirkland/.flair/models/en-ner-conll03-v0.4.pt



```python
# Applying function to text and creating column of names
text_simple['names'] = text_df['newdescp'].apply(find_names)
```

    2020-03-20 15:59:32,664 ACHTUNG: An empty Sentence was created! Are there empty strings in your dataset?
    2020-03-20 15:59:32,666 Ignore 1 sentence(s) with no tokens.
    2020-03-20 16:55:47,542 ACHTUNG: An empty Sentence was created! Are there empty strings in your dataset?
    2020-03-20 16:55:47,549 Ignore 1 sentence(s) with no tokens.
    2020-03-20 17:19:57,990 ACHTUNG: An empty Sentence was created! Are there empty strings in your dataset?
    2020-03-20 17:20:30,706 Ignore 1 sentence(s) with no tokens.
    2020-03-20 17:41:22,967 ACHTUNG: An empty Sentence was created! Are there empty strings in your dataset?
    2020-03-20 17:41:22,969 Ignore 1 sentence(s) with no tokens.
    2020-03-20 18:45:08,357 ACHTUNG: An empty Sentence was created! Are there empty strings in your dataset?
    2020-03-20 18:45:08,360 Ignore 1 sentence(s) with no tokens.
    2020-03-20 19:18:21,727 ACHTUNG: An empty Sentence was created! Are there empty strings in your dataset?
    2020-03-20 19:18:21,729 Ignore 1 sentence(s) with no tokens.
    2020-03-20 19:24:44,244 ACHTUNG: An empty Sentence was created! Are there empty strings in your dataset?
    2020-03-20 19:24:44,245 Ignore 1 sentence(s) with no tokens.
    2020-03-20 20:58:59,907 ACHTUNG: An empty Sentence was created! Are there empty strings in your dataset?
    2020-03-20 20:58:59,909 Ignore 1 sentence(s) with no tokens.
    2020-03-20 21:01:50,005 ACHTUNG: An empty Sentence was created! Are there empty strings in your dataset?
    2020-03-20 21:01:50,007 Ignore 1 sentence(s) with no tokens.
    2020-03-20 21:10:04,642 ACHTUNG: An empty Sentence was created! Are there empty strings in your dataset?
    2020-03-20 21:10:04,643 Ignore 1 sentence(s) with no tokens.
    2020-03-20 22:35:51,588 ACHTUNG: An empty Sentence was created! Are there empty strings in your dataset?
    2020-03-20 22:35:51,590 Ignore 1 sentence(s) with no tokens.
    2020-03-20 22:45:53,566 ACHTUNG: An empty Sentence was created! Are there empty strings in your dataset?
    2020-03-20 22:45:53,567 Ignore 1 sentence(s) with no tokens.
    2020-03-20 22:52:30,500 ACHTUNG: An empty Sentence was created! Are there empty strings in your dataset?
    2020-03-20 22:52:30,502 Ignore 1 sentence(s) with no tokens.


    /Users/eddiekirkland/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      



```python
# Saving analyzed results to file in case of needed restart
text_simple.to_csv('text_simple_names.csv')
```


```python
text_simple.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>article</th>
      <th>oururl</th>
      <th>newdescp</th>
      <th>names</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gaza aid ship to dock in Egypt after Israel pressure</td>
      <td>http://en.wikipedia.org/wiki/Gaza_flotilla_raid</td>
      <td>A ship with supplies for Gaza will dock at el-Arish in Egypt, officials say, after Israeli pressure to stop the vessel breaking its Gaza blockade. The Moldovan-flagged ship chartered by a charity run by the son of Libyan leader Col Muammar Gaddafi, left a Greek port on Saturday. Israel asked for help from the UN, and had talks with Greece and Moldova. But organisers insist they will go to Gaza. An Israeli raid on a Gaza-bound ship in May killed nine Turkish activists. Israel insisted its troops were defending themselves but the raid sparked international condemnation. Israel recently eased its blockade, allowing in almost all consumer goods but maintaining a "blacklist" of some items. Israel says its blockade of the Palestinian territory is needed to prevent the supply of weapons to the Hamas militant group which controls Gaza. The Amalthea, renamed Hope for the mission, set off from the Greek port of Lavrio, loaded with about 2,000 tonnes of food, cooking oil, medicines and pre-fabricated houses. It has been chartered by the Gaddafi International Charity and Development Foundation. Its chairman is Saif al-Islam Gaddafi. The organisation said the 92m (302ft) vessel would also carry "a number of supporters who are keen on expressing solidarity with the Palestinian people".</td>
      <td>[Col Muammar Gaddafi, Hope, Saif al, Islam Gaddafi]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mel Gibson</td>
      <td>http://en.wikipedia.org/wiki/Mel_Gibson_filmography</td>
      <td>Often acts and directs stories involving an individual who is persecuted, and fights for justice Has often portrayed a widower, in films such as Mad Max (the sequels), Lethal Weapon film series, Braveheart, The Patriot, Signs, and Edge of Darkness. Often portrays men who seek revenge for the murder of family or friends Ranked #12 in Empire (UK) magazine's "The Top 100 Movie Stars of All Time" list. [October 1997] Chosen by People (USA) magazine as one of the "50 Most Beautiful People" in the world. Educated at University of New South Wales, Australia. Chosen by People magazine as one of the "50 Most Beautiful People" in the world. Chosen by People magazine as one of the "50 Most Beautiful People" in the world. Awarded the AO (Officer of the Order of Australia), Australia's highest honor, in mid-1997. He took up acting only because his sister submitted an application behind his back. The night before an audition, he got into a fight, and his face was badly beaten, an accident that won him the role.</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Talent Agency WME drops Mel Gibson</td>
      <td>http://en.wikipedia.org/wiki/Lethal_Weapon_(film_series)</td>
      <td>Cast member Mel Gibson (R) and Oksana Grigorieva attend the premiere of the film ''Edge of Darkness'' in Los Angeles January 26, 2010. Earlier this week, the agency's Patrick Whitesell informed the actors' representatives that he would no longer be represented by the agency. Gibson's longtime agent, Ed Limato, died July 3, and a funeral will take place in New York next week. William Morris Endeavor (WME) partner Ari Emanuel had previously expressed hostility toward Gibson after the actor made anti-Semitic remarks and made remarks implying skepticism about the Holocaust. An agency source said the only reason the agency had represented Gibson in the first place was his association with Limato. "Mel was really important to Ed," an agency source said. "He was with him for 32 years and I think Ed saw him as a son." But he added, "The world knows how Ari feels and he has never changed that opinion." Gibson's troubles have only increased in recent weeks with allegations of bigoted tirades and reports that he is under investigation for assaulting his ex-girlfriend. Several studio executives have said in the wake of these disclosures that they consider the troubled actor too untouchable in the industry. "I'd rather get engaged to Lindsay Lohan than have anything to do with him," one studio chief said. A spokesman for Gibson could not be reached for comment.</td>
      <td>[Mel Gibson, Oksana Grigorieva, Patrick Whitesell, Gibson, Ed Limato, Ari Emanuel, Gibson, Gibson, Limato, Mel, Ed, Ed, Ari, Gibson, Lindsay Lohan, Gibson]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Suicide bomber killed in Tehran-Fars</td>
      <td>http://en.wikipedia.org/wiki/Timeline_of_the_2009_Iranian_election_protests</td>
      <td>(Adds details)  TEHRAN, June 20 (Reuters) - A suicide bomber was killed and two people were wounded in Tehran on Saturday, near the shrine of Iran's revolutionary founder, Ayatollah Ruhollah Khomeini, Iran's semi-official Fars news agency reported.  "A suicide bomber was killed at the northern wing of Imam Khomeini's shrine. Two people were injured," Fars said.  It did not explain the exact circumstances.  Iranian riot police used teargas elsewhere in Tehran to disperse demonstrators protesting against a disputed presidential election, a witness said. (Editing by Jon Boyle)</td>
      <td>[Ayatollah Ruhollah Khomeini, Imam Khomeini, Jon Boyle]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Iran's 10% ballot boxes to be recounted</td>
      <td>http://en.wikipedia.org/wiki/Timeline_of_the_2009_Iranian_election_protests</td>
      <td>Tehran - Iran's Guardian Council is ready to recount up to 10 percent of the ballot boxes randomly in last week's presidential election, state television reported on Saturday. "The Guardian Council is ready to recount randomly up to 10 percent of ballot boxes in last week's disputed presidential election," the council's spokesman Abbas Ali Kadkhodai was quoted as saying. "The Guardian Council is not legally obliged," Kadkhodai said, "we will recount the votes in the presence of the three (defeated) candidates." Whenever the examination and the recount is finished the council will announce its final decision, he added. He also said that Mir-Hossein Mousavi and Mehdi Karroubi still have time to express their opinions until Wednesday. Only Iran's former Revolutionary Guards Chief Mohsen Rezaei attended a special meeting of the Guardians Council with presidential candidates on Saturday, the official IRNA news agency reported. Iran's former Prime Minister Mir-Hossein Mousavi and former Parliament Speaker Mehdi Karroubi failed to attend the meeting without giving any reason. The Spokesman of the Guardian Council Abbas-Ali Kadkhodaei said on Wednesday that candidates of Iran's recent presidential election were invited to its upcoming meeting session which is to be held within the next few days. Iran's Supreme Leader Ayatollah Seyyed Ali Khamenei has ordered Iran's Guardian Council, the top legislative body, to investigate the claims of "fraud" in the recent presidential election.</td>
      <td>[Abbas Ali Kadkhodai, Kadkhodai, Mir, Hossein Mousavi, Mehdi Karroubi, Mohsen Rezaei, Mir, Hossein Mousavi, Mehdi Karroubi, Ali Kadkhodaei, Ayatollah Seyyed Ali Khamenei]</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Removing duplicates from name list
text_simple['names_distinct'] = text_simple['names'].apply(lambda x: list(set(x))).copy()
```

## Results

Overall, the results of the text recognition seem very reliable. This process took a significant amount of compute power and processing time. In the future, I would experiment with implementing this process on a Spark cluster to increase compute power.

## Date Recognition

Next, I will parse each article to remove any relevant date information. I will do this in two steps:
* Use Spacy's tagging function to identify any date-related words, placing those words in a list within a new column.
* Converting all date-related words in the column to datetime objects, if possible. This should help with future useability of the date information.


```python
# Defining function to detect dates in text
def date_detect(string):
    # Creating tagged doc from string
    doc = nlp(string)
    # Create empty list of dates
    date_list = []
    # Detecting date using spacy libraray
    for ent in doc.ents:
        if ent.label_=='DATE':
            # append date to list
            date_list.append(ent.text)
        else:
            pass
    return date_list
```


```python
# Applying function to text and creating column of dates
text_simple['dates'] = text_simple['newdescp'].apply(date_detect).copy()
```


```python
# Define function to apply timefhuman for readable timestamp objects
def timestamp(timelist):
    output = []
    for i in timelist:
        try:
            output.append(timefhuman(i))
        # Exception handling for non readable dates
        except Exception:
            pass
    # Removing empty items
    output = list(filter(None, output))
    return output
```


```python
# Applying function to text and creating column of datetime objects
text_simple['datetimes'] = text_simple['dates'].apply(timestamp).copy()
```


```python
text_simple.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>article</th>
      <th>oururl</th>
      <th>newdescp</th>
      <th>names</th>
      <th>names_distinct</th>
      <th>dates</th>
      <th>datetimes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Gaza aid ship to dock in Egypt after Israel pressure</td>
      <td>http://en.wikipedia.org/wiki/Gaza_flotilla_raid</td>
      <td>A ship with supplies for Gaza will dock at el-Arish in Egypt, officials say, after Israeli pressure to stop the vessel breaking its Gaza blockade. The Moldovan-flagged ship chartered by a charity run by the son of Libyan leader Col Muammar Gaddafi, left a Greek port on Saturday. Israel asked for help from the UN, and had talks with Greece and Moldova. But organisers insist they will go to Gaza. An Israeli raid on a Gaza-bound ship in May killed nine Turkish activists. Israel insisted its troops were defending themselves but the raid sparked international condemnation. Israel recently eased its blockade, allowing in almost all consumer goods but maintaining a "blacklist" of some items. Israel says its blockade of the Palestinian territory is needed to prevent the supply of weapons to the Hamas militant group which controls Gaza. The Amalthea, renamed Hope for the mission, set off from the Greek port of Lavrio, loaded with about 2,000 tonnes of food, cooking oil, medicines and pre-fabricated houses. It has been chartered by the Gaddafi International Charity and Development Foundation. Its chairman is Saif al-Islam Gaddafi. The organisation said the 92m (302ft) vessel would also carry "a number of supporters who are keen on expressing solidarity with the Palestinian people".</td>
      <td>['Col Muammar Gaddafi', 'Hope', 'Saif al', 'Islam Gaddafi']</td>
      <td>[a, I, ], S, l, M, C, m, f, i, o, u, r, d, s, p, ,,  , [, H, G, ', e]</td>
      <td>[Saturday, May]</td>
      <td>[2020-03-28 00:00:00]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Mel Gibson</td>
      <td>http://en.wikipedia.org/wiki/Mel_Gibson_filmography</td>
      <td>Often acts and directs stories involving an individual who is persecuted, and fights for justice Has often portrayed a widower, in films such as Mad Max (the sequels), Lethal Weapon film series, Braveheart, The Patriot, Signs, and Edge of Darkness. Often portrays men who seek revenge for the murder of family or friends Ranked #12 in Empire (UK) magazine's "The Top 100 Movie Stars of All Time" list. [October 1997] Chosen by People (USA) magazine as one of the "50 Most Beautiful People" in the world. Educated at University of New South Wales, Australia. Chosen by People magazine as one of the "50 Most Beautiful People" in the world. Chosen by People magazine as one of the "50 Most Beautiful People" in the world. Awarded the AO (Officer of the Order of Australia), Australia's highest honor, in mid-1997. He took up acting only because his sister submitted an application behind his back. The night before an audition, he got into a fight, and his face was badly beaten, an accident that won him the role.</td>
      <td>[]</td>
      <td>[], []</td>
      <td>[October 1997, mid-1997]</td>
      <td>[1997-10-01 00:00:00]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Talent Agency WME drops Mel Gibson</td>
      <td>http://en.wikipedia.org/wiki/Lethal_Weapon_(film_series)</td>
      <td>Cast member Mel Gibson (R) and Oksana Grigorieva attend the premiere of the film ''Edge of Darkness'' in Los Angeles January 26, 2010. Earlier this week, the agency's Patrick Whitesell informed the actors' representatives that he would no longer be represented by the agency. Gibson's longtime agent, Ed Limato, died July 3, and a funeral will take place in New York next week. William Morris Endeavor (WME) partner Ari Emanuel had previously expressed hostility toward Gibson after the actor made anti-Semitic remarks and made remarks implying skepticism about the Holocaust. An agency source said the only reason the agency had represented Gibson in the first place was his association with Limato. "Mel was really important to Ed," an agency source said. "He was with him for 32 years and I think Ed saw him as a son." But he added, "The world knows how Ari feels and he has never changed that opinion." Gibson's troubles have only increased in recent weeks with allegations of bigoted tirades and reports that he is under investigation for assaulting his ex-girlfriend. Several studio executives have said in the wake of these disclosures that they consider the troubled actor too untouchable in the industry. "I'd rather get engaged to Lindsay Lohan than have anything to do with him," one studio chief said. A spokesman for Gibson could not be reached for comment.</td>
      <td>['Mel Gibson', 'Oksana Grigorieva', 'Patrick Whitesell', 'Gibson', 'Ed Limato', 'Ari Emanuel', 'Gibson', 'Gibson', 'Limato', 'Mel', 'Ed', 'Ed', 'Ari', 'Gibson', 'Lindsay Lohan', 'Gibson']</td>
      <td>[a, v, P, ], L, l, g, M, m, k, i, W, t, o, b, u, r, d, c, y, s, ,, h, O,  , [, G, n, E, ', e, A]</td>
      <td>[Earlier this week, July 3, next week, 32 years, recent weeks]</td>
      <td>[2020-07-03 00:00:00]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Suicide bomber killed in Tehran-Fars</td>
      <td>http://en.wikipedia.org/wiki/Timeline_of_the_2009_Iranian_election_protests</td>
      <td>(Adds details)  TEHRAN, June 20 (Reuters) - A suicide bomber was killed and two people were wounded in Tehran on Saturday, near the shrine of Iran's revolutionary founder, Ayatollah Ruhollah Khomeini, Iran's semi-official Fars news agency reported.  "A suicide bomber was killed at the northern wing of Imam Khomeini's shrine. Two people were injured," Fars said.  It did not explain the exact circumstances.  Iranian riot police used teargas elsewhere in Tehran to disperse demonstrators protesting against a disputed presidential election, a witness said. (Editing by Jon Boyle)</td>
      <td>['Ayatollah Ruhollah Khomeini', 'Imam Khomeini', 'Jon Boyle']</td>
      <td>[a, I, ], l, m, R, i, K, t, o, u, y, ,, J, h, B,  , [, n, ', e, A]</td>
      <td>[June 20, Saturday]</td>
      <td>[2020-06-20 00:00:00, 2020-03-28 00:00:00]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Iran's 10% ballot boxes to be recounted</td>
      <td>http://en.wikipedia.org/wiki/Timeline_of_the_2009_Iranian_election_protests</td>
      <td>Tehran - Iran's Guardian Council is ready to recount up to 10 percent of the ballot boxes randomly in last week's presidential election, state television reported on Saturday. "The Guardian Council is ready to recount randomly up to 10 percent of ballot boxes in last week's disputed presidential election," the council's spokesman Abbas Ali Kadkhodai was quoted as saying. "The Guardian Council is not legally obliged," Kadkhodai said, "we will recount the votes in the presence of the three (defeated) candidates." Whenever the examination and the recount is finished the council will announce its final decision, he added. He also said that Mir-Hossein Mousavi and Mehdi Karroubi still have time to express their opinions until Wednesday. Only Iran's former Revolutionary Guards Chief Mohsen Rezaei attended a special meeting of the Guardians Council with presidential candidates on Saturday, the official IRNA news agency reported. Iran's former Prime Minister Mir-Hossein Mousavi and former Parliament Speaker Mehdi Karroubi failed to attend the meeting without giving any reason. The Spokesman of the Guardian Council Abbas-Ali Kadkhodaei said on Wednesday that candidates of Iran's recent presidential election were invited to its upcoming meeting session which is to be held within the next few days. Iran's Supreme Leader Ayatollah Seyyed Ali Khamenei has ordered Iran's Guardian Council, the top legislative body, to investigate the claims of "fraud" in the recent presidential election.</td>
      <td>['Abbas Ali Kadkhodai', 'Kadkhodai', 'Mir', 'Hossein Mousavi', 'Mehdi Karroubi', 'Mohsen Rezaei', 'Mir', 'Hossein Mousavi', 'Mehdi Karroubi', 'Ali Kadkhodaei', 'Ayatollah Seyyed Ali Khamenei']</td>
      <td>[a, v, ], S, l, M, m, k, R, i, K, t, o, b, u, r, d, y, s, ,, h,  , [, H, n, ', e, A, z]</td>
      <td>[last week's, Saturday, last week's, Wednesday, Saturday, Wednesday, the next few days]</td>
      <td>[2020-03-28 00:00:00, 2020-03-25 00:00:00, 2020-03-28 00:00:00, 2020-03-25 00:00:00]</td>
    </tr>
  </tbody>
</table>
</div>



## Results

After the data processing, we have several new columns of relevant data:
* 'names': a list containing all personal names detected in the article
* 'names_distinct': a list of all distinct names detected
* 'dates': a list of all date-type strings detected
* 'datetimes': a list of datetime objects for any detectable timestamps

**Note:** The datetime objects could be more accurate if each article had a corresponding date of publication. This could allow the library to index those dates such as "last week" or "next Tuesday" to specific timestamps. From the given data, we might have been able to use the `'last_judgement_at'` column, but without documentation we cannot tell if this is an appopriate reference date to use.

# Part 2: Check for Document Similarity

For part two of this assessment, we will scan the news articles to find those most similar to a provided sample. To do this, we will try a few machine learning techniques:
* "Term Frequency - Inverse Document Freqency" (TF-IDF) with a cosine similarity test
* Sentence Emedding with Google Sentence Encoder
* Spacy Vectorization - using Spacy's internal vectorization and similariy function

## Preprocessing
To begin, we need to preprocess the data in order to:
* Lowercase all words
* Remove stop words
* Remove puntuation
* Remove single characters (unneeeded for our analysis)
* Stem each word
* Lemmatize each word
* Convert numbers to words

### Creating Preprocessing Functions


```python
# Function to lowercase all words
def convert_lower_case(text):
    return np.char.lower(text)
```


```python
# Function to remove stop words
def remove_stop_words(text):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(text))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text
```


```python
# Function to remove punctuation
def remove_punctuation(text):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        text = np.char.replace(text, symbols[i], ' ')
        text = np.char.replace(text, " ", " ")
    text = np.char.replace(text, ',', '')
    return text
```


```python
# Function to remove apostrophes
# We call this function separately to avoid erros in possessive words
def remove_apostrophe(text):
    return np.char.replace(text, "'", '')
```


```python
# Function to stem words
def stemming(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(str(text))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text
```


```python
# Function to convert numbers to text
def convert_numbers(text):
    tokens = word_tokenize(str(text))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text
```


```python
# Function to preprocess data using previously defined functions
def preprocess(text):
    text = convert_lower_case(text)
    text = remove_punctuation(text)
    text = remove_apostrophe(text)
    text = remove_stop_words(text)
    text = convert_numbers(text)
    text = stemming(text)
    text = remove_punctuation(text)
    text = convert_numbers(text)
    text = stemming(text)
    text = remove_punctuation(text) # needed again as num2word adds some hyphens and commas
    text = remove_stop_words(text) # needed again as num2word adds some stop words
    return text
```


```python
# Applying preprocessing function to all articles
text_simple['processed_text'] = text_simple['newdescp'].apply(preprocess).copy()
```


```python
text_simple.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>article</th>
      <th>oururl</th>
      <th>newdescp</th>
      <th>names</th>
      <th>dates</th>
      <th>datetimes</th>
      <th>names_distinct</th>
      <th>processed_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gaza aid ship to dock in Egypt after Israel pressure</td>
      <td>http://en.wikipedia.org/wiki/Gaza_flotilla_raid</td>
      <td>A ship with supplies for Gaza will dock at el-Arish in Egypt, officials say, after Israeli pressure to stop the vessel breaking its Gaza blockade. The Moldovan-flagged ship chartered by a charity run by the son of Libyan leader Col Muammar Gaddafi, left a Greek port on Saturday. Israel asked for help from the UN, and had talks with Greece and Moldova. But organisers insist they will go to Gaza. An Israeli raid on a Gaza-bound ship in May killed nine Turkish activists. Israel insisted its troops were defending themselves but the raid sparked international condemnation. Israel recently eased its blockade, allowing in almost all consumer goods but maintaining a "blacklist" of some items. Israel says its blockade of the Palestinian territory is needed to prevent the supply of weapons to the Hamas militant group which controls Gaza. The Amalthea, renamed Hope for the mission, set off from the Greek port of Lavrio, loaded with about 2,000 tonnes of food, cooking oil, medicines and pre-fabricated houses. It has been chartered by the Gaddafi International Charity and Development Foundation. Its chairman is Saif al-Islam Gaddafi. The organisation said the 92m (302ft) vessel would also carry "a number of supporters who are keen on expressing solidarity with the Palestinian people".</td>
      <td>[Col Muammar Gaddafi, Hope, Saif al, Islam Gaddafi]</td>
      <td>[Saturday, May]</td>
      <td>[2020-03-21 00:00:00]</td>
      <td>[Col Muammar Gaddafi, Hope, Islam Gaddafi, Saif al]</td>
      <td>ship suppli gaza dock el arish egypt offici say isra pressur stop vessel break gaza blockad moldovan flag ship charter chariti run son libyan leader col muammar gaddafi left greek port saturday israel ask help un talk greec moldova organi insist go gaza isra raid gaza bound ship may kill nine turkish activist israel insist troop defend raid spark intern condemn israel recent ea blockad allow almost consum good maintain blacklist item israel say blockad palestinian territori need prevent suppli weapon hama milit group control gaza amalthea renam hope mission set greek port lavrio load two thousand tonn food cook oil medicin pre fabric hou charter gaddafi intern chariti develop foundat chairman saif al islam gaddafi organi said 92m 302ft vessel would also carri number support keen express solidar palestinian peopl</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Mel Gibson</td>
      <td>http://en.wikipedia.org/wiki/Mel_Gibson_filmography</td>
      <td>Often acts and directs stories involving an individual who is persecuted, and fights for justice Has often portrayed a widower, in films such as Mad Max (the sequels), Lethal Weapon film series, Braveheart, The Patriot, Signs, and Edge of Darkness. Often portrays men who seek revenge for the murder of family or friends Ranked #12 in Empire (UK) magazine's "The Top 100 Movie Stars of All Time" list. [October 1997] Chosen by People (USA) magazine as one of the "50 Most Beautiful People" in the world. Educated at University of New South Wales, Australia. Chosen by People magazine as one of the "50 Most Beautiful People" in the world. Chosen by People magazine as one of the "50 Most Beautiful People" in the world. Awarded the AO (Officer of the Order of Australia), Australia's highest honor, in mid-1997. He took up acting only because his sister submitted an application behind his back. The night before an audition, he got into a fight, and his face was badly beaten, an accident that won him the role.</td>
      <td>[]</td>
      <td>[October 1997, mid-1997]</td>
      <td>[1997-10-01 00:00:00]</td>
      <td>[]</td>
      <td>often act direct stori involv individu persecut fight justic often portray widow film mad max sequel lethal weapon film seri braveheart patriot sign edg dark often portray men seek reveng murder famili friend rank twelv empir uk magazin top one hundr movi star time list octob one thousand nine hundr nineti seven chosen peopl usa magazin one fifti beauti peopl world educ univ new south wale australia chosen peopl magazin one fifti beauti peopl world chosen peopl magazin one fifti beauti peopl world award ao offic order australia australia highest honor mid one thousand nine hundr nineti seven took act sister submit applic behind back night audit got fight face badli beaten accid role</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Talent Agency WME drops Mel Gibson</td>
      <td>http://en.wikipedia.org/wiki/Lethal_Weapon_(film_series)</td>
      <td>Cast member Mel Gibson (R) and Oksana Grigorieva attend the premiere of the film ''Edge of Darkness'' in Los Angeles January 26, 2010. Earlier this week, the agency's Patrick Whitesell informed the actors' representatives that he would no longer be represented by the agency. Gibson's longtime agent, Ed Limato, died July 3, and a funeral will take place in New York next week. William Morris Endeavor (WME) partner Ari Emanuel had previously expressed hostility toward Gibson after the actor made anti-Semitic remarks and made remarks implying skepticism about the Holocaust. An agency source said the only reason the agency had represented Gibson in the first place was his association with Limato. "Mel was really important to Ed," an agency source said. "He was with him for 32 years and I think Ed saw him as a son." But he added, "The world knows how Ari feels and he has never changed that opinion." Gibson's troubles have only increased in recent weeks with allegations of bigoted tirades and reports that he is under investigation for assaulting his ex-girlfriend. Several studio executives have said in the wake of these disclosures that they consider the troubled actor too untouchable in the industry. "I'd rather get engaged to Lindsay Lohan than have anything to do with him," one studio chief said. A spokesman for Gibson could not be reached for comment.</td>
      <td>[Mel Gibson, Oksana Grigorieva, Patrick Whitesell, Gibson, Ed Limato, Ari Emanuel, Gibson, Gibson, Limato, Mel, Ed, Ed, Ari, Gibson, Lindsay Lohan, Gibson]</td>
      <td>[Earlier this week, July 3, next week, 32 years, recent weeks]</td>
      <td>[2020-07-03 00:00:00]</td>
      <td>[Patrick Whitesell, Ed Limato, Ari, Mel Gibson, Oksana Grigorieva, Limato, Ed, Ari Emanuel, Gibson, Mel, Lindsay Lohan]</td>
      <td>cast member mel gibson oksana grigorieva attend premier film edg dark lo angel januari twenti six two thousand ten earlier week agenc patrick whitesel inform actor repr would longer repr agenc gibson longtim agent ed limato die juli funer take place new york next week william morri endeavor wme partner ari emanuel previou express hostil toward gibson actor made anti semit remark made remark impli skeptic holocaust agenc sourc said reason agenc repr gibson first place associ limato mel realli import ed agenc sourc said thirti two year think ed saw son ad world know ari feel never chang opinion gibson troubl increa recent week alleg bigot tirad report investig assault ex girlfriend sever studio execut said wake disclosur consid troubl actor untouch industri id rather get engag lindsay lohan anyth one studio chief said spokesman gibson could reach comment</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Suicide bomber killed in Tehran-Fars</td>
      <td>http://en.wikipedia.org/wiki/Timeline_of_the_2009_Iranian_election_protests</td>
      <td>(Adds details)  TEHRAN, June 20 (Reuters) - A suicide bomber was killed and two people were wounded in Tehran on Saturday, near the shrine of Iran's revolutionary founder, Ayatollah Ruhollah Khomeini, Iran's semi-official Fars news agency reported.  "A suicide bomber was killed at the northern wing of Imam Khomeini's shrine. Two people were injured," Fars said.  It did not explain the exact circumstances.  Iranian riot police used teargas elsewhere in Tehran to disperse demonstrators protesting against a disputed presidential election, a witness said. (Editing by Jon Boyle)</td>
      <td>[Ayatollah Ruhollah Khomeini, Imam Khomeini, Jon Boyle]</td>
      <td>[June 20, Saturday]</td>
      <td>[2020-06-20 00:00:00, 2020-03-21 00:00:00]</td>
      <td>[Jon Boyle, Ayatollah Ruhollah Khomeini, Imam Khomeini]</td>
      <td>add detail tehran june twenti reuter suicid bomber kill two peopl wound tehran saturday near shrine iran revolutionari founder ayatollah ruhollah khomeini iran semi offici far news agenc report suicid bomber kill northern wing imam khomeini shrine two peopl injur far said explain exact circumst iranian riot polic use tearga elsewh tehran disper demonstr protest disput presidenti elect wit said edit jon boyl</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Iran's 10% ballot boxes to be recounted</td>
      <td>http://en.wikipedia.org/wiki/Timeline_of_the_2009_Iranian_election_protests</td>
      <td>Tehran - Iran's Guardian Council is ready to recount up to 10 percent of the ballot boxes randomly in last week's presidential election, state television reported on Saturday. "The Guardian Council is ready to recount randomly up to 10 percent of ballot boxes in last week's disputed presidential election," the council's spokesman Abbas Ali Kadkhodai was quoted as saying. "The Guardian Council is not legally obliged," Kadkhodai said, "we will recount the votes in the presence of the three (defeated) candidates." Whenever the examination and the recount is finished the council will announce its final decision, he added. He also said that Mir-Hossein Mousavi and Mehdi Karroubi still have time to express their opinions until Wednesday. Only Iran's former Revolutionary Guards Chief Mohsen Rezaei attended a special meeting of the Guardians Council with presidential candidates on Saturday, the official IRNA news agency reported. Iran's former Prime Minister Mir-Hossein Mousavi and former Parliament Speaker Mehdi Karroubi failed to attend the meeting without giving any reason. The Spokesman of the Guardian Council Abbas-Ali Kadkhodaei said on Wednesday that candidates of Iran's recent presidential election were invited to its upcoming meeting session which is to be held within the next few days. Iran's Supreme Leader Ayatollah Seyyed Ali Khamenei has ordered Iran's Guardian Council, the top legislative body, to investigate the claims of "fraud" in the recent presidential election.</td>
      <td>[Abbas Ali Kadkhodai, Kadkhodai, Mir, Hossein Mousavi, Mehdi Karroubi, Mohsen Rezaei, Mir, Hossein Mousavi, Mehdi Karroubi, Ali Kadkhodaei, Ayatollah Seyyed Ali Khamenei]</td>
      <td>[last week's, Saturday, last week's, Wednesday, Saturday, Wednesday, the next few days]</td>
      <td>[2020-03-21 00:00:00, 2020-03-25 00:00:00, 2020-03-21 00:00:00, 2020-03-25 00:00:00]</td>
      <td>[Hossein Mousavi, Kadkhodai, Abbas Ali Kadkhodai, Mehdi Karroubi, Mir, Mohsen Rezaei, Ayatollah Seyyed Ali Khamenei, Ali Kadkhodaei]</td>
      <td>tehran iran guardian council readi recount ten percent ballot box randomli last week presidenti elect state televi report saturday guardian council readi recount randomli ten percent ballot box last week disput presidenti elect council spokesman abba ali kadkhodai quot say guardian council legal oblig kadkhodai said recount vote presenc three defeat candid whenev examin recount finish council announc final deci ad also said mir hossein mousavi mehdi karroubi still time express opinion wednesday iran former revolutionari guard chief mohsen rezaei attend special meet guardian council presidenti candid saturday offici irna news agenc report iran former prime minist mir hossein mousavi former parliament speaker mehdi karroubi fail attend meet without give reason spokesman guardian council abba ali kadkhodaei said wednesday candid iran recent presidenti elect invit upcom meet session held within next day iran suprem leader ayatollah seyi ali khamenei order iran guardian council top legisl bodi investig claim fraud recent presidenti elect</td>
    </tr>
  </tbody>
</table>
</div>



### Treating Sample Text


```python
# Importing sample text
sample = '''
         Human Rights Watch says government-controlled health services in Egypt have been pressured into playing down the number of casualties during anti-government protests. The group has documented the deaths of 297 people, but says the final toll is likely to be significantly higher. Human Rights Watch says the vast majority of the deaths in Cairo, Alexandria and Suez were on January 28 and 29 as a result of live gunfire as riot police fought running battles with protesters. A significant proportion came as a result of rubber bullets fired at too close a range and from teargas canisters fired into the crowds at very close range. Human Rights Watch says the actual number of deaths is likely to be an underestimate because the organisation had only included those deaths it had verified itself at key hospitals in the three major cities.
         '''
# Preprocessing sample to match dataset
sample = preprocess(sample)
```


```python
sample
```




    ' human right watch say govern control health servic egypt pressur play number casualti anti govern protest group document death two hundr nineti seven peopl say final toll like significantli higher human right watch say vast major death cairo alexandria suez januari twenti eight twenti nine result live gunfir riot polic fought run battl protest signif proport came result rubber bullet fire close rang tearga canist fire crowd close rang human right watch say actual number death like underestim organi includ death verifi key hospit three major citi'



## Test 1: TF-IDF Cosine Similarity


```python
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```


```python
# defining vectorization function for texts
def get_vectors(strings):
    text = [t for t in strings]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()
```


```python
# defining cosine similarity function
def get_cosine_sim(strings):
    vectors = [t for t in get_vectors(strings)]
    return cosine_similarity(vectors)[1,0]
```


```python
# defining function to combine sample with each item
def cosine_test(item):
    sample = '''
         Human Rights Watch says government-controlled health services in Egypt have been pressured into playing down the number of casualties during anti-government protests. The group has documented the deaths of 297 people, but says the final toll is likely to be significantly higher. Human Rights Watch says the vast majority of the deaths in Cairo, Alexandria and Suez were on January 28 and 29 as a result of live gunfire as riot police fought running battles with protesters. A significant proportion came as a result of rubber bullets fired at too close a range and from teargas canisters fired into the crowds at very close range. Human Rights Watch says the actual number of deaths is likely to be an underestimate because the organisation had only included those deaths it had verified itself at key hospitals in the three major cities.
         '''
    test_items = [item, sample]
    return get_cosine_sim(test_items)
```


```python
# Applying cosine test to all articles
text_simple['cosine_test'] = text_simple['processed_text'].apply(cosine_test).copy()
```


```python
text_simple = text_simple.sort_values(by='cosine_test', ascending=False)
```


```python
text_simple.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>article</th>
      <th>oururl</th>
      <th>newdescp</th>
      <th>names</th>
      <th>dates</th>
      <th>datetimes</th>
      <th>names_distinct</th>
      <th>processed_text</th>
      <th>cosine_test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>76</th>
      <td>Egypt hospitals 'told to downplay protest deaths'</td>
      <td>http://en.wikipedia.org/wiki/Egyptian_Revolution_of_2011</td>
      <td>Human Rights Watch says government-controlled health services in Egypt have been pressured into playing down the number of casualties during anti-government protests. The group has documented the deaths of 297 people, but says the final toll is likely to be significantly higher. Human Rights Watch says the vast majority of the deaths in Cairo, Alexandria and Suez were on January 28 and 29 as a result of live gunfire as riot police fought running battles with protesters. A significant proportion came as a result of rubber bullets fired at too close a range and from teargas canisters fired into the crowds at very close range. Human Rights Watch says the actual number of deaths is likely to be an underestimate because the organisation had only included those deaths it had verified itself at key hospitals in the three major cities.</td>
      <td>[]</td>
      <td>[January 28 and 29]</td>
      <td>[]</td>
      <td>[]</td>
      <td>human right watch say govern control health servic egypt pressur play number casualti anti govern protest group document death two hundr nineti seven peopl say final toll like significantli higher human right watch say vast major death cairo alexandria suez januari twenti eight twenti nine result live gunfir riot polic fought run battl protest signif proport came result rubber bullet fire close rang tearga canist fire crowd close rang human right watch say actual number death like underestim organi includ death verifi key hospit three major citi</td>
      <td>0.223969</td>
    </tr>
    <tr>
      <th>2361</th>
      <td>Libya: Security Forces Kill 84 Over Three Days</td>
      <td>http://en.wikipedia.org/wiki/Libyan_Civil_War_(2011)</td>
      <td>Muammar Gaddafi's security forces are firing on Libyan citizens and killing scores simply because they're demanding change and accountability. Libyan authorities should allow peaceful protesters to have their say. (New York) - Government security forces have killed at least 84 people in three days of protests in several cities in Libya, Human Rights Watch said today, based on telephone interviews with local hospital staff and witnesses. The Libyan authorities should immediately end attacks on peaceful protesters and protect them from assault by pro-government armed groups, Human Rights Watch said. Thousands of demonstrators gathered in the eastern Libyan cities of Benghazi, Baida, Ajdabiya, Zawiya, and Derna on February 18, 2011, following violent attacks against peaceful protests the day before that killed 20 people in Benghazi, 23 in Baida, three in Ajdabiya, and three in Derna. Hospital sources told Human Rights Watch that security forces killed 35 people in Benghazi on February 18, almost all with live ammunition. "Muammar Gaddafi's security forces are firing on Libyan citizens and killing scores simply because they're demanding change and accountability," said Joe Stork, deputy Middle East and North Africa director at Human Rights Watch. "Libyan authorities should allow peaceful protesters to have their say." The protests in Benghazi on February 18 began during funerals for the 20 demonstrators killed by security forces the day before. Eyewitnesses told Human Rights Watch that security forces with distinctive yellow uniforms opened fire on protesters near the Fadil Bu Omar Katiba, a security force base in the center of Benghazi. One protester told Human Rights Watch he witnessed four men shot dead. By 11 p.m. on February 18, Al Jalaa Hospital in Benghazi had received the bodies of 35 people killed that day, a senior hospital official told Human Rights Watch. He said the deaths had been caused by gunshot wounds to the chest, neck, and head. Two sources at the hospital confirmed to Human Rights Watch that the death toll for February 17 was 20, and that at least 45 people had been wounded by bullets. The senior hospital official told Human Rights Watch, "We put out a call to all the doctors in Benghazi to come to the hospital and for everyone to contribute blood because I've never seen anything like this before." Witnesses said that after the February 18 shootings, protesters in Benghazi continued on to the courthouse and gathered there throughout the evening, the crowd swelling to thousands. In Baida, further to the east, protesters on February 18 buried the 23 people who had been shot dead the day before. One protester told Human Rights Watch that police were patrolling the streets but he had seen no further clashes.</td>
      <td>[Muammar Gaddafi, Muammar Gaddafi, Joe Stork]</td>
      <td>[three days, today, February 18, 2011, the day, February 18, February 18, the day before, February 18, that day, February 17, 20, February 18, February 18, the day before]</td>
      <td>[2020-03-21 00:00:00, 2011-02-18 00:00:00, 2020-02-18 00:00:00, 2020-02-18 00:00:00, 2020-02-18 00:00:00, 2020-02-17 00:00:00, 2020-03-21 20:00:00, 2020-02-18 00:00:00, 2020-02-18 00:00:00]</td>
      <td>[Muammar Gaddafi, Joe Stork]</td>
      <td>muammar gaddafi secur forc fire libyan citizen kill score simpli theyr demand chang account libyan author allow peac protest say new york govern secur forc kill least eighti four peopl three day protest sever citi libya human right watch said today base telephon interview local hospit staff wit libyan author immedi end attack peac protest protect assault pro govern arm group human right watch said thousand demonstr gather eastern libyan citi benghazi baida ajdabiya zawiya derna februari eighteen two thousand eleven follow violent attack peac protest day kill twenti peopl benghazi twenti three baida three ajdabiya three derna hospit sourc told human right watch secur forc kill thirti five peopl benghazi februari eighteen almost live ammunit muammar gaddafi secur forc fire libyan citizen kill score simpli theyr demand chang account said joe stork deputi middl east north africa director human right watch libyan author allow peac protest say protest benghazi februari eighteen began funer twenti demonstr kill secur forc day eyewit told human right watch secur forc distinct yellow uniform open fire protest near fadil bu omar katiba secur forc base center benghazi one protest told human right watch wit four men shot dead eleven februari eighteen al jalaa hospit benghazi receiv bodi thirti five peopl kill day senior hospit offici told human right watch said death cau gunshot wound chest neck head two sourc hospit confirm human right watch death toll februari seventeen twenti least forti five peopl wound bullet senior hospit offici told human right watch put call doctor benghazi come hospit everyon contribut blood ive never seen anyth like wit said februari eighteen shoot protest benghazi continu courthou gather throughout even crowd swell thousand baida east protest februari eighteen buri twenti three peopl shot dead day one protest told human right watch polic patrol street seen clash</td>
      <td>0.102610</td>
    </tr>
    <tr>
      <th>2624</th>
      <td>Libya: Security Forces Fire on 'Day of Anger' Demonstrations</td>
      <td>http://en.wikipedia.org/wiki/Libyan_Civil_War_(2011)</td>
      <td>The security forces' vicious attacks on peaceful demonstrators lay bare the reality of Muammar Gaddafi's brutality when faced with any internal dissent. Libyans should not have to risk their lives to make a stand for their rights as human beings. (New York) - The Libyan security forces killed at least 24 protesters and wounded many others in a crackdown on peaceful demonstrations across the country, Human Rights Watch said today. The authorities should cease the use of lethal force unless absolutely necessary to protect lives and open an independent investigation into the lethal shootings, Human Rights Watch said. Hundreds of peaceful protesters took to the streets on February 17, 2011, in Baida, Benghazy, Zenten, Derna, and Ajdabiya. According to multiple witnesses, Libyan security forces shot and killed the demonstrators in efforts to disperse the protests. "The security forces' vicious attacks on peaceful demonstrators lay bare the reality of Muammar Gaddafi's brutality when faced with any internal dissent," said Sarah Leah Whitson, Middle East and North Africa director at Human Rights Watch. "Libyans should not have to risk their lives to make a stand for their rights as human beings." Some of the worst violence was in the eastern city of Baida. At around 1 p.m. on February 17, according to sources in Libya, hospital staff put out a call for additional medical supplies, as they became overwhelmed with the medical needs of 70 injured protesters, half of them in critical condition due to gunshot wounds. On the night of February 16, security forces had attacked peaceful protesters with teargas and live ammunition, shooting dead two protesters, according to protesters who spoke to Human Rights Watch. Geneva-based Libya Human Rights Solidarity has confirmed three of the names of those shot dead so far: Safwan Attiya, Nasser Al Juweigi, and Ahmad El Qabili. One protester told Human Rights Watch that a new protest started on February 17, after noon prayers and the funerals of those killed on February 16. Joined by hundreds of other protesters, families marched toward the Internal Security office, chanting, "Down with the regime" and "Get out Muammar Gaddafi." Some protesters filmed the protests with mobile phones and posted them online. One injured protester in a hospital in Baida told Human Rights Watch that he was sitting near the intensive care unit there and had confirmed that security forces had shot dead 16 people and wounded dozens of others. He said that Special Forces and armed men in street clothes fired live ammunition to deter protesters. A protester in Benghazi told Human Rights Watch that hundreds of lawyers, activists, and other protesters gathered on the steps of the Benghazi Court calling for a constitution and respect for the rule of law. Early in the day, sources in Libya told Human Rights Watch that security forces had arrested a Benghazi journalist, Hind El Houny, and Salem Souidan, a family member of a group that has been seeking justice for the massacre of inmates in Abu Salim prison in 1996. Security forces also arrested a former political prisoner, Abdel Nasser al-Rabbasi, in Bani Walid. The protester said he saw groups of men in street clothes armed with knives, later joined by Internal Security forces, charging the protesters to disperse them. The protester told Human Rights Watch that he believed security forces had shot dead at least 17 protesters during the day, mostly near Abdel Nasser Street. Human Rights Watch was able to confirm eight of those deaths. It appears that the government also has coordinated pro-government supporters to confront the demonstrations. On February 16, subscribers to Libyana, one of two Libyan mobile phone networks, received a text message calling upon "nationalist youth" to go out and "defend national symbols." At around 11:30 p.m. on February 17, a protester in Tripoli told Human Rights Watch that anti-government protests had started in Tripoli also.</td>
      <td>[Muammar Gaddafi, Muammar Gaddafi, Sarah Leah Whitson, Safwan Attiya, Nasser Al Juweigi, Ahmad El Qabili, Muammar Gaddafi, Hind El Houny, Salem Souidan, Abdel Nasser al, Rabbasi]</td>
      <td>[today, February 17, 2011, February 17, February 17, February 16, the day, 1996, the day, February 16, February 17]</td>
      <td>[2020-03-21 00:00:00, 2011-02-17 00:00:00, 2020-02-17 00:00:00, 2020-02-17 00:00:00, 2020-02-16 00:00:00, 2020-02-16 00:00:00, 2020-02-17 00:00:00]</td>
      <td>[Ahmad El Qabili, Rabbasi, Nasser Al Juweigi, Sarah Leah Whitson, Hind El Houny, Abdel Nasser al, Salem Souidan, Safwan Attiya, Muammar Gaddafi]</td>
      <td>secur forc viciou attack peac demonstr lay bare realiti muammar gaddafi brutal face intern dissent libyan risk live make stand right human new york libyan secur forc kill least twenti four protest wound mani crackdown peac demonstr across countri human right watch said today author cea use lethal forc unless absolut necessari protect live open independ investig lethal shoot human right watch said hundr peac protest took street februari seventeen two thousand eleven baida benghazi zenten derna ajdabiya accord multipl wit libyan secur forc shot kill demonstr effort disper protest secur forc viciou attack peac demonstr lay bare realiti muammar gaddafi brutal face intern dissent said sarah leah whitson middl east north africa director human right watch libyan risk live make stand right human worst violenc eastern citi baida around februari seventeen accord sourc libya hospit staff put call addit medic suppli becam overwhelm medic need seventi injur protest half critic condit due gunshot wound night februari sixteen secur forc attack peac protest tearga live ammunit shoot dead two protest accord protest spoke human right watch geneva base libya human right solidar confirm three name shot dead far safwan attiya nasser al juweigi ahmad el qabili one protest told human right watch new protest start februari seventeen noon prayer funer kill februari sixteen join hundr protest famili march toward intern secur offic chant regim get muammar gaddafi protest film protest mobil phone post onlin one injur protest hospit baida told human right watch sit near inten care unit confirm secur forc shot dead sixteen peopl wound dozen said special forc arm men street cloth fire live ammunit deter protest protest benghazi told human right watch hundr lawyer activist protest gather step benghazi court call constitut respect rule law earli day sourc libya told human right watch secur forc arrest benghazi journalist hind el houni salem souidan famili member group seek justic massacr inmat abu salim prison one thousand nine hundr nineti six secur forc also arrest former polit prison abdel nasser al rabbasi bani walid protest said saw group men street cloth arm knive later join intern secur forc charg protest disper protest told human right watch believ secur forc shot dead least seventeen protest day mostli near abdel nasser street human right watch abl confirm eight death appear govern also coordin pro govern support confront demonstr februari sixteen subscrib libyana one two libyan mobil phone network receiv text messag call upon nationalist youth go defend nation symbol around eleven thirti februari seventeen protest tripoli told human right watch anti govern protest start tripoli also</td>
      <td>0.099331</td>
    </tr>
    <tr>
      <th>1434</th>
      <td>Clue to slow human bird flu jump</td>
      <td>http://en.wikipedia.org/wiki/Singapore_2006</td>
      <td>Flu viruses which target man tend to attach to cells further up the airway - maximising their chances of being passed on by coughing or sneezing. Researchers found the bird flu virus attaches itself to cells deep down in the human airways. The University of Wisconsin research is published in the journal Nature. But it still cannot jump easily from human to human. Scientists fear that if it mutates and gains that ability, it could result in a human flu pandemic, with millions of deaths world-wide. The Wisconsin team investigated why the virus could not spread easily between humans despite the fact that it could replicate efficiently in human lungs. Flu viruses infecting humans and birds are known to home in on slightly different versions of the same molecule, found on the surface of cells which line the respiratory tract. The latest study found the version of the molecule targeted by human viruses was more prevalent on cells higher up in the airway. The molecule targeted by bird viruses, on the other hand, tended to be found on cells deep within the lungs, in structures called alveoli. Thus the bird flu virus tended to be buried so deep in the lungs that it was unlikely to be spread by coughing or sneezing.</td>
      <td>[]</td>
      <td>[]</td>
      <td>[]</td>
      <td>[]</td>
      <td>flu viru target man tend attach cell airway maximi chanc pass cough sneez research found bird flu viru attach cell deep human airway univ wisconsin research publish journal natur still jump easili human human scientist fear mutat gain abil could result human flu pandem million death world wide wisconsin team investig viru could spread easili human despit fact could replic effici human lung flu viru infect human bird known home slightli differ version molecul found surfac cell line respiratori tract latest studi found version molecul target human viru preval cell higher airway molecul target bird viru hand tend found cell deep within lung structur call alveoli thu bird flu viru tend buri deep lung unlik spread cough sneez</td>
      <td>0.078726</td>
    </tr>
    <tr>
      <th>1332</th>
      <td>Mystery China bug toll reaches 17</td>
      <td>http://en.wikipedia.org/wiki/Influenza_A_virus_subtype_H5N1</td>
      <td>The indications are that the disease is a bacterial infection spread by contact with dead pigs, and not a virus, officials in Sichuan province said. At least 58 people showed symptoms, which include high fever, nausea and vomiting, during June and July. The World Health Organization has urged calm, saying the disease is unable to spread from human to human. "I can assure you that the disease is absolutely not Sars, anthrax or bird flu," Zeng Huajin, a Sichuan health official, told the China Daily newspaper. The number of people infected with the illness has risen steadily as health officials searched through remote villages in the province for people with symptoms. A total of 17 people have died, with just two discharged from hospital. Twelve people remain in a critical condition while 27 are described as "stable", doctors said. Health officials said the illness could be a variant of the streptococcus bacteria, often found in pigs. The symptoms cannot be spread from human to human, and those most at risk from animal carcasses are people with vulnerable, low immune systems, officials said. Experts had expressed fears that pigs, which can also carry human influenza, could accelerate mutation of the bird flu virus into a form which can be transmitted between people.</td>
      <td>[Zeng Huajin]</td>
      <td>[June, July]</td>
      <td>[]</td>
      <td>[Zeng Huajin]</td>
      <td>indic disea bacteri infect spread contact dead pig viru offici sichuan provinc said least fifti eight peopl show symptom includ high fever nausea vomit june juli world health organ urg calm say disea unabl spread human human assur disea absolut sar anthrax bird flu zeng huajin sichuan health offici told china daili newspap number peopl infect ill risen steadili health offici search remot villag provinc peopl symptom total seventeen peopl die two discharg hospit twelv peopl remain critic condit twenti seven describ stabl doctor said health offici said ill could variant streptococcu bacteria often found pig symptom spread human human risk anim carcass peopl vulner low immun system offici said expert express fear pig also carri human influenza could accel mutat bird flu viru form transmit peopl</td>
      <td>0.068132</td>
    </tr>
  </tbody>
</table>
</div>



From this test, we can tell that the sample text is still represented in the dataset. For the purposes of our testing, we will leave the sample text in the dataset to ensure that our similarity testing is working appropriately. In each case, this should be our top result. For now, let's explore the remaining top 5 articles in similarity.


```python
cosine_samples = text_simple.sort_values(by='cosine_test', ascending=False).head(6)
cosine_samples = cosine_samples[1:]
```


```python
cosine_samples.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>article</th>
      <th>oururl</th>
      <th>newdescp</th>
      <th>names</th>
      <th>dates</th>
      <th>datetimes</th>
      <th>names_distinct</th>
      <th>processed_text</th>
      <th>cosine_test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2361</th>
      <td>Libya: Security Forces Kill 84 Over Three Days</td>
      <td>http://en.wikipedia.org/wiki/Libyan_Civil_War_(2011)</td>
      <td>Muammar Gaddafi's security forces are firing on Libyan citizens and killing scores simply because they're demanding change and accountability. Libyan authorities should allow peaceful protesters to have their say. (New York) - Government security forces have killed at least 84 people in three days of protests in several cities in Libya, Human Rights Watch said today, based on telephone interviews with local hospital staff and witnesses. The Libyan authorities should immediately end attacks on peaceful protesters and protect them from assault by pro-government armed groups, Human Rights Watch said. Thousands of demonstrators gathered in the eastern Libyan cities of Benghazi, Baida, Ajdabiya, Zawiya, and Derna on February 18, 2011, following violent attacks against peaceful protests the day before that killed 20 people in Benghazi, 23 in Baida, three in Ajdabiya, and three in Derna. Hospital sources told Human Rights Watch that security forces killed 35 people in Benghazi on February 18, almost all with live ammunition. "Muammar Gaddafi's security forces are firing on Libyan citizens and killing scores simply because they're demanding change and accountability," said Joe Stork, deputy Middle East and North Africa director at Human Rights Watch. "Libyan authorities should allow peaceful protesters to have their say." The protests in Benghazi on February 18 began during funerals for the 20 demonstrators killed by security forces the day before. Eyewitnesses told Human Rights Watch that security forces with distinctive yellow uniforms opened fire on protesters near the Fadil Bu Omar Katiba, a security force base in the center of Benghazi. One protester told Human Rights Watch he witnessed four men shot dead. By 11 p.m. on February 18, Al Jalaa Hospital in Benghazi had received the bodies of 35 people killed that day, a senior hospital official told Human Rights Watch. He said the deaths had been caused by gunshot wounds to the chest, neck, and head. Two sources at the hospital confirmed to Human Rights Watch that the death toll for February 17 was 20, and that at least 45 people had been wounded by bullets. The senior hospital official told Human Rights Watch, "We put out a call to all the doctors in Benghazi to come to the hospital and for everyone to contribute blood because I've never seen anything like this before." Witnesses said that after the February 18 shootings, protesters in Benghazi continued on to the courthouse and gathered there throughout the evening, the crowd swelling to thousands. In Baida, further to the east, protesters on February 18 buried the 23 people who had been shot dead the day before. One protester told Human Rights Watch that police were patrolling the streets but he had seen no further clashes.</td>
      <td>[Muammar Gaddafi, Muammar Gaddafi, Joe Stork]</td>
      <td>[three days, today, February 18, 2011, the day, February 18, February 18, the day before, February 18, that day, February 17, 20, February 18, February 18, the day before]</td>
      <td>[2020-03-21 00:00:00, 2011-02-18 00:00:00, 2020-02-18 00:00:00, 2020-02-18 00:00:00, 2020-02-18 00:00:00, 2020-02-17 00:00:00, 2020-03-21 20:00:00, 2020-02-18 00:00:00, 2020-02-18 00:00:00]</td>
      <td>[Muammar Gaddafi, Joe Stork]</td>
      <td>muammar gaddafi secur forc fire libyan citizen kill score simpli theyr demand chang account libyan author allow peac protest say new york govern secur forc kill least eighti four peopl three day protest sever citi libya human right watch said today base telephon interview local hospit staff wit libyan author immedi end attack peac protest protect assault pro govern arm group human right watch said thousand demonstr gather eastern libyan citi benghazi baida ajdabiya zawiya derna februari eighteen two thousand eleven follow violent attack peac protest day kill twenti peopl benghazi twenti three baida three ajdabiya three derna hospit sourc told human right watch secur forc kill thirti five peopl benghazi februari eighteen almost live ammunit muammar gaddafi secur forc fire libyan citizen kill score simpli theyr demand chang account said joe stork deputi middl east north africa director human right watch libyan author allow peac protest say protest benghazi februari eighteen began funer twenti demonstr kill secur forc day eyewit told human right watch secur forc distinct yellow uniform open fire protest near fadil bu omar katiba secur forc base center benghazi one protest told human right watch wit four men shot dead eleven februari eighteen al jalaa hospit benghazi receiv bodi thirti five peopl kill day senior hospit offici told human right watch said death cau gunshot wound chest neck head two sourc hospit confirm human right watch death toll februari seventeen twenti least forti five peopl wound bullet senior hospit offici told human right watch put call doctor benghazi come hospit everyon contribut blood ive never seen anyth like wit said februari eighteen shoot protest benghazi continu courthou gather throughout even crowd swell thousand baida east protest februari eighteen buri twenti three peopl shot dead day one protest told human right watch polic patrol street seen clash</td>
      <td>0.102610</td>
    </tr>
    <tr>
      <th>2624</th>
      <td>Libya: Security Forces Fire on 'Day of Anger' Demonstrations</td>
      <td>http://en.wikipedia.org/wiki/Libyan_Civil_War_(2011)</td>
      <td>The security forces' vicious attacks on peaceful demonstrators lay bare the reality of Muammar Gaddafi's brutality when faced with any internal dissent. Libyans should not have to risk their lives to make a stand for their rights as human beings. (New York) - The Libyan security forces killed at least 24 protesters and wounded many others in a crackdown on peaceful demonstrations across the country, Human Rights Watch said today. The authorities should cease the use of lethal force unless absolutely necessary to protect lives and open an independent investigation into the lethal shootings, Human Rights Watch said. Hundreds of peaceful protesters took to the streets on February 17, 2011, in Baida, Benghazy, Zenten, Derna, and Ajdabiya. According to multiple witnesses, Libyan security forces shot and killed the demonstrators in efforts to disperse the protests. "The security forces' vicious attacks on peaceful demonstrators lay bare the reality of Muammar Gaddafi's brutality when faced with any internal dissent," said Sarah Leah Whitson, Middle East and North Africa director at Human Rights Watch. "Libyans should not have to risk their lives to make a stand for their rights as human beings." Some of the worst violence was in the eastern city of Baida. At around 1 p.m. on February 17, according to sources in Libya, hospital staff put out a call for additional medical supplies, as they became overwhelmed with the medical needs of 70 injured protesters, half of them in critical condition due to gunshot wounds. On the night of February 16, security forces had attacked peaceful protesters with teargas and live ammunition, shooting dead two protesters, according to protesters who spoke to Human Rights Watch. Geneva-based Libya Human Rights Solidarity has confirmed three of the names of those shot dead so far: Safwan Attiya, Nasser Al Juweigi, and Ahmad El Qabili. One protester told Human Rights Watch that a new protest started on February 17, after noon prayers and the funerals of those killed on February 16. Joined by hundreds of other protesters, families marched toward the Internal Security office, chanting, "Down with the regime" and "Get out Muammar Gaddafi." Some protesters filmed the protests with mobile phones and posted them online. One injured protester in a hospital in Baida told Human Rights Watch that he was sitting near the intensive care unit there and had confirmed that security forces had shot dead 16 people and wounded dozens of others. He said that Special Forces and armed men in street clothes fired live ammunition to deter protesters. A protester in Benghazi told Human Rights Watch that hundreds of lawyers, activists, and other protesters gathered on the steps of the Benghazi Court calling for a constitution and respect for the rule of law. Early in the day, sources in Libya told Human Rights Watch that security forces had arrested a Benghazi journalist, Hind El Houny, and Salem Souidan, a family member of a group that has been seeking justice for the massacre of inmates in Abu Salim prison in 1996. Security forces also arrested a former political prisoner, Abdel Nasser al-Rabbasi, in Bani Walid. The protester said he saw groups of men in street clothes armed with knives, later joined by Internal Security forces, charging the protesters to disperse them. The protester told Human Rights Watch that he believed security forces had shot dead at least 17 protesters during the day, mostly near Abdel Nasser Street. Human Rights Watch was able to confirm eight of those deaths. It appears that the government also has coordinated pro-government supporters to confront the demonstrations. On February 16, subscribers to Libyana, one of two Libyan mobile phone networks, received a text message calling upon "nationalist youth" to go out and "defend national symbols." At around 11:30 p.m. on February 17, a protester in Tripoli told Human Rights Watch that anti-government protests had started in Tripoli also.</td>
      <td>[Muammar Gaddafi, Muammar Gaddafi, Sarah Leah Whitson, Safwan Attiya, Nasser Al Juweigi, Ahmad El Qabili, Muammar Gaddafi, Hind El Houny, Salem Souidan, Abdel Nasser al, Rabbasi]</td>
      <td>[today, February 17, 2011, February 17, February 17, February 16, the day, 1996, the day, February 16, February 17]</td>
      <td>[2020-03-21 00:00:00, 2011-02-17 00:00:00, 2020-02-17 00:00:00, 2020-02-17 00:00:00, 2020-02-16 00:00:00, 2020-02-16 00:00:00, 2020-02-17 00:00:00]</td>
      <td>[Ahmad El Qabili, Rabbasi, Nasser Al Juweigi, Sarah Leah Whitson, Hind El Houny, Abdel Nasser al, Salem Souidan, Safwan Attiya, Muammar Gaddafi]</td>
      <td>secur forc viciou attack peac demonstr lay bare realiti muammar gaddafi brutal face intern dissent libyan risk live make stand right human new york libyan secur forc kill least twenti four protest wound mani crackdown peac demonstr across countri human right watch said today author cea use lethal forc unless absolut necessari protect live open independ investig lethal shoot human right watch said hundr peac protest took street februari seventeen two thousand eleven baida benghazi zenten derna ajdabiya accord multipl wit libyan secur forc shot kill demonstr effort disper protest secur forc viciou attack peac demonstr lay bare realiti muammar gaddafi brutal face intern dissent said sarah leah whitson middl east north africa director human right watch libyan risk live make stand right human worst violenc eastern citi baida around februari seventeen accord sourc libya hospit staff put call addit medic suppli becam overwhelm medic need seventi injur protest half critic condit due gunshot wound night februari sixteen secur forc attack peac protest tearga live ammunit shoot dead two protest accord protest spoke human right watch geneva base libya human right solidar confirm three name shot dead far safwan attiya nasser al juweigi ahmad el qabili one protest told human right watch new protest start februari seventeen noon prayer funer kill februari sixteen join hundr protest famili march toward intern secur offic chant regim get muammar gaddafi protest film protest mobil phone post onlin one injur protest hospit baida told human right watch sit near inten care unit confirm secur forc shot dead sixteen peopl wound dozen said special forc arm men street cloth fire live ammunit deter protest protest benghazi told human right watch hundr lawyer activist protest gather step benghazi court call constitut respect rule law earli day sourc libya told human right watch secur forc arrest benghazi journalist hind el houni salem souidan famili member group seek justic massacr inmat abu salim prison one thousand nine hundr nineti six secur forc also arrest former polit prison abdel nasser al rabbasi bani walid protest said saw group men street cloth arm knive later join intern secur forc charg protest disper protest told human right watch believ secur forc shot dead least seventeen protest day mostli near abdel nasser street human right watch abl confirm eight death appear govern also coordin pro govern support confront demonstr februari sixteen subscrib libyana one two libyan mobil phone network receiv text messag call upon nationalist youth go defend nation symbol around eleven thirti februari seventeen protest tripoli told human right watch anti govern protest start tripoli also</td>
      <td>0.099331</td>
    </tr>
    <tr>
      <th>1434</th>
      <td>Clue to slow human bird flu jump</td>
      <td>http://en.wikipedia.org/wiki/Singapore_2006</td>
      <td>Flu viruses which target man tend to attach to cells further up the airway - maximising their chances of being passed on by coughing or sneezing. Researchers found the bird flu virus attaches itself to cells deep down in the human airways. The University of Wisconsin research is published in the journal Nature. But it still cannot jump easily from human to human. Scientists fear that if it mutates and gains that ability, it could result in a human flu pandemic, with millions of deaths world-wide. The Wisconsin team investigated why the virus could not spread easily between humans despite the fact that it could replicate efficiently in human lungs. Flu viruses infecting humans and birds are known to home in on slightly different versions of the same molecule, found on the surface of cells which line the respiratory tract. The latest study found the version of the molecule targeted by human viruses was more prevalent on cells higher up in the airway. The molecule targeted by bird viruses, on the other hand, tended to be found on cells deep within the lungs, in structures called alveoli. Thus the bird flu virus tended to be buried so deep in the lungs that it was unlikely to be spread by coughing or sneezing.</td>
      <td>[]</td>
      <td>[]</td>
      <td>[]</td>
      <td>[]</td>
      <td>flu viru target man tend attach cell airway maximi chanc pass cough sneez research found bird flu viru attach cell deep human airway univ wisconsin research publish journal natur still jump easili human human scientist fear mutat gain abil could result human flu pandem million death world wide wisconsin team investig viru could spread easili human despit fact could replic effici human lung flu viru infect human bird known home slightli differ version molecul found surfac cell line respiratori tract latest studi found version molecul target human viru preval cell higher airway molecul target bird viru hand tend found cell deep within lung structur call alveoli thu bird flu viru tend buri deep lung unlik spread cough sneez</td>
      <td>0.078726</td>
    </tr>
    <tr>
      <th>1332</th>
      <td>Mystery China bug toll reaches 17</td>
      <td>http://en.wikipedia.org/wiki/Influenza_A_virus_subtype_H5N1</td>
      <td>The indications are that the disease is a bacterial infection spread by contact with dead pigs, and not a virus, officials in Sichuan province said. At least 58 people showed symptoms, which include high fever, nausea and vomiting, during June and July. The World Health Organization has urged calm, saying the disease is unable to spread from human to human. "I can assure you that the disease is absolutely not Sars, anthrax or bird flu," Zeng Huajin, a Sichuan health official, told the China Daily newspaper. The number of people infected with the illness has risen steadily as health officials searched through remote villages in the province for people with symptoms. A total of 17 people have died, with just two discharged from hospital. Twelve people remain in a critical condition while 27 are described as "stable", doctors said. Health officials said the illness could be a variant of the streptococcus bacteria, often found in pigs. The symptoms cannot be spread from human to human, and those most at risk from animal carcasses are people with vulnerable, low immune systems, officials said. Experts had expressed fears that pigs, which can also carry human influenza, could accelerate mutation of the bird flu virus into a form which can be transmitted between people.</td>
      <td>[Zeng Huajin]</td>
      <td>[June, July]</td>
      <td>[]</td>
      <td>[Zeng Huajin]</td>
      <td>indic disea bacteri infect spread contact dead pig viru offici sichuan provinc said least fifti eight peopl show symptom includ high fever nausea vomit june juli world health organ urg calm say disea unabl spread human human assur disea absolut sar anthrax bird flu zeng huajin sichuan health offici told china daili newspap number peopl infect ill risen steadili health offici search remot villag provinc peopl symptom total seventeen peopl die two discharg hospit twelv peopl remain critic condit twenti seven describ stabl doctor said health offici said ill could variant streptococcu bacteria often found pig symptom spread human human risk anim carcass peopl vulner low immun system offici said expert express fear pig also carri human influenza could accel mutat bird flu viru form transmit peopl</td>
      <td>0.068132</td>
    </tr>
    <tr>
      <th>1070</th>
      <td>U.N. rights council condemns Syrian abuses</td>
      <td>http://en.wikipedia.org/wiki/Syrian_Civil_War</td>
      <td>The U.N. Human Rights Council decried a wide range of human rights violations in Syria on Friday and called for U.N. bodies to consider a recent report detailing the abuses and take" appropriate action." The council passed a resolution that "strongly condemns the continued widespread, systematic and gross violations of human rights and fundamental freedoms by the Syrian authorities, such as arbitrary executions, excessive use of force and the killing and persecution of protesters, human rights defenders and journalists, arbitrary detention, enforced disappearances, torture and ill-treatment, including against children." There were 37 yes votes, four against and six abstentions at the meeting in Geneva, Switzerland. The group convened to consider action against Syria after a troubling report issued Monday by the Independent International Commission of Inquiry, a body appointed by the council. That report concluded security and military forces "committed crimes against humanity" against civilians. The resolution recommends that U.N. bodies "urgently consider" the commission report and "take appropriate action." The group decided to send the Commission of Inquiry report to U.N. Secretary-General Ban Ki-moon "for appropriate action and transmission to all U.N. relevant bodies." It backs "efforts to protect the population of the Syrian Arab Republic and to bring an immediate end to gross human rights violations." And, it urged Syria "to protect its population" and "to immediately put an end to all human rights violations." The resolution also decided "to establish a mandate of a Special Rapporteur on the situation of human rights in Syria and urges Syria to cooperate with it. Before the resolution was adopted, U.N. High Commissioner for Human Rights Navi Pillay told the council Syria faces a "full-fledged civil war" if the regime's "continual ruthless repression" against peaceful demonstrators and civilians isn't stopped now. She noted with concern the reports of "increased armed attacks by the opposition forces, including the so-called Free Syrian army, against the Syrian military and security apparatus." "In light of the manifest failure of the Syrian authorities to protect their citizens, the international community needs to take urgent and effective measures to protect the Syrian people," Pillay said.</td>
      <td>[Ban Ki, Navi Pillay, Pillay]</td>
      <td>[Friday, Monday]</td>
      <td>[2020-03-27 00:00:00, 2020-03-23 00:00:00]</td>
      <td>[Navi Pillay, Pillay, Ban Ki]</td>
      <td>human right council decri wide rang human right violat syria friday call bodi consid recent report detail abu take appropri action council pass resolut strongli condemn continu widespread systemat gross violat human right fundament freedom syrian author arbitrari execut excess use forc kill persecut protest human right defend journalist arbitrari detent enforc disappear tortur ill treatment includ children thirti seven ye vote four six abstent meet geneva switzerland group conven consid action syria troubl report issu monday independ intern commiss inquiri bodi appoint council report conclud secur militari forc commit crime human civilian resolut recommend bodi urgent consid commiss report take appropri action group decid send commiss inquiri report secretari gener ban ki moon appropri action transmiss relev bodi back effort protect popul syrian arab republ bring immedi end gross human right violat urg syria protect popul immedi put end human right violat resolut also decid establish mandat special rapporteur situat human right syria urg syria cooper resolut adopt high commiss human right navi pillay told council syria face full fledg civil war regim continu ruthless repress peac demonstr civilian isnt stop note concern report increa arm attack opposit forc includ call free syrian armi syrian militari secur apparatu light manifest failur syrian author protect citizen intern commun need take urgent effect measur protect syrian peopl pillay said</td>
      <td>0.064439</td>
    </tr>
  </tbody>
</table>
</div>



### Test 1: Results

From exploring the top 5 articles by cosine similarity, it seems that articles 1, 2 and 5 have a good deal of contextual overlap. These artciles refer to protests and human rights abuses in Libya and Syria, while the sample text is about human rights abuses from protests in Egypt.

However, articles 3 and 4 are related mainly to viruses and have nothing to do protests, human rights, or the middle east. Further similarity testing is needed to refine results.

## Test 2: Sentence Embedding

For this test, we will utilize the Google Sentence Encoder. This encoder uses embeddings for words within larger sentence structures which are averaged together and given an embedding for the overall sentence. These larger embeddings are compared for correlation.

Because this model utilizes sentence structure, we will use the raw text from each article rather than its cleaned and tokenized version.


```python
# Import module from Universal Sentence Encoder
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)
print ("module %s loaded" % module_url)

# Define function to embed sentences.
def embed(input):
  return model(input)
```

    INFO:absl:Using /var/folders/1_/yqjpttyx6yvg73s_v4rn7_gc0000gn/T/tfhub_modules to cache modules.
    INFO:absl:Downloading TF-Hub Module 'https://tfhub.dev/google/universal-sentence-encoder/4'.
    INFO:absl:Downloading https://tfhub.dev/google/universal-sentence-encoder/4: 180.00MB
    INFO:absl:Downloading https://tfhub.dev/google/universal-sentence-encoder/4: 400.00MB
    INFO:absl:Downloading https://tfhub.dev/google/universal-sentence-encoder/4: 620.00MB
    INFO:absl:Downloading https://tfhub.dev/google/universal-sentence-encoder/4: 850.00MB
    INFO:absl:Downloaded https://tfhub.dev/google/universal-sentence-encoder/4, Total size: 987.47MB
    INFO:absl:Downloaded TF-Hub Module 'https://tfhub.dev/google/universal-sentence-encoder/4'.


    module https://tfhub.dev/google/universal-sentence-encoder/4 loaded



```python
# Define a function to score similarity based on sentence enbeddings
def embedding_corr(test_items):
  features = embed(test_items)
  # Return correlation between sentence embeddings
  # Using absolute value to standardize correlations
  corr = np.abs(np.inner(features, features))
  return corr[1,0]

# defining function to combine sample with each item
def embedding_test(item):
    sample = '''
         Human Rights Watch says government-controlled health services in Egypt have been pressured into playing down the number of casualties during anti-government protests. The group has documented the deaths of 297 people, but says the final toll is likely to be significantly higher. Human Rights Watch says the vast majority of the deaths in Cairo, Alexandria and Suez were on January 28 and 29 as a result of live gunfire as riot police fought running battles with protesters. A significant proportion came as a result of rubber bullets fired at too close a range and from teargas canisters fired into the crowds at very close range. Human Rights Watch says the actual number of deaths is likely to be an underestimate because the organisation had only included those deaths it had verified itself at key hospitals in the three major cities.
         '''
    test_items = [item, sample]
    return embedding_corr(test_items)
```


```python
# Applying sentence embedding test to all articles
text_simple['sentence_test'] = text_simple['newdescp'].apply(embedding_test).copy()
```


```python
sentence_samples = text_simple.sort_values(by='sentence_test', ascending=False).head(6)
sentence_samples = sentence_samples[1:]
sentence_samples.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>article</th>
      <th>oururl</th>
      <th>newdescp</th>
      <th>names</th>
      <th>dates</th>
      <th>datetimes</th>
      <th>names_distinct</th>
      <th>processed_text</th>
      <th>cosine_test</th>
      <th>sentence_test</th>
      <th>spacy_test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>89</th>
      <td>585</td>
      <td>Deaths in Egypt's Suez after Port Said football unrest</td>
      <td>http://en.wikipedia.org/wiki/2012?13_Egyptian_protests</td>
      <td>Two people have been killed and more than 400 injured in protests across Egypt sparked by the deaths of 74 people after a football match. The two killed were shot by police trying to disperse angry crowds in the city of Suez, medical officials said. In the capital Cairo, thousands of protesters remained on the streets following a day of clashes with police. Thousands marched to the interior ministry, where security forces fired tear gas to keep them back. Earlier, the Egyptian prime minister announced the sackings of several senior officials. Funerals of some of the 74 victims took place in Port Said, where the football match had taken place on Wednesday. The deaths came when fans invaded the pitch after a fixture between top Cairo club al-Ahly and the Port Said side al-Masry. As night fell in Cairo, several thousand demonstrators remained in the streets around the interior ministry, witnesses said. In Suez, health official Mohammed Lasheen said two people had been shot dead early on Friday. A witness quoted by Reuters said: "Protesters are trying to break into the Suez police station and police are now firing live ammunition."</td>
      <td>['Masry', 'Mohammed Lasheen']</td>
      <td>['a day', 'Wednesday', 'Friday']</td>
      <td>[datetime.datetime(2020, 3, 25, 0, 0), datetime.datetime(2020, 3, 27, 0, 0)]</td>
      <td>['Masry', 'Mohammed Lasheen']</td>
      <td>two peopl kill four hundr injur protest across egypt spark death seventi four peopl footbal match two kill shot polic tri disper angri crowd citi suez medic offici said capit cairo thousand protest remain street follow day clash polic thousand march interior ministri secur forc fire tear ga keep back earlier egyptian prime minist announc sack sever senior offici funer seventi four victim took place port said footbal match taken place wednesday death came fan invad pitch fixtur top cairo club al ahli port said side al masri night fell cairo sever thousand demonstr remain street around interior ministri wit said suez health offici moham lasheen said two peopl shot dead earli friday wit quot reuter said protest tri break suez polic station polic fire live ammunit</td>
      <td>0.033836</td>
      <td>0.708794</td>
      <td>0.844297</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2361</td>
      <td>Libya: Security Forces Kill 84 Over Three Days</td>
      <td>http://en.wikipedia.org/wiki/Libyan_Civil_War_(2011)</td>
      <td>Muammar Gaddafi's security forces are firing on Libyan citizens and killing scores simply because they're demanding change and accountability. Libyan authorities should allow peaceful protesters to have their say. (New York) - Government security forces have killed at least 84 people in three days of protests in several cities in Libya, Human Rights Watch said today, based on telephone interviews with local hospital staff and witnesses. The Libyan authorities should immediately end attacks on peaceful protesters and protect them from assault by pro-government armed groups, Human Rights Watch said. Thousands of demonstrators gathered in the eastern Libyan cities of Benghazi, Baida, Ajdabiya, Zawiya, and Derna on February 18, 2011, following violent attacks against peaceful protests the day before that killed 20 people in Benghazi, 23 in Baida, three in Ajdabiya, and three in Derna. Hospital sources told Human Rights Watch that security forces killed 35 people in Benghazi on February 18, almost all with live ammunition. "Muammar Gaddafi's security forces are firing on Libyan citizens and killing scores simply because they're demanding change and accountability," said Joe Stork, deputy Middle East and North Africa director at Human Rights Watch. "Libyan authorities should allow peaceful protesters to have their say." The protests in Benghazi on February 18 began during funerals for the 20 demonstrators killed by security forces the day before. Eyewitnesses told Human Rights Watch that security forces with distinctive yellow uniforms opened fire on protesters near the Fadil Bu Omar Katiba, a security force base in the center of Benghazi. One protester told Human Rights Watch he witnessed four men shot dead. By 11 p.m. on February 18, Al Jalaa Hospital in Benghazi had received the bodies of 35 people killed that day, a senior hospital official told Human Rights Watch. He said the deaths had been caused by gunshot wounds to the chest, neck, and head. Two sources at the hospital confirmed to Human Rights Watch that the death toll for February 17 was 20, and that at least 45 people had been wounded by bullets. The senior hospital official told Human Rights Watch, "We put out a call to all the doctors in Benghazi to come to the hospital and for everyone to contribute blood because I've never seen anything like this before." Witnesses said that after the February 18 shootings, protesters in Benghazi continued on to the courthouse and gathered there throughout the evening, the crowd swelling to thousands. In Baida, further to the east, protesters on February 18 buried the 23 people who had been shot dead the day before. One protester told Human Rights Watch that police were patrolling the streets but he had seen no further clashes.</td>
      <td>['Muammar Gaddafi', 'Muammar Gaddafi', 'Joe Stork']</td>
      <td>['three days', 'today', 'February 18, 2011', 'the day', 'February 18', 'February 18', 'the day before', 'February 18', 'that day', 'February 17', '20', 'February 18', 'February 18', 'the day before']</td>
      <td>[datetime.datetime(2020, 3, 21, 0, 0), datetime.datetime(2011, 2, 18, 0, 0), datetime.datetime(2020, 2, 18, 0, 0), datetime.datetime(2020, 2, 18, 0, 0), datetime.datetime(2020, 2, 18, 0, 0), datetime.datetime(2020, 2, 17, 0, 0), datetime.datetime(2020, 3, 21, 20, 0), datetime.datetime(2020, 2, 18, 0, 0), datetime.datetime(2020, 2, 18, 0, 0)]</td>
      <td>['Muammar Gaddafi', 'Joe Stork']</td>
      <td>muammar gaddafi secur forc fire libyan citizen kill score simpli theyr demand chang account libyan author allow peac protest say new york govern secur forc kill least eighti four peopl three day protest sever citi libya human right watch said today base telephon interview local hospit staff wit libyan author immedi end attack peac protest protect assault pro govern arm group human right watch said thousand demonstr gather eastern libyan citi benghazi baida ajdabiya zawiya derna februari eighteen two thousand eleven follow violent attack peac protest day kill twenti peopl benghazi twenti three baida three ajdabiya three derna hospit sourc told human right watch secur forc kill thirti five peopl benghazi februari eighteen almost live ammunit muammar gaddafi secur forc fire libyan citizen kill score simpli theyr demand chang account said joe stork deputi middl east north africa director human right watch libyan author allow peac protest say protest benghazi februari eighteen began funer twenti demonstr kill secur forc day eyewit told human right watch secur forc distinct yellow uniform open fire protest near fadil bu omar katiba secur forc base center benghazi one protest told human right watch wit four men shot dead eleven februari eighteen al jalaa hospit benghazi receiv bodi thirti five peopl kill day senior hospit offici told human right watch said death cau gunshot wound chest neck head two sourc hospit confirm human right watch death toll februari seventeen twenti least forti five peopl wound bullet senior hospit offici told human right watch put call doctor benghazi come hospit everyon contribut blood ive never seen anyth like wit said februari eighteen shoot protest benghazi continu courthou gather throughout even crowd swell thousand baida east protest februari eighteen buri twenti three peopl shot dead day one protest told human right watch polic patrol street seen clash</td>
      <td>0.102610</td>
      <td>0.697863</td>
      <td>0.877740</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2625</td>
      <td>Libya protests leave 24 dead, says rights group</td>
      <td>http://en.wikipedia.org/wiki/Libyan_Civil_War_(2011)</td>
      <td>At least 24 people have been killed in anti-government protests in Libya in recent days, rights activists say. Many others were wounded in the clashes between security forces and protesters, the US-based Human Rights Watch said. Protests continued overnight with thousands on the streets of the eastern city of Benghazi, where there is now a heavy military presence, witnesses said. Large protests are uncommon in Libya, where dissent is rarely allowed. Pro-democracy protests have recently swept through several Arab nations, with the presidents of Tunisia and Egypt forced from power amid growing unrest. The BBC's Jon Leyne in Cairo says violent confrontations are reported to have spread to five Libyan cities in demonstrations so far, but not yet to Tripoli, the capital, in any large numbers. Our correspondent says the reports reflect an extremely tough government response, including the use of gunfire and even denying supplies to hospitals. Funerals of some of those killed are expected to be held on Friday in Benghazi and al-Bayda, which correspondents say could spur more protests. Activists set up camps in al-Bayda after Thursday's "Day of Rage" protest against the government, witnesses said. Eyewitnesses believe that the death toll could be even higher, our correspondent says.</td>
      <td>['Jon Leyne']</td>
      <td>['recent days', 'Friday', 'Thursday']</td>
      <td>[datetime.datetime(2020, 3, 27, 0, 0), datetime.datetime(2020, 3, 26, 0, 0)]</td>
      <td>['Jon Leyne']</td>
      <td>least twenti four peopl kill anti govern protest libya recent day right activist say mani wound clash secur forc protest us base human right watch said protest continu overnight thousand street eastern citi benghazi heavi militari presenc wit said larg protest uncommon libya dissent rare allow pro democraci protest recent swept sever arab nation presid tunisia egypt forc power amid grow unrest bbc jon leyn cairo say violent confront report spread five libyan citi demonstr far yet tripoli capit larg number correspond say report reflect extrem tough govern respon includ use gunfir even deni suppli hospit funer kill expect held friday benghazi al bayda correspond say could spur protest activist set camp al bayda thursday day rage protest govern wit said eyewit believ death toll could even higher correspond say</td>
      <td>0.044797</td>
      <td>0.691777</td>
      <td>0.873319</td>
    </tr>
    <tr>
      <th>873</th>
      <td>638</td>
      <td>Hundreds hurt, 6 killed in Yemen violence</td>
      <td>http://en.wikipedia.org/wiki/Yemeni_Revolution</td>
      <td>(CNN) -- Yemeni protesters and military and pro-government gangs clashed in several areas Tuesday, with at least six killed and hundreds more injured, as the future of President Ali Abdullah Saleh remained uncertain. The United States has no intention of stopping its military aid to Yemen, despite the unrest, Pentagon spokesman Geoff Morrell said Tuesday. The aid, in support of Yemeni counterterrorism efforts, continues to be essential because of the "real threat" from al Qaeda in the country, he said. In Sanaa, the capital, eyewitnesses and field medical teams told CNN that security forces and anti-riot police used batons to attack protesters among 40,000 people marching on Zubairy Street Tuesday evening. In addition, pro-government gangs attacked protesters on Tuesday near a military base. Four people were killed -- three pro-government demonstrators and one anti-government demonstrator. Windows were shattered on an ambulance carrying some of the 56 injured protesters to a hospital, witnesses said. "The government forces are killing us," said Abdullah Salem, a youth activist who was at the protest. "Saleh and his militia will not succeed, and every blood spilt will be accounted for in international courts." In the city of Taiz, meanwhile, at least two anti-government protesters were killed when security forces and Republican Guards fired on protesters, according to medical teams. Hundreds of people were injured, 55 of them from gunshot wounds. The security chief in Taiz denied his forces fired on demonstrators. "Security forces did not attack protesters," said Abdullah Qiaran. "We were dispersing pro and anti-government protesters after we saw that both sides were clashing." An estimated 30,000 demonstrators marched near the presidential palace in the port city of Hodeida Tuesday evening, witnesses said. The violence comes as the United States is helping to mediate a transition out of office for Saleh, who has been facing popular protests for weeks, according to two Yemeni officials.</td>
      <td>['Ali Abdullah Saleh', 'Geoff Morrell', 'Abdullah Salem', 'Saleh', 'Abdullah Qiaran', 'Saleh']</td>
      <td>['Tuesday', 'Tuesday', 'Tuesday', 'Tuesday', 'Tuesday', 'weeks']</td>
      <td>[datetime.datetime(2020, 3, 24, 0, 0), datetime.datetime(2020, 3, 24, 0, 0), datetime.datetime(2020, 3, 24, 0, 0), datetime.datetime(2020, 3, 24, 0, 0), datetime.datetime(2020, 3, 24, 0, 0)]</td>
      <td>['Abdullah Qiaran', 'Saleh', 'Ali Abdullah Saleh', 'Geoff Morrell', 'Abdullah Salem']</td>
      <td>cnn yemeni protest militari pro govern gang clash sever area tuesday least six kill hundr injur futur presid ali abdullah saleh remain uncertain unit state intent stop militari aid yemen despit unrest pentagon spokesman geoff morrel said tuesday aid support yemeni counterterror effort continu essenti real threat al qaeda countri said sanaa capit eyewit field medic team told cnn secur forc anti riot polic use baton attack protest among forti thousand peopl march zubairi street tuesday even addit pro govern gang attack protest tuesday near militari base four peopl kill three pro govern demonstr one anti govern demonstr window shatter ambul carri fifti six injur protest hospit wit said govern forc kill us said abdullah salem youth activist protest saleh militia succeed everi blood spilt account intern court citi taiz meanwhil least two anti govern protest kill secur forc republican guard fire protest accord medic team hundr peopl injur fifti five gunshot wound secur chief taiz deni forc fire demonstr secur forc attack protest said abdullah qiaran disper pro anti govern protest saw side clash estim thirti thousand demonstr march near presidenti palac port citi hodeida tuesday even wit said violenc come unit state help mediat transit offic saleh face popular protest week accord two yemeni offici</td>
      <td>0.013653</td>
      <td>0.687851</td>
      <td>0.816593</td>
    </tr>
    <tr>
      <th>161</th>
      <td>804</td>
      <td>Yemen toll rises as U.S. seen pressing Saleh to go</td>
      <td>http://en.wikipedia.org/wiki/Yemeni_Revolution</td>
      <td>1 of 13. Anti-government protesters run after police fired tear gas during a demonstration in the southern Yemeni city of Taiz April 4, 2011. The attempt to suppress mounting protests inspired by uprisings in Egypt and Tunisia came amid signs that the United States is seeking an end to Saleh's 32-year rule, long seen as a rampart against Yemen-based al Qaeda in the Arabian Peninsula. In Taiz, south of the capital Sanaa, police shot at protesters trying to storm the provincial government building, killing at least 15 and wounding 30, hospital doctors said. "The regime has surprised us with this extent of killing. I don't think the people will do anything other than come out with bare chests to drain the government of all its ammunition," parliamentarian Mohammed Muqbil al-Hamiri told Al Jazeera TV. The television showed a row of men, apparent tear gas victims, lying motionless and being tended by medics on the carpeted floor of a makeshift hospital in Taiz. In the Red Sea port of Hudaida, police and armed men in civilian clothes attacked a march toward a presidential palace. Three people were hit by bullets, around 30 were stabbed with knifes, and 270 were hurt from inhaling tear gas, doctors said. Later on Monday, doctors said at least six demonstrators were shot dead and several wounded during evening rallies, and that the toll was likely to rise. In Washington, the U.S. State Department called the latest violence in Yemen "appalling." Yemen's opposition coalition appealed in a statement to the United Nations, human rights groups and other international bodies "to intervene quickly to stop President Saleh and his entourage from shedding more blood." As opposition forces stepped up their actions, Saleh again appeared defiant.</td>
      <td>['Saleh', 'Mohammed Muqbil al', 'Hamiri', 'Saleh', 'Saleh']</td>
      <td>['April 4, 2011', 'march', 'Monday']</td>
      <td>[datetime.datetime(2011, 4, 4, 0, 0), datetime.datetime(2020, 3, 23, 0, 0)]</td>
      <td>['Hamiri', 'Saleh', 'Mohammed Muqbil al']</td>
      <td>thirteen anti govern protest run polic fire tear ga demonstr southern yemeni citi taiz april two thousand eleven attempt suppress mount protest inspir upri egypt tunisia came amid sign unit state seek end saleh thirti two year rule long seen rampart yemen base al qaeda arabian peninsula taiz south capit sanaa polic shot protest tri storm provinci govern build kill least fifteen wound thirti hospit doctor said regim surpri us extent kill dont think peopl anyth come bare chest drain govern ammunit parliamentarian moham muqbil al hamiri told al jazeera tv televi show row men appar tear ga victim lie motionless tend medic carpet floor makeshift hospit taiz red sea port hudaida polic arm men civilian cloth attack march toward presidenti palac three peopl hit bullet around thirti stab knife two hundr seventi hurt inhal tear ga doctor said later monday doctor said least six demonstr shot dead sever wound even ralli toll like rise washington state depart call latest violenc yemen appal yemen opposit coalit appeal statement unit nation human right group intern bodi interven quickli stop presid saleh entourag shed blood opposit forc step action saleh appear defiant</td>
      <td>0.028330</td>
      <td>0.685098</td>
      <td>0.862826</td>
    </tr>
  </tbody>
</table>
</div>



### Test 2: Results

The results of the sentence encoder seem much more promising. Overall, all five articles share many similarities. They all refer to violent protests, and all are located somewhere in the Middle East/North Africa region (Egypt, Libya, Yemen). Several refer to human rights issues and anti-government protests, much like the sample article.

The results of this test seem far more accurate than the cosine similarity test.

## Test 3: Spacy Vectorization

Finally, we will run a similarity test using Spacy's internal similarity function. This model vectorizes the sample and article texts using spacy's nlp vectorizer utilized above. It then compares similarity in both directions (sample->test, test->sample) and averages the similarity score.

For this test, we will utilize the tokenized and cleaned version of the article text.


```python
# Loading spacy vectorized model - en_core_web_md
nlp = spacy.load("en_core_web_md")
```


```python
# Defining function to use spacy's vectorizer
def spacy_test(string):
    # Checking to make sure each value contains information
    if string:
        # Storing sample article and string, vectorizing using spacy nlp
        sample = nlp('''
             Human Rights Watch says government-controlled health services in Egypt have been pressured into playing down the number of casualties during anti-government protests. The group has documented the deaths of 297 people, but says the final toll is likely to be significantly higher. Human Rights Watch says the vast majority of the deaths in Cairo, Alexandria and Suez were on January 28 and 29 as a result of live gunfire as riot police fought running battles with protesters. A significant proportion came as a result of rubber bullets fired at too close a range and from teargas canisters fired into the crowds at very close range. Human Rights Watch says the actual number of deaths is likely to be an underestimate because the organisation had only included those deaths it had verified itself at key hospitals in the three major cities.
             ''')
        testdoc = nlp(string)
        # Testing for similarity in both directions
        similarity = sample.similarity(testdoc)
        similarity_rev = testdoc.similarity(sample)
        # Averaging both similarity scores
        sim_score = (similarity + similarity_rev) / 2
    else:
        pass
    return sim_score
```


```python
# Applying spacy vectorization test to all articles
text_simple['spacy_test'] = text_simple['processed_text'].apply(spacy_test).copy()
```


```python
spacy_samples = text_simple.sort_values(by='spacy_test', ascending=False).head(6)
spacy_samples = spacy_samples[1:]
spacy_samples.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>article</th>
      <th>oururl</th>
      <th>newdescp</th>
      <th>names</th>
      <th>dates</th>
      <th>datetimes</th>
      <th>names_distinct</th>
      <th>processed_text</th>
      <th>cosine_test</th>
      <th>sentence_test</th>
      <th>spacy_test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35</th>
      <td>1188</td>
      <td>UPDATE 1-Libya said to use cluster arms, Tripoli denies it</td>
      <td>http://en.wikipedia.org/wiki/Timeline_of_the_2011_Libyan_Civil_War</td>
      <td>Cluster munitions -- fired from artillery or rockets -- can scatter bomblets over a wide area. They often fail to detonate immediately and may explode years after a conflict, killing or maiming people, according to humanitarian groups. Human Rights Watch said in a statement that government forces had fired cluster munitions into residential areas in the western city of Misrata, "posing a grave risk to civilians". The group said it had observed at least three such weapons explode over the coastal city's el-Shawahda neighbourhood on the night of April 14. "Researchers inspected the remnants of a cluster submunition and interviewed witnesses to two other apparent cluster munition strikes," it said on its website, which also had photographs of what it said were the remnants of the munitions used in Misrata. "They pose a huge risk to civilians, both during attacks because of their indiscriminate nature and afterward because of the still-dangerous unexploded duds scattered about," he said. Based on a submunition first discovered by a New York Times reporter and inspected by Human Rights Watch, the group said the cluster munition was a Spanish-produced MAT-120 120 mm mortar projectile, which it said opens in mid-air and releases 21 submunitions over a wide area. It said it had not yet been able to determine if civilians in Misrata had been wounded or killed by such munitions. A rebel spokesman in Misrata, the insurgents' last major enclave in western Libya, also told Reuters cluster munitions had been used by pro-government forces there. "To use these bombs, the evidence would remain for days and weeks, and we know the international community is coming en masse to our country soon, so we can't do this." Libya has invited UNICEF to visit Misrata and on Saturday a Red Crescent and Red Cross team will go there, he said.</td>
      <td>[]</td>
      <td>['may explode years', 'days', 'weeks', 'Saturday']</td>
      <td>[datetime.datetime(2020, 5, 1, 0, 0), datetime.datetime(2020, 3, 21, 0, 0)]</td>
      <td>[]</td>
      <td>cluster munit fire artilleri rocket scatter bomblet wide area often fail deton immedi may explod year conflict kill maim peopl accord humanitarian group human right watch said statement govern forc fire cluster munit residenti area western citi misrata pose grave risk civilian group said observ least three weapon explod coastal citi el shawahda neighbourhood night april fourteen research inspect remnant cluster submunit interview wit two appar cluster munit strike said websit also photograph said remnant munit use misrata pose huge risk civilian attack indiscrimin natur afterward still danger unexplod dud scatter said base submunit first discov new york time report inspect human right watch group said cluster munit spanish produc mat one hundr twenti one hundr twenti mm mortar projectil said open mid air relea twenti one submunit wide area said yet abl determin civilian misrata wound kill munit rebel spokesman misrata insurg last major enclav western libya also told reuter cluster munit use pro govern forc use bomb evid would remain day week know intern commun come en mass countri soon cant libya invit unicef visit misrata saturday red crescent red cross team go said</td>
      <td>0.043845</td>
      <td>0.563499</td>
      <td>0.914272</td>
    </tr>
    <tr>
      <th>53</th>
      <td>8</td>
      <td>US admits Afghan airstrike errors</td>
      <td>http://en.wikipedia.org/wiki/Drone_strikes_in_Pakistan</td>
      <td>A row has been rumbling since the strikes in early May   Failure by US forces to follow their own rules was the "likely" cause of civilian deaths in Afghan airstrikes last month, a US military report says. US officials looked at seven strikes on Taliban targets in Farah province on 4 May, and concluded that three had not complied with military guidelines. The report accepts that at least 26 civilians died, but acknowledges that the real figure could be much higher. The Afghan government has said 140 civilians were killed in the strikes. Washington and Kabul have been at loggerheads for weeks over the number of civilians killed in the incident. The report's conclusions are couched in caveats, but by releasing it late on a Friday afternoon the Pentagon has underlined its embarrassment at what may be the worst case of civilian deaths since coalition forces entered the country in 2001. As well as acknowledging that there was a failure to follow strict military guidelines, the report recommends unspecified steps to be taken to refine that guidance and urges a greater engagement in the public relations battle.   It states that the coalition should be "first with the truth". Yet the report also calls into question whether the true number of civilian deaths, in this incident, will ever be known. It sticks with the US military's initial estimate of 26, but describes a report by the Afghan Independent Human Rights Commission, which speaks of at least 86 civilian casualties, as "balanced" and "thorough". The US report defends the Farah operation, saying the use of force "was an appropriate means to destroy that enemy threat". "However, the inability to discern the presence of civilians and avoid and/or minimise accompanying collateral damage resulted in the unintended consequence of civilian casualties," the report says. It says the final three strikes of the engagement, which took place after dark, did not adhere to "specific guidance" in the controlling directive. "Not applying all of that guidance likely resulted in civilian casualties," the report says. It concedes that the precise number of civilians killed in the attack may never be known because many victims were buried before the investigation started. The document makes a number of recommendations to reduce the likelihood of civilian deaths. It says lines of communication must be improved, new guidelines should be introduced and personnel need to be retrained. Gen Stanley McChrystal, the US commander in Afghanistan, is currently reviewing US rules in relation to airstrikes. He said last month that US forces should use them only if the lives of Nato personnel or American troops were clearly at risk. Both Nato and US have have insisted that avoiding civilian casualties is their priority in all battles.</td>
      <td>['Stanley McChrystal']</td>
      <td>['last month', '4 May', 'weeks', 'Friday', '2001', 'last month']</td>
      <td>[datetime.datetime(2020, 3, 27, 0, 0)]</td>
      <td>['Stanley McChrystal']</td>
      <td>row rumbl sinc strike earli may failur us forc follow rule like cau civilian death afghan airstrik last month us militari report say us offici look seven strike taliban target farah provinc may conclud three compli militari guidelin report accept least twenti six civilian die acknowledg real figur could much higher afghan govern said one hundr forti civilian kill strike washington kabul loggerhead week number civilian kill incid report conclu couch caveat relea late friday afternoon pentagon underlin embarrass may worst case civilian death sinc coalit forc enter countri two thousand one well acknowledg failur follow strict militari guidelin report recommend unspecifi step taken refin guidanc urg greater engag public relat battl state coalit first truth yet report also call question whether true number civilian death incid ever known stick us militari initi estim twenti six describ report afghan independ human right commiss speak least eighti six civilian casualti balanc thorough us report defend farah oper say use forc appropri mean destroy enemi threat howev inabl discern presenc civilian avoid minimi accompani collat damag result unintend consequ civilian casualti report say say final three strike engag took place dark adher specif guidanc control direct appli guidanc like result civilian casualti report say conc preci number civilian kill attack may never known mani victim buri investig start document make number recommend reduc likelihood civilian death say line commun must improv new guidelin introduc personnel need retrain gen stanley mcchrystal us command afghanistan current review us rule relat airstrik said last month us forc use live nato personnel american troop clearli risk nato us insist avoid civilian casualti prioriti battl</td>
      <td>0.038737</td>
      <td>0.575944</td>
      <td>0.912859</td>
    </tr>
    <tr>
      <th>657</th>
      <td>2473</td>
      <td>Report on sex abuse 'to be worse than Ferns'</td>
      <td>http://en.wikipedia.org/wiki/Sexual_abuse_in_Cloyne_diocese</td>
      <td>THE Government will today discuss the fourth major report into clerical child sexual abuse when Justice Minister Alan Shatter presents horrific findings from the Cork diocese of Cloyne. The report's findings are expected to be even graver than in Dublin and Ferns. Last night sources indicated that the Cabinet will approve the 400-page report of the Murphy Commission of Inquiry, and order its immediate publication tomorrow. Preparations were last night being made to allow victims and the media to read in advance the detailed 26 chapters of abuse complaints against 19 priests over a 13-year period from January 1 1996 to 2009. "An hour has been allocated for a pre-publication read ahead of a news conference which Mr Shatter is planning to hold at midday," a source said last night. The report is said to be damning of former Bishop John Magee's failures to implement agreed child protection procedures.But it likely to highlight the failure of the gardai and health services in dealing with a number of abuse complaints.</td>
      <td>['Alan Shatter', 'Shatter', 'John Magee']</td>
      <td>['today', 'tomorrow', 'January 1 1996 to 2009']</td>
      <td>[datetime.datetime(2020, 3, 21, 0, 0), datetime.datetime(2020, 3, 22, 0, 0)]</td>
      <td>['John Magee', 'Shatter', 'Alan Shatter']</td>
      <td>govern today discuss fourth major report cleric child sexual abu justic minist alan shatter present horrif find cork dioc cloyn report find expect even graver dublin fern last night sourc indic cabinet approv four hundr page report murphi commiss inquiri order immedi public tomorrow prepar last night made allow victim media read advanc detail twenti six chapter abu complaint nineteen priest thirteen year period januari one thousand nine hundr nineti six two thousand nine hour alloc pre public read ahead news confer mr shatter plan hold midday sourc said last night report said damn former bishop john mage failur implement agr child protect procedur like highlight failur gardai health servic deal number abu complaint</td>
      <td>0.016445</td>
      <td>0.384078</td>
      <td>0.910068</td>
    </tr>
    <tr>
      <th>418</th>
      <td>2815</td>
      <td>Iraq death toll 'soared post-war'</td>
      <td>http://en.wikipedia.org/wiki/Casualties_of_the_Iraq_War</td>
      <td>A study published by the Lancet says the risk of death by violence for civilians in Iraq is now 58 times higher than before the US-led invasion. Unofficial estimates of civilian deaths had varied from 10,000 to over 37,000. The Lancet admits the research is based on a small sample - under 1,000 homes - but says the findings are "convincing". Responding to the Lancet article, a Pentagon spokesman defended coalition action in Iraq. "This conflict has been prosecuted in the most precise fashion of any conflict in the history of modern warfare", he said. UK foreign secretary Jack Straw said his government would examine the findings "with very great care". But he told BBC's Today that another independent estimate of civilian deaths was around 15,000. The Iraq Body Count, a respected database run by a group of academics and peace activists, has put the number of reported civilian deaths at between 14,000-16,000. The Lancet published research by scientists from the Johns Hopkins Bloomberg School of Public Health in the US city of Baltimore. They gathered data on births and deaths since January 2002 from 33 clusters of 30 households each across Iraq.</td>
      <td>['Jack Straw']</td>
      <td>['Today', 'January 2002']</td>
      <td>[datetime.datetime(2002, 1, 1, 0, 0)]</td>
      <td>['Jack Straw']</td>
      <td>studi publish lancet say risk death violenc civilian iraq fifti eight time higher us led inva unoffici estim civilian death vari ten thousand thirti seven thousand lancet admit research base small sampl one thousand home say find convinc respond lancet articl pentagon spokesman defend coalit action iraq conflict prosecut preci fashion conflict histori modern warfar said uk foreign secretari jack straw said govern would examin find great care told bbc today anoth independ estim civilian death around fifteen thousand iraq bodi count respect databa run group academ peac activist put number report civilian death fourteen thousand sixteen thousand lancet publish research scientist john hopkin bloomberg school public health us citi baltimor gather data birth death sinc januari two thousand two thirti three cluster thirti household across iraq</td>
      <td>0.020383</td>
      <td>0.588396</td>
      <td>0.908171</td>
    </tr>
    <tr>
      <th>1986</th>
      <td>2508</td>
      <td>U.S. military scales down aid efforts in Philippines</td>
      <td>http://en.wikipedia.org/wiki/Typhoon_Haiyan</td>
      <td>Typhoon Haiyan, the most powerful storm to make landfall this year, struck the central Philippines on November 8, killing more than 5,200 people, displacing 4.4 million and destroying an estimated 12 billion pesos ($274 million) worth of crops and infrastructure. The U.S. Navy has pulled out its nuclear-powered aircraft carrier, the USS George Washington, but still has ten C-130 aircraft delivering relief supplies. Last week, the United States had 50 ships and aircraft in the disaster zone. Jeremy Konyndyk, director for Foreign Disaster Assistance at the U.S. Agency for International Development (USAID), said the U.S. military had started to reduce its presence to allow civilian aid agencies to step up efforts. "What we have seen, particularly over the past week, is now civilian and private-sector commercial capacity has started coming back up again and that is taking the burden off of the military actors," Konyndyk told Reuters in an interview. "You don't want the military playing that role in the long run, they are an interim bridging capacity there, but in the long run, that really needs to be civilian role." Konyndyk said there had been significant progress in meeting people's basic needs as more roads and ports opened in the worst-hit Leyte and Samar islands. "Food has been distributed to 3 million people, shelter kits have been delivered to tens of thousands of families. I think the situation with immediate humanitarian needs is becoming stabilized." Aid delivery was gathering pace as access to affected areas improved, the U.N. humanitarian office said it its latest report. However, major issues remained including the distribution of food and access to clean water and shelter material. Konyndyk said the next step was for USAID and other international aid agencies to refocus their efforts on long-term recovery and reconstruction, giving priority to shelter and livelihoods for farmers and fishermen. The United States has increased its typhoon aid to nearly $52 million, but latest estimates from the United Nations showed the disaster rehabilitation plan would cost $348 million. Only 38 percent of the plan is funded.</td>
      <td>['Jeremy Konyndyk', 'Konyndyk', 'Konyndyk', 'Konyndyk']</td>
      <td>['this year', 'November 8', 'Last week', 'the past week']</td>
      <td>[datetime.datetime(2020, 11, 8, 0, 0)]</td>
      <td>['Konyndyk', 'Jeremy Konyndyk']</td>
      <td>typhoon haiyan power storm make landfal year struck central philippin novemb kill five thousand two hundr peopl displac million destroy estim twelv billion peso two hundr seventi four million worth crop infrastructur navi pull nuclear power aircraft carrier uss georg washington still ten one hundr thirti aircraft deliv relief suppli last week unit state fifti ship aircraft disast zone jeremi konyndyk director foreign disast assist agenc intern develop usaid said militari start reduc presenc allow civilian aid agenc step effort seen particularli past week civilian privat sector commerci capac start come back take burden militari actor konyndyk told reuter interview dont want militari play role long run interim bridg capac long run realli need civilian role konyndyk said signif progress meet peopl basic need road port open worst hit leyt samar island food distribut million peopl shelter kit deliv ten thousand famili think situat immedi humanitarian need becom stabil aid deliveri gather pace access affect area improv humanitarian offic said latest report howev major issu remain includ distribut food access clean water shelter materi konyndyk said next step usaid intern aid agenc refocu effort long term recoveri reconstruct give prioriti shelter livelihood farmer fishermen unit state increa typhoon aid nearli fifti two million latest estim unit nation show disast rehabilit plan would cost three hundr forti eight million thirti eight percent plan fund</td>
      <td>0.005314</td>
      <td>0.454155</td>
      <td>0.907923</td>
    </tr>
  </tbody>
</table>
</div>



### Test 3 Results

This test required a significant amount more compute power than the others, and its results are similar to those of Test 1. Two articles seem to be related to civil unrest in the middle east region. The others, however, relate to a child sex abuse scandal in Cork, an estimate of civilian death toll in Iraq, and an article on Typhoon Haian in the Philippines.

On the whole, this model seems to be the least accurate.

# Part 2: Results

After processing the text using three separate tools, it seems that the Sentence Encoding test provides the most accurate result.

# Outputting Final Results to Files

Finally, I will output two files:
* 'texts_processed.csv': a final csv file of all relevant information, including the columns of names, distinct names, dates and datetime objects.
* 'closest_matches.csv': a sampling of the top five articles in similarity to the sample article, using the Sentence Embedding method.


```python
text_simple.to_csv('texts_processed.csv')
sentence_samples.to_csv('closest_matches.csv')
```
