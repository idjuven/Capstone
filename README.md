**Overview**

This project focus on the data analysis of an experiment from starbuck, which simulate how customer make purchaing decisions and how these decisions are influenced by promotional offers.
There are three types of offers that can be sent: buy-one-get-one (BOGO), discount, and informational. In a BOGO offer, a user needs to spend a certain amount to get a reward equal to that threshold amount. In a discount, a user gains a reward equal to a fraction of the amount spent. In an informational offer, there is no reward, but neither is there a requisite amount that the user is expected to spend. Offers can be delivered via multiple channels.

**Data Files:**
**profile.json**

Rewards program users (17000 users x 5 fields)

gender: (categorical) M, F, O, or null
age: (numeric) missing value encoded as 118
id: (string/hash)
became_member_on: (date) format YYYYMMDD
income: (numeric)


**portfolio.json**

Offers sent during 30-day test period (10 offers x 6 fields)

reward: (numeric) money awarded for the amount spent
channels: (list) web, email, mobile, social
difficulty: (numeric) money required to be spent to receive reward
duration: (numeric) time for offer to be open, in days
offer_type: (string) bogo, discount, informational
id: (string/hash)

**transcript.json**

Event log (306648 events x 4 fields)

person: (string/hash)
event: (string) offer received, offer viewed, transaction, offer completed
value: (dictionary) different values depending on event type
offer id: (string/hash) not associated with any "transaction"
amount: (numeric) money spent in "transaction"
reward: (numeric) money gained from "offer completed"
time: (numeric) hours after start of test

**Results and Conclusion**

Starbucks_Capstone_notebook.ipynb

After careful data cleaning and modeling, I have found deep insights from this Starbucks promotion experiment. 
I found additional demographic information including the membership year, age, sex and income distribution.

In addition, I built a machine learning model to determine which is the most important factor in determining the offer. 
I found "the total amount" plays key role in the offer decision. 
Other important factors include "income", " duration" and "reward". 
On the other hand, I found limited relation between customer sex, mobile or web and offer decisions.


**Blog**

https://medium.com/@shenl024_61906/using-data-science-to-understand-the-starbucks-promotional-program-fef836a520fd



**References**

https://towardsdatascience.com/optimizing-hyperparameters-in-random-forest-classification-ec7741f9d3f6

https://towardsdatascience.com/running-random-forests-inspect-the-feature-importances-with-this-code-2b00dd72b92e

https://towardsdatascience.com/decision-trees-and-random-forests-df0c3123f991

**List of Libraries**

import pandas as pd

import numpy as np

import math

import json

import matplotlib.pyplot as plt

import pickle

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score,f1_score

from sklearn.pipeline import Pipeline
