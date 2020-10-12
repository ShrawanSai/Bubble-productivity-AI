import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from stemming.porter2 import stem
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
sp = spacy.load('en_core_web_sm')
all_stopwords = sp.Defaults.stop_words
import re
import numpy as np


class Recommendation:
    
    def __init__(self,path = 'tfidf_vectorizer.pkl' ):
        
        self.path = os.path.join(os.getcwd(), path)
        with open(self.path, 'rb') as f:
            self.vectorizer = pickle.load(f)


    def __preprocess_text(self,text : str) -> str:

   

        text = text.lower()
        text = re.sub(r'[^\x00-\x7F]+',' ', text)
        text = re.sub(r'[^a-zA-Z0-9 ]', '', text).strip()
        text = re.sub(r'\w*\d\w*', '', text).strip()
        text = re.sub(' +', ' ', text)

        pure_words_list = self.__remove_stops(text)
        stemmed_text = self.__stem_all_words(pure_words_list)

        return stemmed_text


    def __remove_stops(self,text: str) -> list:

        text_tokens = word_tokenize(text)
        tokens_without_sw= [word for word in text_tokens if not word in all_stopwords]

        return tokens_without_sw

    def __stem_all_words(self,text: list) -> str:

        w = [stem(word) for word in text]
        return ' '.join(w)
    
    def recommend(self,person: str, rest_vectors: list) -> list:


        ids = []
        descriptions = []
        for ind, value in enumerate(rest_vectors):

            ids.append(ind)
            descriptions.append(value)

        descriptions.insert(0,person)

        d = [self.__preprocess_text(i) for i in descriptions]
        descriptions = d

        tfidf_term_vectors  = self.vectorizer.transform(descriptions)


        tfidf_term_vectors = tfidf_term_vectors.toarray()

        similarities = []
        for i in range(1,len(rest_vectors)+1):
            similarities.append(cosine_similarity(np.array([tfidf_term_vectors[0]]), np.array([tfidf_term_vectors[i]])))

        ans = [x for _,x in sorted(zip(similarities,ids))]


        return ans[::-1]



# recommender = Recommendation()
# print(recommender.recommend(my_about_me,all_descriptions))


"""
my_about_me = '''
I am a fourth year engineering student at VIT University pursuing my computer science degree. I am currently into the fields of machine learning,
 natural language processing and artificial intelligence. I have 4 years of experience with python and
 I am looking forward to work on such projects in the future
'''
person_1 = '''
I am a front-end developer with 5 years of experience working with e-commerce companies.I specialize in using Java, PHP,
and Ruby to build customer facing APIs, and also have experience integrating payment systems.
'''

person_2 = '''
I’m a web developer from Mumbai, India. I focus on front-end web development to bring the best experience to your users. I have worked on many 
frontend and backend frameworks like Django, NodeJs etc.
'''

person_3 = '''
Highly organized and detail-oriented honors graduate from the University of Georgia seeking an entry-level position
as an accountant. Served as a peer tutor for courses such as general accounting, budgeting and forecasting, 
and accounting principles and legislation.
'''

person_4 = '''
I am a student pursuing to be an engineer at Manipal Institue of technology. I like working in Machine learning and Artificial Intelligence projects.
I am still a beginner but I want to be a successfull AI enthusiast in the future'''

person_5 = '''
I'm a senior software engineer working in Novartis working on machine learning and natural language processing. I am looking to work with students
pursuing engineering for exciting projects
'''

all_descriptions = [person_1, person_2, person_3, person_4, person_5]

recommender = Recommendation()
print(recommender.recommend(my_about_me,all_descriptions))
"""
