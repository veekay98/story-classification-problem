import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import inflect
from nltk.tokenize import sent_tokenize
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import math
from nltk import tokenize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from functionwords import FunctionWords


nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# Load the concreteness lexicon file into a DataFrame
dfc = pd.read_excel('Concreteness_ratings.xlsx', sheet_name='Sheet1')

# Basic hippocorpus data
df = pd.read_csv("hippoCorpusV2.csv")

# Encoding the memTypes with numeric values for text labels
encoder = LabelEncoder()
df['encoded_story_types'] = encoder.fit_transform(df['memType'])

#Excluding all rows of 'retold' Stories because we need only recalled and imagined stories
memType_to_exclude = ['retold']
df = df[~df['memType'].isin(memType_to_exclude)]

#######################################################################
# Calculating the sequentiality and topic NLL

# Function to calculate the log likelihood of the target sentence given the prompt
def calculate_log_likelihood(target_sentence, prompt):

        likelihood = 1.0
        log_likelihood = 0
        prompt_tokens = tokenizer.encode(prompt, return_tensors="pt")

        prompt_tokens_length = len(prompt_tokens[0])
        target_sentence_tokens = tokenizer.encode(target_sentence, return_tensors="pt")

        combined_token = torch.cat((prompt_tokens, target_sentence_tokens), 1)

        if model == "gpt3":
            return self.askGPT(prompt + " " + target_sentence, prompt_tokens_length)

        output = model(combined_token)
        logits = output[0]
        with torch.no_grad():

            for i in range(prompt_tokens_length, len(combined_token[0])):
                next_token_prob = torch.softmax(logits[:, i - 1, :], dim=-1)
                # Get the index of the next token in the sequence
                next_token_index = combined_token[0][i]

                # Get the likelihood of the actual next token
                token_likelihood = next_token_prob[0][next_token_index].item()

                # Multiply the likelihood with the running total
                likelihood *= token_likelihood
                log_likelihood += math.log(token_likelihood, 2)


        return log_likelihood

# Function to calculate the sequentiality
def calculate_seq(story, topic, history, full_size):

    sentences = sent_tokenize(story) # This discovers the sentences in the story

    # The sentences have to be at least of size 9 to find sequentiality for history size 9.
    # This check is done for all types of stories for consistency
    if (len(sentences)<=9):
        return None

    if (full_size=="true"):  # For the case of full size history
        history=len(sentences)-1

    for i in range(len(sentences)-history):

        sentence_context=""

        for k in range(history):
            sentence_context+=" "
            sentence_context+=sentences[i+k]

        prompt1=topic
        prompt2=topic+sentence_context


        l_topic = calculate_log_likelihood(sentences[i+history], prompt1) # Returns the log likelihood for target sentence from just the topic as prompt
        l_topic_context = calculate_log_likelihood(sentences[i+history], prompt2) # Returns the log likelihood for target sentence from both the topic and context as prompt
        topic_NLL=(-1)/len(sentences[i+history])*(l_topic)
        context_NLL=(-1)/len(sentences[i+history])*(l_topic_context)
        seq=1/len(sentences[i+history])*(l_topic_context-l_topic)


    return [seq, topic_NLL]

# This returns the sequentiality and topic NLL of all rows for a full size history
x=df.apply(lambda row: calculate_seq(row['story'],row['summary'],0,"true"), axis=1)

# This removes any rows that returned NA from the function and filters rows from df without NAs
x = x.dropna()
df = df.loc[x.index]

# Extracts first and second elements of the lists returned by the function
first_elements = x.apply(lambda y: y[0])
second_elements = x.apply(lambda y: y[1])

# This adds two new columns to the original df
df['Sequentiality_FullSize']=first_elements
df['Topic_NLL_FullSize']=second_elements

#######################################################################
# Calculate the concreteness of all Stories

# Returns the participle form of the verb
def get_participle_form(word):
    tag = pos_tag([word])[0][1]

    if tag == 'VBG':  # Present participle
        return 'present participle'
    elif tag == 'VBN':  # Past participle
        return 'past participle'
    else:
        return None

# Returns if a word is plural or no
def is_plural(word):
    p = inflect.engine()
    singular = p.singular_noun(word)

    if singular:
        return True  # If singular form exists, then word was plural
    return False

# Function to remove an element from an array
def remove_element(arr, elem):
    return [x for x in arr if x != elem]

# This returns the concreteness score of a string
def return_concreteness(text):
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()


    words=remove_element(words,'.')
    words=remove_element(words,',')
    words=remove_element(words,';')
    words=remove_element(words,':')

    sum=0
    counter=0

    for wd in words:
        if (wd=='has'): # This word wasn't present in the lexicon
            continue
        if (wd=="'s"):
            wd='is'
        wd=wd.lower()
        if wd=='i':
            wd=wd.upper() # 'i' is not a word whereas 'I' is
        if dfc[dfc['Word']==wd].empty: # If the lexicon doesnt contain the current form of the word
            if is_plural(wd):
                if dfc[dfc['Word']==wd].empty:
                    continue
                sing = p.singular_noun(wd) # convert the plural word to singular
                wd=sing
            elif (get_participle_form(wd)=='present participle' or get_participle_form(wd)=='past participle'):
                if dfc[dfc['Word']==wd].empty:
                    continue
                base_form = lemmatizer.lemmatize(wd, pos='v') # converts the word to the base form
                wd=base_form
            else:
                continue # if no cases match then skip it
        val=dfc[dfc['Word']==wd]['Conc.M'].iloc[0] # The concreteness value of the word from the lexicon
        sum+=val
        counter+=1

    print(sum/counter) # The average concreteness of all words in the sentence
    return (sum/counter)

# Drops rows where exception is thrown while finding concreteness
def safe_return_concreteness(text):
    try:
        return return_concreteness(text)
    except Exception as e:
        print(f"Error processing text: {e}")  # Optionally print error message
        return None

# Adds two new columns 'Concreteness_Story' and 'Concreteness_Summary'
df['Concreteness_Story'] = df['story'].apply(safe_return_concreteness)
df['Concreteness_Summary'] = df['summary'].apply(safe_return_concreteness)
df = df.dropna(subset=['Concreteness_Story', 'Concreteness_Summary'])

#########################################################################################

# Calculating the normalized emotion scores

# Load the lexicon
emotion_lexicon = load_nrc_emotion_lexicon('NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')

def load_nrc_emotion_lexicon(filename):
    # Initialize an empty dictionary to store the emotion lexicon
    emotion_dict = {}

    # Open the lexicon file for reading
    with open(filename, 'r') as file:
        # Iterate through each line in the file
        for line in file:
            # Split the line into word, emotion, and association value
            word, emotion, association = line.strip().split('\t')

            # If the word is not already in the dictionary, add it with an empty dict
            if word not in emotion_dict:
                emotion_dict[word] = {}

            # Store the association value for the emotion under the word
            emotion_dict[word][emotion] = int(association)

    # Return the populated emotion dictionary
    return emotion_dict

def aggregated_emotion_score(sentence, emotion_dict):
    # Initialize a counter for the total number of emotion-associated words
    total_emotion_count = 0

    # Split the sentence into words and convert them to lowercase
    words = sentence.lower().split()

    # Iterate through each word in the sentence
    for word in words:
        # Check if the word is in the emotion lexicon
        if word in emotion_dict:
            # Iterate through each emotion associated with the word
            for emotion in emotion_dict[word]:
                # If the association value is 1, increment the emotion count
                if emotion_dict[word][emotion] == 1:
                    total_emotion_count += 1

    # Return the total emotion count for the sentence
    return total_emotion_count

# Function to apply to each row in the DataFrame
def calculate_normalized_emotion_score(row):
    # Extract the story text from the row
    story = row['story']
    # Calculate the total emotion count for the story
    emotion_count = aggregated_emotion_score(story, emotion_lexicon)
    # Count the number of words in the story
    word_count = len(story.split())

    # Normalize the emotion count by the word count, avoiding division by zero
    if word_count > 0:
        return emotion_count / word_count
    else:
        return 0


# Apply the function and create a new column
df['emotion_score_norm'] = df.apply(calculate_normalized_emotion_score, axis=1)

###################################################################################
# Calculating the average sentence length and number of sentences in a story

def sentence_stats(text):
    sentences = sent_tokenize(text)
    num_sentences = len(sentences)
    total_words = sum(len(word_tokenize(sentence)) for sentence in sentences)
    avg_sentence_length = total_words / num_sentences if num_sentences > 0 else 0
    return num_sentences, avg_sentence_length

# Apply the function to each row and add 2 new features to the df 'num_sentences' and 'avg_sentence_length'
df[['num_sentences', 'avg_sentence_length']] = df['story'].apply(
    lambda x: pd.Series(sentence_stats(x))
)

###################################################################################
# Calculating the normalized function word count

# Function to calculate the function word count divided by the number of words in the story
def calculate_functionword_count(row):
    st = row['story']
    fw = FunctionWords(function_words_list='english')
    fw_count = sum(fw.transform(st))

    # Splitting the story into words and counting them
    words = st.split()
    num_words = len(words)

    # Avoid division by zero
    if num_words > 0:
        return fw_count / num_words
    else:
        return 0


# Apply the function and create a new column for the normalized function word score
df['fw_score_normalized'] = df.apply(calculate_functionword_count, axis=1)

#####################################################################################
# Convertng the df with all features added, to a csv file
df.to_csv('hippo-final-data.csv', index=False)
