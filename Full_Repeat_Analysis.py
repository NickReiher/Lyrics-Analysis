"""
1 - Choose Artists or Songs
2 - download JSON files
2 - Find most repeated phrase of each length in each song
3 - Test different scoring methods within songs
4 - Choose the scoring method I want to use 
5 - Look for songs that have lots of repeats

Can I do this all a lot quicker? https://towardsdatascience.com/very-simple-python-script-for-extracting-most-common-words-from-a-story-1e3570d0b9d0 
Finding the n-grams: https://markhneedham.com/blog/2015/01/19/pythonnltk-finding-the-most-common-phrases-in-how-i-met-your-mother/
But this might do some weird stuff with contractions and such

"""

import lyricsgenius as genius

import json
import pandas as pd
import numpy as np
from collections import Counter 
import re


#Downloading lyrics as JSON files
def download_lyrics_by_artist(artists):
    
    #put your own Genius API code here - https://docs.genius.com/
    api_code = '<insert your own api code here'
    api = genius.Genius(api_code)
        

    #ideas - want to ignore additional things - 'Setlist', 'Intro', 'Costumes', 'Playlist', 'Tour Dates'
    #have an issue with skipping duplicates in lyricsgenius -- right now I am including multiple versions of the same song: very bad for my discography results
    for artist in artists:
        songs_by_artist = api.search_artist(artist, get_full_song_info=False)
        songs_by_artist.save_lyrics(overwrite=True, skip_duplicates=False)


def download_lyrics_by_title(titles):     
    #put your own Genius API code here - https://docs.genius.com/
    api_code = 'FH3YZPIr25z08uywhWrvdZG1BwHk3vOMPa7a9uAU5LAyRamQjdcU_1QhtRu3jriT'
    api = genius.Genius(api_code)
    
    for title, artist in titles:
        songs_by_title = api.search_song(title, artist_name=artist)
        songs_by_title.save_lyrics(filename = 'Lyrics ' + title + ' ' + artist, overwrite = True, binary_encoding = True)

        
#This method takes a tokenized text and a number n as input, then finds out which 1,2,3,...n-word phrase is repeated the most in the text, 
#as well as how often it is repeated.        
#If there is a tie, the first phrase to be looked at is chosen as the most common phrase of that length            
            
def find_max_repeats(words, number):
    repeat_dict = {}
    
    for i in np.arange(1, number + 1):
        words_now = []
        for x in np.arange(1, len(words) - i):
            current_word = ""
            for j in np.arange(1,i+1):
                current_word += words[x+j-1] + ' '
       
            words_now.append(current_word)
        
        counter_now = Counter(words_now)
        most_common_words = counter_now.most_common(1)[0][0]
        amount_of_repeats = counter_now[most_common_words]
        repeat_dict[i] = (most_common_words, amount_of_repeats) 
        
    
    return repeat_dict


def artists_json_to_dataframe(artists):
    df = pd.DataFrame(np.empty((0,3)))
    df.columns = ['artist', 'title', 'lyrics']

    for artist in artists:
        with open('lyrics_' + artist.replace(' ','') + '.json', 'r') as f:
            songs = json.load(f)
            for x in np.arange(len(songs['songs'])):
                a = songs['songs'][x]['artist']
                b = songs['songs'][x]['title']
                c = songs['songs'][x]['lyrics']
                if c is not None:
                    df.loc[len(df)] = [a, b, c] 
                
        
    return df


def title_txt_to_dataframe(titles):
    df = pd.DataFrame(np.empty((0,3)))
    df.columns = ['artist', 'title', 'lyrics']

    for title, artist in titles:
        with open('Lyrics ' + title + ' ' + artist + '.txt', 'r') as f:
            lyrics = f.read()            
            a = artist
            b = title
            c = lyrics
            if c is not None:
                    df.loc[len(df)] = [a, b, c] 
        
    return df
    
        
#Loading lyrics from JSON files to a pandas dataframe 
#and finding the most used phrase of each word length in each song    
def most_repeated_phrases(df, number_to_check):
    #This creates a dataframe where each song is a different entry
    #and includes the artist, title, and lyrics (as a string)

    #Create the empty dataframe df_most_common_phrases
    columns = ['Artist', 'Song', 
           'Number of Words', 'Number of Unique Words']

    for i in np.arange(1, number_to_check + 1):
        columns.append('Most Common Phrase - Length ' + str(i)) 
        columns.append('Number of Repeats - Length ' + str(i))
    
    df_most_common_phrases = pd.DataFrame(columns = columns)   
            
    
    for index, row in df.iterrows():
        lyrics = row['lyrics']    
        
        description_regex = '\[.*\]'
        lyrics = re.sub(description_regex, '', lyrics)
        
        #take out punctuation and grammar from the lyrics, and make it all lowercase
        lyrics_lower = lyrics.lower()
        lyrics_lower = lyrics_lower.replace('.','') 
        lyrics_lower = lyrics_lower.replace(',','')    
        lyrics_lower = lyrics_lower.replace('(','')
        lyrics_lower = lyrics_lower.replace(')','')
        lyrics_lower = lyrics_lower.replace('-','')
        lyrics_lower = lyrics_lower.replace('"','')
        lyrics_lower = lyrics_lower.replace('[','')
        lyrics_lower = lyrics_lower.replace(']','')
        
        #take out specific words - la, na
        lyrics_lower = lyrics_lower.replace(' la ','')
        lyrics_lower = lyrics_lower.replace(' na ','')
        lyrics_lower = lyrics_lower.replace(' oh ','')
        lyrics_lower = lyrics_lower.replace(' verse ','')
        lyrics_lower = lyrics_lower.replace(' ba ','')
        
        words = [w for w in lyrics_lower.split()]
        
        #idea - set up a minimum number of words for songs to have
        if len(words) > 50:
            
            repeats = find_max_repeats(words, number_to_check)                                  
            new_row = []
            
            #calculates the number of words, number of unique words
            new_row = [row['artist'], row['title'], len(words), len(set(words))]
            
            #adds in the max phrase for each phrase-length in the appropriate column, and finds which phrase has the highest max score
            for i in np.arange(1, number_to_check + 1):
                new_row.append(repeats[i][0])
                new_row.append(repeats[i][1])
                
            df_most_common_phrases.loc[len(df_most_common_phrases)] = new_row
    
    return(df_most_common_phrases)
    

def most_repeated_phrases_discography(df, number_to_check):
    #This creates a dataframe where each song is a different entry
    #and includes the artist, title, and lyrics (as a string)

    #Create the empty dataframe df_most_common_phrases
    columns = ['Artist',
           'Number of Words', 'Number of Unique Words']

    for i in np.arange(1, number_to_check + 1):
        columns.append('Most Common Phrase - Length ' + str(i)) 
        columns.append('Number of Repeats - Length ' + str(i))
    
    df_most_common_phrases = pd.DataFrame(columns = columns)   
            
    
    for index, row in df.iterrows():
        lyrics = row['lyrics']    
        
        description_regex = '\[.*\]'
        lyrics = re.sub(description_regex, '', lyrics)
        
        #take out punctuation and grammar from the lyrics, and make it all lowercase
        lyrics_lower = lyrics.lower()
        lyrics_lower = lyrics_lower.replace('.','') 
        lyrics_lower = lyrics_lower.replace(',','')    
        lyrics_lower = lyrics_lower.replace('(','')
        lyrics_lower = lyrics_lower.replace(')','')
        lyrics_lower = lyrics_lower.replace('-','')
        lyrics_lower = lyrics_lower.replace('"','')
        lyrics_lower = lyrics_lower.replace('[','')
        lyrics_lower = lyrics_lower.replace(']','')
        
        #take out specific words - la, na
        lyrics_lower = lyrics_lower.replace(' la ','')
        lyrics_lower = lyrics_lower.replace(' na ','')
        lyrics_lower = lyrics_lower.replace(' oh ','')
        lyrics_lower = lyrics_lower.replace(' verse ','')
        lyrics_lower = lyrics_lower.replace(' ba ','')
        
        words = [w for w in lyrics_lower.split()]
        
        #idea - set up a minimum number of words for songs to have
        if len(words) > 50:
            
            repeats = find_max_repeats(words, number_to_check)                                  
            new_row = []
            
            #calculates the number of words, number of unique words
            new_row = [row['artist'], len(words), len(set(words))]
            
            #adds in the max phrase for each phrase-length in the appropriate column, and finds which phrase has the highest max score
            for i in np.arange(1, number_to_check + 1):
                new_row.append(repeats[i][0])
                new_row.append(repeats[i][1])
                
            df_most_common_phrases.loc[len(df_most_common_phrases)] = new_row
    
    return(df_most_common_phrases)


#Scores each repeated phrase and returns the phrase, number of repeats, length, and score
#Can easily add more scoring methods to test with each other    
    
    
#Scoring Thoughts:
    #at the moment, there are a lot of long phrases having the highest score - not really what I'm going for
    #could put a minimum number of repetitions ( must be > 5 to be a repeated phrase - some songs won't have any, and then also a minimum score - like 5 times one word doesn't work)
    #sometimes it's the same phrase repeated multiple times, and actually the shorter phrase is more "memorable" - could see when a long phrase is repeated a few times, and the shorter one often within it
    #could look for the 'big drop' -- where adding one more word makes the number of times said go way down
    #one word repeated isn't that interesting (ignore those)
    #phrases like na na na and la la la aren't interesting (somehow ignore them?) - just take them out earlier in the process

    
def score_repeats_test(df, number_to_check):
    df['Top Phrase A'] = np.nan
    df['Top Phrase Length A'] = np.nan
    df['Top Phrase Number of Repeats A'] = np.nan
    df['Top Phrase Score A'] = np.nan
    
    for index, row in df.iterrows():
        top_phrase = ''
        top_phrase_length = 0
        top_phrase_repeats = 0
        top_phrase_score = 0
        for i in np.arange(2, number_to_check + 1):
            if df.iloc[index, 2*i + 3] < 5:
                score = 0
            else:
                score = df.iloc[index, 2*i + 3] * i**1.4
            if score > top_phrase_score:
                top_phrase = df.iloc[index, 2*i + 2]
                top_phrase_length = i
                top_phrase_repeats = df.iloc[index, 2*i + 3]
                top_phrase_score = score
        df.loc[index, 'Top Phrase A'] = top_phrase
        df.loc[index, 'Top Phrase Length A'] = top_phrase_length
        df.loc[index, 'Top Phrase Number of Repeats A'] = top_phrase_repeats
        df.loc[index, 'Top Phrase Score A'] = top_phrase_score
    
    df['Top Phrase B'] = np.nan
    df['Top Phrase Length B'] = np.nan
    df['Top Phrase Number of Repeats B'] = np.nan
    df['Top Phrase Score B'] = np.nan
    
    
    for index, row in df.iterrows():
        top_phrase = ''
        top_phrase_length = 0
        top_phrase_repeats = 0
        top_phrase_score = 0
        for i in np.arange(2, number_to_check + 1):
            if df.iloc[index, 2*i + 3] < 5:
                score = 0
            else:
                score = df.iloc[index, 2*i + 3] * i**1.2
            if score > top_phrase_score:
                top_phrase = df.iloc[index, 2*i + 2]
                top_phrase_length = i
                top_phrase_repeats = df.iloc[index, 2*i + 3]
                top_phrase_score = score
        df.loc[index, 'Top Phrase B'] = top_phrase
        df.loc[index, 'Top Phrase Length B'] = top_phrase_length
        df.loc[index, 'Top Phrase Number of Repeats B'] = top_phrase_repeats
        df.loc[index, 'Top Phrase Score B'] = top_phrase_score
        
    df['Top Phrase C'] = np.nan
    df['Top Phrase Length C'] = np.nan
    df['Top Phrase Number of Repeats C'] = np.nan
    df['Top Phrase Score C'] = np.nan
    
    for index, row in df.iterrows():
        top_phrase = ''
        top_phrase_length = 0
        top_phrase_repeats = 0
        top_phrase_score = 0
        for i in np.arange(1, number_to_check + 1):
            score = df.iloc[index, 2*i + 3] * i**1.2
            if score > top_phrase_score:
                top_phrase = df.iloc[index, 2*i + 2]
                top_phrase_length = i
                top_phrase_repeats = df.iloc[index, 2*i + 3]
                top_phrase_score = score
        df.loc[index, 'Top Phrase C'] = top_phrase
        df.loc[index, 'Top Phrase Length C'] = top_phrase_length
        df.loc[index, 'Top Phrase Number of Repeats C'] = top_phrase_repeats
        df.loc[index, 'Top Phrase Score C'] = top_phrase_score
           
    return(df)


#Only one score used here                
def score_repeats(df, number_to_check):
    df['Top Phrase'] = np.nan
    df['Top Phrase Length'] = np.nan
    df['Top Phrase Number of Repeats'] = np.nan
    df['Top Phrase Score'] = np.nan
    
    for index, row in df.iterrows():
        top_phrase = ''
        top_phrase_length = 0
        top_phrase_repeats = 0
        top_phrase_score = 0
        for i in np.arange(2, number_to_check + 1):
            if df.iloc[index, 2*i + 3] < 5:
                score = 0
            else:
                score = df.iloc[index, 2*i + 3] * i**1.4
            if score > top_phrase_score:
                top_phrase = df.iloc[index, 2*i + 2]
                top_phrase_length = i
                top_phrase_repeats = df.iloc[index, 2*i + 3]
                top_phrase_score = score
        df.loc[index, 'Top Phrase'] = top_phrase
        df.loc[index, 'Top Phrase Length'] = top_phrase_length
        df.loc[index, 'Top Phrase Number of Repeats'] = top_phrase_repeats
        df.loc[index, 'Top Phrase Score'] = top_phrase_score
    
    return(df)
                        
            
                
#What I'm currently running

 
#What artists will I analyze     
#idea - could also do it by song title, or with a list (like the top 100 chart)    
country_artists = ['Dylan Scott', 
                   'Brett Young',
                   'Morgan Evans',
                   'Kacey Musgraves',
                   'Dierks Bentley',
                   'Keith Urban',
                   'Taylor Swift']

country_artists2 = [
           'Bebe Rexha',
           'Dan + Shay',
           'Florida Georgia Line',
           'Kane Brown',
           'Old Dominion',
           'Luke Combs',
           'Luke Bryan',
           'Russell Dickerson',
           'Cole Swindell',
           'Jason Aldean',
           'Mitchell Tenpenny',
           'Kenny Chesney',
           'Chris Janson',
           'Eric Church',
           'Chris Young',
           'Jimmie Allen',
           'Maren Morris',
           'Brett Young']

artists_with_many_number_ones = ['The Beatles', 
                                 'Jay-Z',
                                 'Bruce Springsteen',
                                 'Barbra Streisand',
                                 'Elvis Presley',
                                 'Garth Brooks',
                                 'The Rolling Stones',
                                 'Kenny Chesney',
                                 'Eminem',
                                 'Madonna',
                                 'U2',
                                 'Kanye West']

artists_german_top_rock_bands = ['Die Toten Hosen',
           'Die ärzte']


#idea - import this list somehow from the charts and convert it to a list of tuples
#was having issues here b/c you have to match the artist exactly with lyricsgenius - not easy in
#collaborations
titles_country = [('Meant to Be', 'Bebe Rexha'),
          ('Tequila', 'Dan + Shay'),
          ('Simple', 'Florida Georgia Line'),
          ('Heaven', 'Kane Brown'),
          ('Hotel Key', 'Old Dominion'),
          ('She Got the Best of Me', 'Luke Combs'),
          ('Sunrise, Sunburn, Sunset', 'Luke Bryan'),
          ('Blue Tacoma', 'Russell Dickerson'),
          ('Hooked', 'Dylan Scott')]

titles_german_charts_current = [('Bonnie & Clyde', 'Loredana & Mozzik'),
          ('Dior 2001', 'Rin'),
          ('500 PS', 'Bonez MC & RAF Camora'),
          ('Promises', 'Calvin Harris & Sam Smith'),
          ('Mephisto', 'Bushido')]


#download lyrics from genius.com

download_lyrics_by_artist(country_artists) 
download_lyrics_by_artist(country_artists2) 


#download_lyrics_by_title(titles) 

"""
#put artist, title, and lyrics into a dataframe
#idea - could at this point also make a method that does it for json files of songs
#df_lyrics = title_txt_to_dataframe(titles)   
df_lyrics = artists_json_to_dataframe(country_artists) 

df_lyrics_discography = df_lyrics.groupby('artist')['lyrics'].apply(lambda x: x.sum()).reset_index()


#give the artists I want to look at and the max number of words
most_repeated_phrases = most_repeated_phrases(df_lyrics, 16)

most_repeated_phrases_discography = most_repeated_phrases_discography(df_lyrics_discography, 10)

"""
#score the repeats
#to test different scoring methods
#unscaled_scores_test = score_repeats_test(most_repeated_phrases, 10)
#unscaled_scores_compact = unscaled_scores_test[['Artist', 'Song', 'Top Phrase A', 'Top Phrase B', 'Top Phrase C', 'Top Phrase Score A']]
"""
#when I know which one I want to use
unscaled_scores = score_repeats(most_repeated_phrases, 15)

#Create a new dataframe that includes:
#artist, title, number of words, top phrase, length of top phrase, number of repetitions, max phrase of length x + 1, number of repetitions, max score
#this will help me look at how good the scoring methods are 
unscaled_scores['Top Phrase Plus One'] = np.nan
unscaled_scores['Top Phrase Plus One Number of Repeats'] = np.nan    

for index, row in unscaled_scores.iterrows():
      unscaled_scores.loc[index, 'Top Phrase Plus One'] = unscaled_scores.loc[index, 'Most Common Phrase - Length ' + str(int(unscaled_scores.loc[index, 'Top Phrase Length'] + 1))]
      unscaled_scores.loc[index, 'Top Phrase Plus One Number of Repeats'] = unscaled_scores.loc[index, 'Number of Repeats - Length ' + str(int(unscaled_scores.loc[index, 'Top Phrase Length'] + 1))]
    
unscaled_scores_compact = unscaled_scores[['Artist', 'Song', 'Number of Words', 'Top Phrase', 'Top Phrase Length', 'Top Phrase Number of Repeats', 'Top Phrase Plus One', 'Top Phrase Plus One Number of Repeats', 'Top Phrase Score']]


"""
