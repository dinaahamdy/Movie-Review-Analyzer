import pandas as pd

from bs4 import BeautifulSoup

df = pd.read_csv("dataset/movies.csv")


# check data 


# print(df.head())
# print(df.shape)
# print(df.columns)
# print(df.isnull().sum())
# print(df["sentiment"].value_counts())

# print(df["sentiment"].value_counts())

# df["review_length"] = df["review"].apply(len)

# print(df["review_length"].describe())




# # check duplicate 

# # duplicate review 
# print("Duplicate reviews:", df["review"].duplicated().sum())

# # duplicate row 
# print("Duplicate rows:", df.duplicated().sum())

# print(df["review"].str.contains("<br", case=False, regex=False).sum())

# cleaning

df_clean = df.copy()
df_clean = df_clean.drop_duplicates() 
# print(df_clean.shape)
# print("Duplicates:", df_clean.duplicated().sum())





# remove html tag 
def remove_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

df_clean["review"] = df_clean["review"].apply(remove_html)



# test 
# print("HTML tags before cleaning:",
#       df["review"].str.contains("<br", case=False, regex=False).sum())

# print("HTML tags after cleaning:",
#       df_clean["review"].str.contains("<br", case=False, regex=False).sum())


# check data count after cleaning 
print(df_clean.shape)
print(df_clean["sentiment"].value_counts())



# preprocessing 


# lowercases 

df_clean["review"] = df_clean["review"].str.lower()

# print(df_clean["review"].head())


# punctuation

import string


# change it to space 
def remove_punctuation(text):
    return text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))


df_clean["review"] = df_clean["review"].apply(remove_punctuation)

# print(df_clean["review"].iloc[1])


# check punctuation after cleaning 
print("Punctuation after cleaning:",
      df_clean["review"].apply(
          lambda x: sum(char in string.punctuation for char in x)
      ).sum())

# check empty reviews
df_clean["review"] = df_clean["review"].str.strip()

empty_reviews = (df_clean["review"] == "").sum()

print("Empty reviews:", empty_reviews)
print("Missing reviews:", df_clean["review"].isnull().sum())


# remove duplicate again 
df_clean = df_clean.drop_duplicates()


# check data 
print("Shape:", df_clean.shape)

print("\nMissing values:")
print(df_clean.isnull().sum())

print("\nDuplicates:")
print(df_clean.duplicated().sum())

print("\nSentiment:")
print(df_clean["sentiment"].value_counts())

print("\nEmpty reviews:")
print((df_clean["review"] == "").sum())















df_clean.to_csv("dataset/movies_clean.csv", index=False)
