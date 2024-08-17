import spacy
from spacy.symbols import ORTH

# Section 1: Basic Tokenization
nlp = spacy.blank("en")
doc = nlp("Dr. Strange loves pav bhaji of Mumbai as it costs only 2$ per plate.")
print("Tokens in the sentence:")
for token in doc:
    print(token)

# Section 2: Token Attributes
print("\nToken attributes:")
for token in doc:
    print(f"{token.text} => is_alpha: {token.is_alpha}, like_num: {token.like_num}, is_currency: {token.is_currency}")

# Section 3: Extracting Email IDs
text = '''
Dayton high school, 8th grade students information
Name  birth day   email
Virat   5 June, 1882    virat@kohli.com
Maria  12 April, 2001  maria@sharapova.com
Serena  24 June, 1998   serena@williams.com 
Joe      1 May, 1997    joe@root.com
'''
doc = nlp(text)
emails = [token.text for token in doc if token.like_email]
print("\nExtracted emails:", emails)

# Section 4: Customizing Tokenizer
nlp.tokenizer.add_special_case("gimme", [{ORTH: "gim"}, {ORTH: "me"}])
doc = nlp("gimme double cheese extra large healthy pizza")
tokens = [token.text for token in doc]
print("\nCustom tokenizer tokens:", tokens)

# Section 5: Sentence Segmentation
nlp.add_pipe('sentencizer')
doc = nlp("Dr. Strange loves pav bhaji of Mumbai. Hulk loves chat of Delhi")
print("\nSentences in the text:")
for sentence in doc.sents:
    print(sentence)

# Section 6: Exercise - Extract URLs
text = '''
Look for data to help you address the question. Governments are good
sources because data from public research is often freely available. Good
places to start include http://www.data.gov/, and http://www.science.gov/, 
and in the United Kingdom, http://data.gov.uk/.
Two of my favorite data sets are the General Social Survey at http://www3.norc.org/gss+website/, 
and the European Social Survey at http://www.europeansocialsurvey.org/.
'''
doc = nlp(text)
urls = [token.text for token in doc if token.like_url]
print("\nExtracted URLs:", urls)

# Section 7: Exercise - Extract Money Transactions
transactions = "Tony gave two $ to Peter, Bruce gave 500 € to Steve"
doc = nlp(transactions)
money_transactions = []
for token in doc:
    if token.is_currency:
        prev_token = doc[token.i - 1]
        money_transactions.append(f"{prev_token} {token}")

print("\nExtracted money transactions:", money_transactions)
