# English-Grammar-Checker
It's an AI-powered NLP (Natural Language Processsing) app to check english grammar. It was developed before [Grammarly](https://www.grammarly.com/) being registered, but unfortunately, the project was dead-end due to lack of financial support.
It should be noted that it's not comprehensive, as there are so many websites and tools out there, but should be known as an effort to simulate the grammar checker...

## Sections
### loading prerequisites
there are some packages that should be loaded before:
- spacy
- contextualSpellCheck as csc  #pip install contextualSpellCheck (approx. 10 MB)
- pandas and numpy and csv
- sklearn and matplotlib
after loading the packages, we load our language package from spacy library ('en_core_web_lg'). It's about 700 MB and as you deployed your application, it would be loaded once and that's it.

### statistics
This section includes:
- counting words, sentences, NERs
- spell checking
- scoring based on use of verbs/vocabs
- timeline of the essay and plotting it on a chart
- checking order of adjectives. as it should be like this:
  det - quantity - quality - size - shape - age - color - nationality - purpose (n) + noun
- cheking pronouns and their accordance to verbs
- chekcing subject-verb acceptance
- checking whether determinants are used correctly or not
- checking pronouns refer correctly
- checking noun adjectives
- checking phrasal verbs with correct prepositions
- checking object and passive verbs accordance
- checking punctuation
- chekcing two parts verbs

we excluded the topic modeling and reasoning because it's in developping phase and not ready to be published
