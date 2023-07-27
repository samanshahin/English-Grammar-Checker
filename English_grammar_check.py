import spacy    #pip install spacy (approx. 20 MB)
from spacy.matcher import phrasematcher
import contextualSpellCheck as csc  #pip install contextualSpellCheck (approx. 10 MB)
import pandas as pnd
import numpy as np
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

nlp = spacy.load('en_core_web_lg')  # it's 700+ MB
csc.add_to_pipe(nlp)


    doc = nlp('the text should be here.')
    word_count_whole = []
    verbRawList = []
    paragraphs = []
    parag = []
    newdoc = []
    for w in doc:
        # change "n't" to " not"
        if (w.endswith('n\'t')):
            ww = doc.index(w)
            doc[ww] = w.strip('n\'t') + ' not'
        if (w == 'won\'t'):
            w = doc.index(w)
            doc[ww] = 'will not'
    lastindex = 0
    for w in doc:
        word_count_whole.append(w)
        vocabScore(w)
        if (w == '\n'):
            newindex = doc.index(w)
            for j in range(lastindex, newindex):
                parag.append(doc[j])
            p = ' '.join(parag)
            paragraphs.append(p)
            parag.clear()
            lastindex += newindex
        if ((w.pos_ == 'VERB') or (w in ['not'])):
            verbRawList.append(w)
        if ((w.pos_ != 'VERB') or (w not in ['not'])):
            verbRawList.append('0')
        if ((w.pos_ == 'ADV') or (w.dep_ == 'advmod')):
            newdoc.append('<b class="highlighted">' + w + '</b>')
        else:
            newdoc.append(w)
    nadjCheck()
    wlen = len(word_count_whole)

    print(f"Word count should be between 250-300 words.\nword_count_whole : {wlen}")
    if ((wlen not in range(250, 300)) or (wlen < 250)):
        print('But it\'s NOT in the range or less than the minimum and it may be considered as a weakness.')

    sentsList = []  # check this for simplicity: sentsList = doc.sents
    word_count_sents = {}
    for s in doc.sents:
        sentsList.append(s)
        word_count_sents[s] = len(s)
        adjOrder(s)
        check_pronoun(s)
        detCheck(s)
        i = sentsList.index(s)
        if (i >= 1):
            pronRef(i)
        checkPunct(s)
        subjVerbCheck(s)
        wordCombo(s)
        objpassivVerbCheck(s, i)
        prepverbCheck(s)

    sentlen = len(sentsList)
    print(
        f"The number of sentences is: {sentlen} \n- Consider that it should be between 12-20.")
    if ((sentlen not in range(12, 20)) or (sentlen < 12)):
        print('But it\'s NOT in the range or less than the minimum and it may be considered as a weakness.')

    if(len(paragraphs) < 3):
        print(f'The essay should contain at least 3 paragraphs, but it\'s less than that ({len(paragraphs)} paragraphs) in this essay. The student should consider the punctuation or break down (split) the essay to make more paragraphs.')

    entityList = []
    for ent in doc.ents:
        entityList.append(ent)
    print(
        f"The number of NERs used in the essay is {len(entityList)}\nThe List includes:")
    entityList.insert(0, 'NERs in the essay:')
    print(*entityList,sep='\n- ')


    if(doc._.contextual_spellCheck and doc._.performed_spellCheck):
        print('SpellCheck is done. Here\'s the result:')
    if(len(doc._.suggestions_spellCheck) > 1):
        print('There are some errors! here\'s the list of wrong words and their correct case:')
        for d in doc._.suggestions_spellCheck.keys():
            print(f"- {d} >> {doc._.suggestions_spellCheck[d]}")
    #consider this:
    #print('The essay should be like this:')
    #print(doc._.outcome_spellCheck)


    # vocabs levels list (academic + score)
    vocab_scored = []
    vocab_score = 0
    f = open('level2_vocab.csv', newline='')
    rdrr = csv.reader(f)
    d = list(rdrr)
    ff = open('level3_vocab.csv', newline='')
    rdr = csv.reader(ff)
    dd = list(rdr)
    def vocabScore(ww):
        # one word matching
        if (str(ww) in d[0] or ww.lemma_ in d[0]):
            vocab_score += 1
            vocab_scored.append(ww)
        if (str(ww) in dd[0] or ww.lemma_ in dd[0]):
            vocab_score += 3
            vocab_scored.append(ww)
        # multi-words matching - search patterns in doc
    print(f"The student's score in Vocabulary is: {vocab_score} .")
    print(f"The Vocabulary list includes:\n{vocab_scored} .")


    # timeline of the essay by examining verbs
    verbPast = []
    verbPres = []
    tmpvList = []
    verbListNew = []
    for n in range(len(verbRawList)):
        if (str(verbRawList[n]) != '0'):
            tmpvList.append(verbRawList[n])
        if (str(verbRawList[n]) == '0'):
            ttmp = ' '.join(tmpvList)
            verbListNew.append(ttmp)
            tmpvList.clear()
        
    tok = 0
    for i in range(len(verbListNew)):
        v = str(verbListNew[i])
        lll = len(verbListNew[i])
        if (lll == 1):
            if (nlp.vocab.morphology.tag_map[nlp(str(verbListNew[i]))[0].tag_]['Tense_past']):
                verbPast.append(v)
            if (nlp.vocab.morphology.tag_map[nlp(str(verbListNew[i]))[0].tag_]['Tense_pres']):
                verbPres.append(v)
        if (lll == 2):
            if (nlp.vocab.morphology.tag_map[nlp(str(verbListNew[i]))[0].tag_]['VerbType_mod'] and nlp.vocab.morphology.tag_map[nlp(str(verbListNew[i]))[1].tag_]['Tense_pres'] and nlp.vocab.morphology.tag_map[nlp(str(verbListNew[i]))[1].tag_]['Aspect_prog']):
                verbPres.append(v)
            if (nlp.vocab.morphology.tag_map[nlp(str(verbListNew[i]))[0].tag_]['VerbType_mod'] and nlp.vocab.morphology.tag_map[nlp(str(verbListNew[i]))[1].tag_]['VerbForm_inf']):
                verbPres.append(v)
            if (nlp.vocab.morphology.tag_map[nlp(str(verbListNew[i]))[0].tag_]['VerbType_mod'] and nlp.vocab.morphology.tag_map[nlp(str(verbListNew[i]))[1].tag_]['VerbForm_part']):
                verbPast.append(v)
                for u in v:
                    if (u not in ['have', 'has', 'had']):
                        notOrderedWell += 1
        if (lll == 3):
            if (nlp.vocab.morphology.tag_map[v[0].tag_]['VerbType_mod'] and nlp.vocab.morphology.tag_map[v[1].tag_]['VerbForm_part'] and nlp.vocab.morphology.tag_map[v[2].tag_]['Tense_pres'] and nlp.vocab.morphology.tag_map[v[2].tag_]['Aspect_prog']):
                verbPast.append(v)
    verbTimeline = {}
    for past in verbPast:
        for o in verbListNew:
            if (past == o):
                verbTimeline[past] = verbListNew.index(o)
    verbTimeline['present'] = 'present'
    for pres in verbPres:
        for q in verbListNew:
            if (past == q):
                verbTimeline[pres] = verbListNew.index(q)
    dates = []
    for i in verbTimeline.keys():
        if (i == 'present'):
            dates.append('present')
        dates.append('event')
    levels = np.tile([-5, 5, -3, 3, -1, 1], int(np.ceil(len(dates)/6)))[:len(dates)]
    fig, ax = plt.subplots(figsize=(8.8, 4), constrained_layout=True)
    ax.set(title="Timeline of verbs in this essay")
    markerline, stemline, baseline = ax.stem(dates, levels, linefmt="C3-", basefmt="k-", use_line_collection=True)
    plt.setp(markerline, mec="k", mfc="w", zorder=3)
    markerline.set_ydata(np.zeros(len(dates)))
    vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
    for d, l, r, va in zip(dates, levels, verbTimeline, vert):
        ax.annotate(r, xy=(d, l), xytext=(-3, np.sign(l)*3), textcoords="offset points", va=va, ha="right")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.get_yaxis().set_visible(False)
    for spine in ["left", "top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.margins(y=0.1)
    plt.savefig("timeline.png", dpi=150)



    # adj order: det - quantity - quality - size - shape - age - color - nationality - purpose (n) + noun
    def adjOrder(snt):
        s = snt.split()
        adjOrderedCorrect = 0
        natCat = pnd.read_csv('csv/natCat.csv')
        natCat = natCat.transpose()
        colorCat = pnd.read_csv('csv/colorCat.csv')
        colorCat = colorCat.transpose()
        ageCat = pnd.read_csv('csv/ageCat.csv')
        ageCat = ageCat.transpose()
        shapeCat = pnd.read_csv('csv/shapeCat.csv')
        shapeCat = shapeCat.transpose()
        qualCat = pnd.read_csv('csv/qualCat.csv')
        qualCat = qualCat.transpose()
        adjsent = []
        adjtemp = []
        adjOrderList = []
        for k in range(len(s)):     # also: range(0, len(s))
            if (s[k].pos_ == 'ADJ'):
                adjsent.append(s[k])
            else:
                adjsent.append('-')
        for i in adjsent:
            if (i != '-'):
                adjtemp.append(i)
            if (i == '-'):
                if (len(adjtemp) > 1):
                    word = ' '.join(adjtemp)
                    adjOrderList.append(word)
                adjtemp.clear()
        print('start checking adj order:')
        for adjj in str(adjOrderList):
            print(f"Checking order of adjectives in \'{adjj}\':")
            adj = adjj.split()
            adj.reverse()
            nat = color = age = shape = qual = 0
            for ii in range(len(adj)):
                if (adj[ii] in natCat) and (nat == 0):
                    adjOrderedCorrect += 1
                    nat = 1
                if (adj[ii] in colorCat) and (color == 0):
                    adjOrderedCorrect += 1
                    color = 1
                if (adj[ii] in ageCat) and (age == 0):
                    adjOrderedCorrect += 1
                    age = 1
                if (adj[ii] in shapeCat) and (shape == 0):
                    adjOrderedCorrect += 1
                    shape
                if (adj[ii] in qualCat) and (qual == 0):
                    adjOrderedCorrect += 1
                    qual = 1
            if (adjOrderedCorrect <= 1):
                print('This adjective combination is NOT ordered correctly:\n' + adj)


    def check_pronoun(snt):
        s = snt.split()
        pronList = []
        pronindList = []
        notCheckedPron = 0
        verbSing = nsubjSingle = True
        for n in range(len(s)):
            w = s[n]
            if (w.dep_ == 'nsubj'):
                pronList.append(w)
                pronindList.append(n)
            if ((w.pos_ == 'VERB') and (nlp.vocab.morphology.tag_map[w.tag_]['VerbType_mod'] != True) and (nlp.vocab.morphology.tag_map[w.tag_]['Number_sing']) and (nlp.vocab.morphology.tag_map[w.tag_]['Number_sing'] != True)):
                verbSing = False
        if (len(pronList) > 1):
            if (s[pronList[1]] == 'and'):
                nsubjSingle = False
        if (len(pronList) == 1):
            if ((pronList[0].dep_ == 'nsubj') and (nlp.vocab.morphology.tag_map[pronList[0].tag_]['Number_sing']) and (nlp.vocab.morphology.tag_map[pronList[0].tag_]['Number_sing'] == True)):
                nsubjSingle = True
            if ((verbSing != True) and (nsubjSingle)):
                notCheckedPron += 1
        if (len(pronList) == 0):
            notCheckedPron = 0
        for n in range(len(pronList)):
            curpronsingstatus = nlp.vocab.morphology.tag_map[pronList[n].tag_]['Number_sing']
            nextpronsingstatus = nlp.vocab.morphology.tag_map[pronList[n+1].tag_]['Number_sing']
            if (curpronsingstatus and nextpronsingstatus and (curpronsingstatus != nextpronsingstatus)):
                notCheckedPron += 1
        if (notCheckedPron > 1):
            print(f'there may be a problem in pronoun-verb agreement in this essay. Here\'s the sentence:\n{s}')


    def subjVerbCheck(snt):
        s = snt.split()
        print('subject-verb agreement:')
        verbSing = True
        subjSingle = True
        def checksubjVerb():
            if ((subjSingle == True and verbSing == False) or (subjSingle == False and verbSing == True)):
                print(f'there may be a problem in subject-verb agreement in this essay. Here\'s the sentence:\n{s}')
        for n in range(len(s)):
            w = s[n]
            if ((w.dep_ == 'subj') and (nlp.vocab.morphology.tag_map[w.tag_]['Number_sing']) and (nlp.vocab.morphology.tag_map[w.tag_]['Number_sing'] != True)):
                sss = s.index(w)
                if (s[sss+1] == 'and' and s[sss+2].pos_ == 'noun'):
                    subjSingle = False
            if ((w.pos_ == 'VERB') and (nlp.vocab.morphology.tag_map[w.tag_]['VerbType_mod'] != True) and (nlp.vocab.morphology.tag_map[w.tag_]['Number_sing']) and (nlp.vocab.morphology.tag_map[w.tag_]['Number_sing'] != True)):
                verbSing = False
            checksubjVerb()

            

    def detCheck(snt):
        s = snt.split()
        notOrderedWell = 0
        for k in range(len(s)):
            if (s[k].pos_ == "DET"):
                if ((s[k+1].pos_ == 'VERB') or (s[k+1].pos_ == 'DET')):
                    notOrderedWell += 1
        if (notOrderedWell >= 1):
            print(f'one or more error case(s) has been detected; i.e. a verb follows a determine. It may be of misspelled error. Here\'s the sentence:\n{s}')



    def pronRef(i):
        curSent = sentsList[i].split()
        prevSent = sentsList[i-1].split()
        curSentList = []
        prevSentList = []
        hasPassiveVerb = False
        subjtotal = objtotal = 0
        notmatched = 0
        for n in curSent:
            if ((n.dep_ == 'subj') and (nlp.vocab.morphology.tag_map[w.tag_]['Number_sing']) and (nlp.vocab.morphology.tag_map[w.tag_]['Number_sing'] == True)):
                curSentList.append([n, 'subj', 'sing'])
            if ((n.dep_ == 'subj') and (nlp.vocab.morphology.tag_map[w.tag_]['Number_sing']) and (nlp.vocab.morphology.tag_map[w.tag_]['Number_sing'] != True)):
                curSentList.append([n, 'subj', 'plu'])
            if ((n.dep_ == 'obj') and (nlp.vocab.morphology.tag_map[w.tag_]['Number_sing']) and (nlp.vocab.morphology.tag_map[w.tag_]['Number_sing'] == True)):
                curSentList.append([n, 'obj', 'sing'])
            if ((n.dep_ == 'obj') and (nlp.vocab.morphology.tag_map[w.tag_]['Number_sing']) and (nlp.vocab.morphology.tag_map[w.tag_]['Number_sing'] != True)):
                curSentList.append([n, 'obj', 'plu'])
        for n in prevSent:
            if ((n.dep_ == 'subj') and (nlp.vocab.morphology.tag_map[w.tag_]['Number_sing']) and (nlp.vocab.morphology.tag_map[w.tag_]['Number_sing'] == True)):
                prevSentList.append([n, 'subj', 'sing'])
            if ((n.dep_ == 'subj') and (nlp.vocab.morphology.tag_map[w.tag_]['Number_sing']) and (nlp.vocab.morphology.tag_map[w.tag_]['Number_sing'] != True)):
                prevSentList.append([n, 'subj', 'plu'])
            if ((n.dep_ == 'obj') and (nlp.vocab.morphology.tag_map[w.tag_]['Number_sing']) and (nlp.vocab.morphology.tag_map[w.tag_]['Number_sing'] == True)):
                prevSentList.append([n, 'obj', 'sing'])
            if ((n.dep_ == 'obj') and (nlp.vocab.morphology.tag_map[w.tag_]['Number_sing']) and (nlp.vocab.morphology.tag_map[w.tag_]['Number_sing'] != True)):
                prevSentList.append([n, 'obj', 'plu'])
        if (i in passiveVerbsList):
            hasPassiveVerb = True
            passiveVerb = passiveVerbsList[i]
        if (hasPassiveVerb):
            if ('subj' in curSentList):
                subjtotal += 1
            if ('obj' in curSentList):
                objtotal += 1
            if (subjtotal > 1 or objtotal > 1):
                print('you may write your essay more simple.')
            for c in curSentList:
                for p in prevSentList:
                    if (not (('sing' in p) and ('subj' in p)) and not (('sing' in c) and ('obj' in c))):
                        print('not matched')
                        notmatched += 1
                    if (not (('plu' in p) and ('subj' in p)) and not (('plu' in c) and ('obj' in c))):
                        print('not matched')
                        notmatched += 1
            if notmatched == 0:
                print('matched')
            else:
                print('pronouns in this sentence may NOT refer to the equivalent in previous sentences. NOT MATCHED')



    def nadjCheck():
        nadjList = []
        tmpList = []
        for i in word_count_whole:
            if (i.pos_ == 'NOUN'):    # or SUBJ ???
                tmpList.append(i)
            else:
                if len(tmpList) > 1:
                    wtmp = ' '.join(tmpList)
                    nadjList.append(wtmp)
                tmpList.clear()
        print('There are cases that nouns sequenced in such an order that might not be correct. Here\'s the list:')
        print(*nadjList, sep='\n- ')



    print('There are adverbs in the essay. here\'s the essay with adverbs are highlited in it:')
    print(' '.join(newdoc)) # containing html tags, must be converted



    # make a dictionary of verb-prep (vprepfile.csv - a table in excel with 5 cols and sep col), then check in each sentence
    def prepverbCheck(snt):
        s = snt.split()
        vprepfile = open('vprepfile.csv', newline='')
        vp = csv.reader(vprepfile)
        verbsList = list(vp)
        print('Checking the verb-prep combination in the essay:')
        matched = 0
        for i in range(len(s)):
            if ((s[i].pos_ == 'VERB') and (nlp.vocab.morphology.tag_map[s[i].tag_]['VerbType_mod'] != True)):
                if (s[i].lemma_ in verbsList):
                    tmpverb = s[i]
                    tmpprep = verbsList[tmpverb]
                    for n in range(i, len(s)):
                        if (s[n].pos_ == 'PREP') and (s[n] == tmpprep):  #or its tag_
                            matched = 1
                        else:
                            matched = 2
            if (matched == 1):
                print(f'seems MATCHED in sentence #{i}')            
            if (matched == 2):
                print(f'seems NOT MATCHED in sentence #{i}. Please check the sentence:\n{s}')
        # check for 'seperation' criteria, must be seperated and by what element


    passiveVerbsList = []
    def objpassivVerbCheck(snt, j):
        jjj = j
        s = snt.split()
        print('Checking if there\'s any incorrect usage of objective(s) after a passive verb:')
        tmpvl1 = ['is', 'are', 'was', 'were']
        tmpvl2 = ['has', 'have', 'had']
        notmatched = 0
        tmppasslist = []
        for i in range(len(s)):
            if (s[i].lemma_ in tmpvl1):
                if ((s[i+1] not in ['not']) and (nlp.vocab.morphology.tag_map[s[i+1].tag_]['Aspect_perf']) and (nlp.vocab.morphology.tag_map[s[i+1].tag_]['Tense_past']) and (nlp.vocab.morphology.tag_map[s[i+1].tag_]['VerbForm_part'])):
                    tmppasslist.append(s[i])
                    tmppasslist.append(s[i+1])
                    for n in range(i+2,len(s)):
                        if (s[n].pos_ == 'nobj'):
                            notmatched = 1
                        if ((s[n] in ['and', 'that']) or (s[n].pos_ == 'VERB')):
                            break
                if ((s[i+1] in ['not']) and (nlp.vocab.morphology.tag_map[s[i+2].tag_]['Aspect_perf']) and (nlp.vocab.morphology.tag_map[s[i+2].tag_]['Tense_past']) and (nlp.vocab.morphology.tag_map[s[i+2].tag_]['VerbForm_part'])):
                    tmppasslist.append(s[i])
                    tmppasslist.append(s[i+1])
                    tmppasslist.append(s[i+2])
                    for n in range(i+3,len(s)):
                        if (s[n].pos_ == 'nobj'):
                            notmatched = 1
                        if ((s[n] in ['and', 'that']) or (s[n].pos_ == 'VERB')):
                            break
            if ((s[i] in tmpvl2)):
                if ((s[i+1] not in ['not']) and (nlp.vocab.morphology.tag_map[s[i+1].tag_]['Aspect_perf']) and (nlp.vocab.morphology.tag_map[s[i+1].tag_]['Tense_past']) and (nlp.vocab.morphology.tag_map[s[i+1].tag_]['VerbForm_part'])):
                    if (((nlp.vocab.morphology.tag_map[s[i+2].tag_]['Aspect_perf']) and (nlp.vocab.morphology.tag_map[s[i+2].tag_]['Tense_past']) and (nlp.vocab.morphology.tag_map[s[i+2].tag_]['VerbForm_part']))):
                        tmppasslist.append(s[i])
                        tmppasslist.append(s[i+1])
                        tmppasslist.append(s[i+2])
                        for n in range(i+2,len(s)):
                            if ((s[n].pos_ == 'nobj')):
                                notmatched = 1
                            if ((s[n] in ['and', 'that']) or (s[n].pos_ == 'VERB')):
                                break
                if ((s[i+1] in ['not']) and (nlp.vocab.morphology.tag_map[s[i+2].tag_]['Aspect_perf']) and (nlp.vocab.morphology.tag_map[s[i+2].tag_]['Tense_past']) and (nlp.vocab.morphology.tag_map[s[i+2].tag_]['VerbForm_part'])):
                    if (((nlp.vocab.morphology.tag_map[s[i+3].tag_]['Aspect_perf']) and (nlp.vocab.morphology.tag_map[s[i+3].tag_]['Tense_past']) and (nlp.vocab.morphology.tag_map[s[i+3].tag_]['VerbForm_part']))):
                        tmppasslist.append(s[i])
                        tmppasslist.append(s[i+1])
                        tmppasslist.append(s[i+2])
                        tmppasslist.append(s[i+3])
                        for n in range(i+4,len(s)):
                            if ((s[n].pos_ == 'nobj')):
                                notmatched = 1
                            if ((s[n] in ['and', 'that']) or (s[n].pos_ == 'VERB')):
                                break
            p = ' '.join(tmppasslist)
            passiveVerbsList.append([jjj, p])
            if (notmatched == 0):    
                print(f'found a passive structure ({p}), but seems HAVE NO PROBLEM in the sentence.')            
            if (notmatched == 1):
                print(f'found a case ({p}) and seems WE HAVE A PROBLEM in this sentence. Please check it out:\n{s}')
            tmppasslist.clear()
        print('nothing more...')



    def checkPunct(snt):
        s = snt.split()
        print('Checking punctuation:')  # it should be outside the function, specifying a section for checking the punctuation
        w = s[0]
        puntchecked = 0
        errs = []
        if (w[0].islower() == True):
            puntchecked += 1
            errs.append('First letter of the first word in this sentence should be UPPERCASE!')
        for i in range(len(s)):
            if (s[i] == '...' and s[i-1] in ['and', 'or']):
                puntchecked += 1
                errs.append(f'there should not be used \'...\' instead of \'etc.\' in the sentence. Please check it out:\n{s}')
            if (s[i] == '...' and s[i-1] not in ['and', 'or']):
                puntchecked += 1
                errs.append(f'meaningless use of \'...\' in the sentence. Please check it out:\n{s}')
            if (s[i] == ',' and (s[i+1][0].islower() == False)):
                puntchecked += 1
                errs.append('It seems to be a complex sentence with origin and subordinate parts. Then, first letter of the first word after comma should not be uppercase')
            if (s[i] == ';' or s[i] == '.' and (s[i+1][0].islower() == False)):
                puntchecked += 1
                errs.append('wrong punctuation')
            if (s[i] == '...' and s[i+1] in ['and', 'or']):
                puntchecked += 1
                errs.append('there should not be used \'and\' and \'or\' after \'...\'.')
        if (puntchecked >= 1):
            print('Here is the list of punctuation error(s) in the essay:')
            print(*errs, sep='\n- ')



    # word combo / word choice - make a list of high score combinations (wordcombo.csv)
    def wordCombo(snt):
        s = snt.split()
        wcombofile = open('wordcombo.csv', newline='')
        wc = csv.reader(wcombofile)
        wcList = list(wc)
        print('word combo + Vocab score')
        # must make matcher: https://spacy.io/usage/rule-based-matching
        # Phrase matching
        matcher = phrasematcher(nlp.vocal)
        phrase_patterns = [nlp(text) for text in wcList]	# convertnig each phrase to pattern
        matcher.add('a_name', none, *phrase_patterns)	# add a_name to the matcher, note the use of asterisk!
        found_matches = matcher(s)	# pass the doc for matching and build a list of matches
        print(len(found_matches))
        vocab_score += len(found_matches)*0.5
        for match_id, start, end in found_matches:
            string_id = nlp.vocab.strings[match_id]		# get string representation
            span = s[start:end]				        # get the matched span or for sorounding texts: [start-5:end+5]
            print(match_id, string_id, start, end, span.text)
    
