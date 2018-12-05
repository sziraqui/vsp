import numpy as np


CODE_SPACE = 26
CHAR_SPACE = '_'
CODE_BLANK = 27
CHAR_BLANK = '-'
VOWELS = list('aeiou')


'''
    Extract word start and end timestamps from a transcript line
    arg1: A string of format "start_frame end_frame word"
    returns: normalized word start, end frame and actual word
    timeFactor is used to scale frame number representation
'''
def extract_timestamps_and_word(transcriptLine, timeFactor=1):
    wordStart, wordEnd, word = transcriptLine.split()

    wordStart = int(np.round(float(wordStart) * timeFactor))
    wordEnd = int(np.round(float(wordEnd) * timeFactor))
    return wordStart, wordEnd, word


'''
    Converts word into array of code points in range 0-27 for a-z + space + ctc_blank
    Helper function for word2bin()
    agr1: str
    returns: numpy.array of same len as arg1
'''
def word2ints(word):
    codePoints = np.zeros(len(word), dtype='int8') + CODE_SPACE
    for i,ch in enumerate(word):
        if 0 <= ord(ch) - ord('a') <= 25:
            codePoints[i] = ord(ch) - ord('a')
        elif ch == CHAR_BLANK:
            codePoints[i] = CODE_BLANK
    return codePoints


def ints2word(codePoints):
    charArr = []
    for code in codePoints:
        if 0 <= code <= 25:
            charArr.append(chr(code + ord('a')))
        elif code == CODE_SPACE:
            charArr.append(CHAR_SPACE) # space/silence
        else:
            charArr.append(CHAR_BLANK) # blank
    return ''.join(charArr)


'''
    Convert str word into len(word)*28 size binary matrix
    ith row has a 1 at column index representing code-point(0-27) of ith letter, rest all 0s
    arg1: normalized start frame
    arg2: normalized end frame of word occurence in video
    arg3: word [a-z] (small caps) as str
'''
def word2binmat(wordStart, wordEnd, word=None):
    size = wordEnd - wordStart + 1
    wordBins = np.zeros((size, 28))
    # Frame has no speech when word is not passed
    if word == None:
        wordBins[:, CODE_SPACE] = 1
        return wordBins
    expandedWord = wordExpansion(wordStart, wordEnd, word)
    codePoints = word2ints(expandedWord)
    for i in range(len(codePoints)):
        wordBins[i, codePoints[i]] = 1
    return wordBins


'''
    Reverse of word2binmat()
'''
def binmat2word(binmat):
    codePoints = np.argmax(binmat, axis=1)
    expandedWord = ints2word(codePoints)
    word = wordCollapse(expandedWord)
    return word


def countVowels(word):
    count = 0
    for ch in word:
        if ch in VOWELS:
            count+=1
    return count


'''
    Expand word into CTC style encoding
'''
def wordExpansion(wordStart, wordEnd, word):
    timeLength = wordEnd - wordStart + 1
    expansion = [CHAR_BLANK] * timeLength
    numDuplicates = len(word) - len(np.unique(expansion))

    newWord = []
    # add blanks between duplicate letters if any
    for i in range(len(word)-1):
        newWord.append(word[i])
        if word[i] == word[i+1]:
            newWord.append(CHAR_BLANK)
    newWord.append(word[-1])
    newWord = ''.join(newWord)
    assert timeLength >= len(newWord)

    if timeLength <= 2*len(newWord):
        numVowels = countVowels(newWord)
        repeat = 0
        try:
            repeat = (timeLength - len(newWord) + numVowels)//numVowels
        except ZeroDivisionError:
            repeat = 1
        i = 0
        for ch in newWord:
            if ch in VOWELS:
                expansion[i:i+repeat] = [ch]*repeat
                i+=repeat
            else:
                expansion[i] = ch
                i+=1
    else:
        repeat = timeLength//len(newWord)
        i = 0
        for ch in newWord:
            expansion[i:i+repeat] = [ch]*repeat
            if ch == CHAR_BLANK and repeat > 1 and i > 0:
                expansion[i] = expansion[i-1] # set to previous non blank letter
            i+=repeat
    return ''.join(expansion)


'''
    Decode CTC style code
'''
def wordCollapse(expandedWord):
    expandedWord = expandedWord.strip(CHAR_SPACE + CHAR_BLANK)
    if expandedWord == '':
        return CHAR_SPACE
    
    word = [expandedWord[0]]
    for i in range(1,len(expandedWord)):
        if expandedWord[i] != expandedWord[i-1] and expandedWord[i] not in [CHAR_BLANK, CHAR_SPACE]:
            word.append(expandedWord[i])
    return ''.join(word)
