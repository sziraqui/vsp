import numpy as np


'''
    Extract word start and end timestamps from a transcript line
    arg1: A string of format "start_frame end_frame word"
    returns: normalized word start, end frame and acutual word
'''
def extract_timestamps_and_word(transcript_line):
    wordStart, wordEnd, word = transcript_line.split()

    wordStart = int(np.round(float(wordStart)))
    wordEnd = int(np.round(float(wordEnd)))
    return wordStart, wordEnd, word


'''
    Converts word into array of code points in range 0-27 for a-z + space + ctc_blank
    Helper function for word2bin_seq()
    agr1: str
    returns: numpy.array of same len as arg1
'''
def word2ints(word):
    codePoints = np.array(list(map(lambda x: ord(x) - ord('a'), word)))
    return codePoints


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
        wordBins[:, 27] = 1 # 27 denotes CTC blank
        return wordBins
    codePoints = word2ints(word)
    # Distribute word over the entire frame width
    repetitions = size//len(word)
    lastFill = 0
    for i in codePoints:
        wordBins[lastFill:(lastFill + repetitions), i] = 1 # ith index for ith letter of a-z
        lastFill+=repetitions
    # add space(s) at the end to mark word end
    wordBins[lastFill:size, :] = 0
    wordBins[lastFill:size, 26] = 1 # 26 denotes space
    return wordBins


'''
    Find minimum repeating sequence in array of chars
    Helper function for binmat2word
'''
def min_repetitions(charArr):
    from collections import defaultdict
    hist = defaultdict(int)
    for ch in charArr:
        hist[ch]+=1
    return min(hist.values())


'''
    Reverse of word2binmat()
'''
def binmat2word(binmat):
    codePoints = np.argmax(binmat, axis=1)
    charArr = list(map(lambda x: chr(ord('a') + x) if 0 <= x <= 25 else '_', codePoints))
    charArr = list(''.join(charArr).strip('_'))
    word = '' # blank
    size = len(charArr)
    repetition = min_repetitions(charArr)
    for i in np.arange(0,size, repetition):
        word += charArr[i]

    return word