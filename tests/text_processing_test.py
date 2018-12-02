#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.textprocessing import *

def extract_timestamps_and_word_test():
    """
    >>> extract_timestamps_and_word('0 7 bin')
    (0, 7, 'bin')
    >>> extract_timestamps_and_word('26 38 again')
    (26, 38, 'again')
    >>> extract_timestamps_and_word('12.5 27.0 continue')
    (12, 27, 'continue')
    """

def word2ints_test():
    """
    >>> word2ints('put')
    array([15, 20, 19], dtype=int8)
    >>> word2ints('place')
    array([15, 11,  0,  2,  4], dtype=int8)
    >>> word2ints('eight')
    array([ 4,  8,  6,  7, 19], dtype=int8)
    """

def word2binmat_test():
    """
    >>> np.argmax(word2binmat(6,7,'at'), axis=1)
    array([ 0, 19])
    >>> np.argmax(word2binmat(0,13,'foo'), axis=1)
    array([ 5,  5,  5, 14, 14, 14, 14, 27, 27, 14, 14, 14, 27, 27])
    >>> np.argmax(word2binmat(0,4,None), axis=1)
    array([26, 26, 26, 26, 26])
    """


def binmat2word_test():
    """
    >>> binmat2word(word2binmat(0,5,'foo')) 
    'foo'
    >>> binmat2word(word2binmat(0,13,'seven'))
    'seven'
    >>> binmat2word(word2binmat(2,7,'oops'))
    'oops'
    """


def wordExpansion_test():
    """
    >>> wordExpansion(0,5,'bin')
    'biiiin'
    >>> wordExpansion(5,17,'fallen')
    'faaaal-leeeen'
    >>> wordExpansion(9,15,'nine')
    'niinee-'
    >>> wordExpansion(0,0,'a')
    'a'
    >>> wordExpansion(0,3,'fall')
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
      File "/home/sziraqui/Documents/vsp-dev/modules/textprocessing.py", line 108, in wordExpansion
        assert timeLength >= len(newWord)
    AssertionError
    """


def wordCollapse_test():
    """
    >>> wordCollapse('biiin')
    'bin'
    >>> wordCollapse('niinne-')
    'nine'
    >>> wordCollapse('_-pp-uuut--')
    'put'
    >>> wordCollapse('faaal-ll--enn--')
    'fallen'
    """


if __name__ == '__main__':
    import doctest
    doctest.testmod()