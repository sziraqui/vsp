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
    array([15, 20, 19])
    >>> word2ints('place')
    array([15, 11,  0,  2,  4])
    >>> word2ints('eight')
    array([ 4,  8,  6,  7, 19])
    """

def word2binmat_test():
    """
    >>> np.argmax(word2binmat(6,7,'at'), axis=1)
    array([ 0, 19])
    >>> np.argmax(word2binmat(0,13,'foo'), axis=1)
    array([ 5,  5,  5,  5, 14, 14, 14, 14, 14, 14, 14, 14, 26, 26])
    >>> np.argmax(word2binmat(0,4,None), axis=1)
    array([27, 27, 27, 27, 27])
    """

def min_repetitions_test():
    """
    >>> min_repetitions(list('at'))
    1
    >>> min_repetitions(list('aaaaatttttt'))
    5
    >>> min_repetitions(list('fffooooooooo'))
    3
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

if __name__ == '__main__':
    import doctest
    doctest.testmod()