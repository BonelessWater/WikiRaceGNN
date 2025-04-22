# test_imports.py
try:
    import gensim
    print(f"Gensim found! Version: {gensim.__version__}")
except ImportError as e:
    print(f"Error importing gensim: {e}")

import sys
print(f"Python path: {sys.path}")