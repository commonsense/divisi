'''
Utility functions for dealing with PyTables files.

'''
import tables
import os.path

# Connection pool
pytfiles = {}

def get_pyt_handle(filename, title=''):
    filename = os.path.abspath(filename)
    if filename not in pytfiles or not pytfiles[filename].isopen:
        pytfiles[filename] = tables.openFile(filename, 'a', title)
    return pytfiles[filename]
