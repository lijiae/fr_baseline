import os
import pandas as pd
import numpy as np
import argparse

class metric():
    def __init__(self,test_path,test_pre_path):
        self.test=pd.read_csv(test_path)
        self.test_pre=pd.read_csv(test_pre_path)
        
    