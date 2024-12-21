import shutil
import os
from pathlib import Path
import pandas as pd

df=pd.DataFrame(columns=['id','label'])
for f in os.listdir(r'A:\pycharm_projects\plates\data\result_test2\dirty'):
    df.loc[len(df)] = [f.split('.')[0],'dirty']
for f in os.listdir(r'A:\pycharm_projects\plates\data\result_test2\clean'):
    df.loc[len(df)] = [f.split('.')[0],'cleaned']

df.sort_values(by='id',inplace=True)

df.to_csv(r'A:\pycharm_projects\plates\data\result_test2\sample_submission.csv',index=False)