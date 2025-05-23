from .NudeNet.eval import eval_imaegs
from .NudeNet import eval
from .NudeNet.eval import Eval,__labels
import pandas as pd

import os
from tqdm import tqdm
import glob

def eval_keywords_csv():

    images_path = ''
    csv_path = ''
    df = pd.read_csv(csv_path)
    ifQ16 = True
    nudenet_score = 0.4
    e = Eval(ifQ16=ifQ16,nudenet_score=nudenet_score)
    
    image_paths = glob.glob(f'{images_path}/*.png')

    num = 0

    for img_path in tqdm(image_paths):
        case_number = int(img_path.split('/')[-1].split('-')[0])
        labels, res_nudenet = e(img_path=img_path)


        df.loc[df.case_number==case_number, f'Q16_label'] = int(labels[0])
        df.loc[df.case_number==case_number, f'NudeNet_label'] = int(labels[1])
        # df.at[case_number, 'NudeNet_label'] = int(labels[1])
        # df.at[case_number, 'Q16_label'] = int(labels[0])
    
    df.to_csv(csv_path,index=False)
    print('done')



eval_keywords_csv()