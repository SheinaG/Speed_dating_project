import pathlib
import os
import numpy as np



if os.name == 'nt':
    BASE_DIR = pathlib.PurePath('V:\\AIMLab')
    REPO_DIR = pathlib.PurePath('C:\\Users\\ShanyBiton\\Documents\\repos\\Causal Inference\\FinalProject')
    REPO_DIR_POSIX = pathlib.PurePath(str(REPO_DIR).replace('\\', '/').replace('C:','/mnt/c'))
elif os.name == 'posix':
    BASE_DIR = pathlib.PurePath('/MLAIM/AIMLab')
    REPO_DIR = pathlib.PurePath('/home/shanybiton/repos/sde/FinalProject/')
    REPO_DIR_POSIX = REPO_DIR


# Paths definition
DATA_DIR = REPO_DIR / "Data"

# feature vectors
x = ['gender', 'age', 'field_cd', 'race', 'imprace', 'imprelig', 'country/continent',
     'date', 'go_out', 'career_c', 'sports', 'tvsports', 'exercise', 'dining', 'museums', 'art', 'hiking',
     'gaming', 'clubbing', 'reading', 'tv', 'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga', 'exphappy',
     'match_es', 'attr3_1', 'sinc3_1', 'intel3_1', 'fun3_1', 'amb3_1', 'attr1_1', 'sinc1_1', 'intel1_1',
     'fun1_1', 'amb1_1', 'shar1_1', 'attr', 'sinc', 'intel',
    'fun', 'amb', 'shar']

x_by_others = ['attr', 'sinc', 'intel', 'fun', 'amb', 'shar']
x_encoding = ['career_c', 'country/continent', 'race']
encoding_dict = {'career_c': {1: 'Lawyer',
                              2: 'Academic/Research',
                              3: 'Psychologist',
                              4: 'Doctor/Medicine',
                              5: 'Engineer',
                              6: 'Creative Arts/Entertainment',
                              7: 'Banking/Consulting/Finance/Marketing/Business/CEO/Entrepreneur/Admin',
                              8: 'Real Estate',
                              9: 'International/Humanitarian Affairs',
                              10: 'Undecided',
                              11: 'Social Work',
                              12: 'Speech Pathology',
                              13: 'Politics',
                              14: 'Pro sports/Athletics',
                              15: 'Other',
                              16: 'Journalism',
                              17: 'Architecture'},
                'race': {1: 'Black/African American',
                         2: 'European/Caucasian-American',
                         3: 'Latino/Hispanic American',
                         4: 'Asian/Pacific Islander/Asian-American',
                         5: 'Native American',
                         6: 'Other'},
                'country/continent': {'USA': 'USA',
                                      'Europe': 'Europe',
                                      'South America': 'South America',
                                      'Asia': 'Asia',
                                      'North America': 'North America',
                                      'Canada': 'Canada',
                                      'Africa':'Africa',
                                      'Siberia': 'Siberia',
                                      'Australia': 'Australia',
                                      'Central America': 'Central America'},
                'goal': {1: 'Seemed like a fun night out',
                         2: 'To meet new people',
                         3: 'To get a date',
                         4: 'Looking for a serious relationship',
                         5: 'To say I did it',
                         6: 'Other'}
               }
