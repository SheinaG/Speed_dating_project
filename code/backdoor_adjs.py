import pandas as pd
from dowhy import CausalModel


# Avoid printing dataconversion warnings from sklearn and numpy
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)


data = pd.read_csv('/MLAIM/AIMLab/ShanySheina/Causal_Inference/data/project_data.csv')

# features groups
T = data['goal']
y = 'norm_match'
data = data.drop(['match'], axis=1)
T = T.astype('int')
T[(T != 3) & (T != 4)] = 0
T[(T == 3) | (T == 4)] = 1
data['goal'] = T

Sociodemographic = [ 'field_cd', 'career_c:Lawyer', 'career_c:Academic/Research', 'career_c:Psychologist',
                 'career_c:Doctor/Medicine', 'career_c:Engineer', 'career_c:Creative Arts/Entertainment',
                 'career_c:Banking/Consulting/Finance/Marketing/Business/CEO/Entrepreneur/Admin',
                 'career_c:Real Estate', 'career_c:International/Humanitarian Affairs',
                 'career_c:Undecided', 'career_c:Social Work',  'career_c:Speech Pathology', 'career_c:Politics',
                 'career_c:Pro sports/Athletics', 'career_c:Other']
Personal_preferences = ['attr3_1', 'sinc3_1', 'intel3_1', 'fun3_1', 'amb3_1']
Speae = ['imprace', 'imprelig', 'attr1_1', 'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1']

model = CausalModel(
        data=data,
        treatment='goal',
        outcome='norm_match',
        common_causes=Sociodemographic+Personal_preferences+Speae)

identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

estimate = model.estimate_effect(identified_estimand,
                                 method_name="backdoor.propensity_score_stratification")
print(estimate)
print("Causal Estimate is " + str(estimate.value))