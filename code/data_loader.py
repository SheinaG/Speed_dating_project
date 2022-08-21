import pandas as pd
from tabulate import tabulate

# Local imports
import consts as cts

def load_df(file_loc=cts.DATA_DIR / 'Speed_Dating_Data.csv', filter_wave=False):
    """
    load original data.
    :param file_loc: original data to load
    :type file_loc: Dataframe
    :param filter_wave: filter the data based on wave number
    :type filter_wave: Str
    """
    sde_df = pd.read_csv(file_loc, encoding='cp1252')

    # omit waves 15 and above, as they were excluded from the original paper
    if filter_wave:
        sde_df = sde_df.loc[sde_df.wave.le(14)]
    return sde_df

def get_data(orig_file, replace_nan=True):
    """
    returns dataframe with all the relavent features for our project.
    :param orig_file: original data to use for amputation
    :type orig_file: Dataframe
    :param replace_nan: wether to deal with nan values or not, if so nan values are replaced with median
    :type replace_nan: Bool
    """
    df_by_other = orig_file.groupby('iid')[cts.x_by_others].mean().reset_index() # average the total rating of
    # the participants by others
    df = orig_file.groupby('iid')['match'].sum() # calculate number of matches per participant
    df = df.reset_index()
    df.rename(columns={0: 'goal'})
    df['goal'] = df['iid'].map(orig_file.drop_duplicates('iid').set_index('iid').goal)
    df['round'] = df['iid'].map(orig_file.drop_duplicates('iid').set_index('iid')['round'])
    for col in cts.x:
        if col in cts.x_by_others:
            # average the total rating of the participants by others
            df[col] = df['iid'].map(df_by_other.drop_duplicates('iid').set_index('iid')[col])
        else:
            df[col] = df['iid'].map(orig_file.drop_duplicates('iid').set_index('iid')[col])
    # normalize the number of matches and estimated matches by the number of dates per participant
    df['norm_match'] = df['match'] / df['round']
    df['norm_match_es'] = df['match_es'] / df['round']
    print(f'number of variables in the subset: {len(df.columns)}')
    for col in cts.x_encoding:
        one_hot = pd.get_dummies(df[col]).rename(columns=lambda x: col + ':' + cts.encoding_dict[col][x]) # Get one hot encoding of columns career
        df = df.drop(col, axis=1)# Drop column career as it is now encoded
        df = df.join(one_hot) # Join the encoded df
    if replace_nan:
        df = df.fillna(df.median())
    return df

if __name__ == '__main__':
    sde_df = load_df(file_loc=cts.DATA_DIR / 'Speed_Dating_Data.csv', filter_wave=True)
    subset = get_data(orig_file=sde_df)
    print(f'Number of total participants in all 21 experiments: {len(sde_df.drop_duplicates("iid"))}')
    print('excluding 7 trials for various reasons results in 14 experimental data available')
    print(f'number of unique participants: {len(subset)}')
    print(f'Number of participants with at least one match: {len(subset[subset.match >0])},'
          f' ({round(100*(len(subset[subset.match >0]) / len(subset)),2)}% out of all participants)')
    print(f'highset number of matches per participant of {subset.loc[subset.norm_match.idxmax(), "match"]} out of '
          f'{subset.loc[subset.norm_match.idxmax(), "round"]}')
    goal_table = subset['goal'].value_counts().rename(
        index=lambda x: cts.encoding_dict['goal'][x]).reset_index().rename(
        columns={'index': 'goal', 'goal': 'Number of Subjects'})
    goal_table['percentage'] = round(100 * (goal_table['Number of Subjects'] / len(subset)), 2)
    print(tabulate(goal_table, headers=goal_table.columns))
