#Installing all dependencies:

import sys
import subprocess

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'keras-tuner', '--upgrade'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'nfl_data_py'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'tensorflow'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'numpy==1.21'])
import pandas as pd
import requests
import nfl_data_py as nfl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import keras_tuner as kt
from keras_tuner import HyperModel, RandomSearch, Hyperband, BayesianOptimization
import seaborn as sns
import matplotlib.pyplot as plt
#import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import decomposition
from sklearn.decomposition import PCA, KernelPCA
import numpy as np

#Interpreting fantasy data
def get_fantasy_season_df(league_id,season,swid,espn_s2):
  slotcodes = {
      0 : 'QB', 2 : 'RB', 4 : 'WR',
      6 : 'TE', 16: 'Def', 17: 'K',
      20: 'Bench', 21: 'IR', 23: 'Flex'
  }

  url = 'https://fantasy.espn.com/apis/v3/games/ffl/seasons/' + \
        str(season) + '/segments/0/leagues/' + str(league_id) + \
        '?view=mMatchup&view=mMatchupScore'
  data = []
  for week in range(1, 18):
      r = requests.get(url,
                      params={'scoringPeriodId': week},
                      cookies={"SWID": swid, "espn_s2": espn_s2})
      d = r.json()
      for tm in d['teams']:
          tmid = tm['id']
          for p in tm['roster']['entries']:
              name = p['playerPoolEntry']['player']['fullName']
              slot = p['lineupSlotId']
              pos  = slotcodes[slot]

              # injured status (need try/exc bc of D/ST)
              inj = 'NA'
              try:
                  inj = p['playerPoolEntry']['player']['injuryStatus']
              except:
                  pass

              # projected/actual points
              proj, act = None, None
              for stat in p['playerPoolEntry']['player']['stats']:
                  if stat['scoringPeriodId'] != week:
                      continue
                  if stat['statSourceId'] == 0:
                      act = stat['appliedTotal']
                  elif stat['statSourceId'] == 1:
                      proj = stat['appliedTotal']

              data.append([
                  week, tmid, name, slot, pos, inj, proj, act
              ])
  data = pd.DataFrame(data, columns=['Week', 'Team', 'Player', 'Slot', 'Pos', 'Status', 'Proj', 'Actual'])
  return data
#The next five functions are all for interpreting and restructuring the play-by-play data
def get_throwing_df(pbp, df_players, df_teams):
  pass_plays=pbp[pbp['play_type']=='pass']
  two_point_mapping={None:0,'failure':0,'success':1}
  pass_plays['two_point_conversions']=pass_plays['two_point_conv_result'].map(two_point_mapping)
  qb_pass_plays = pass_plays.merge(df_players[["player_id", "player_name","position"]], left_on="passer_player_id", right_on="player_id")
  qb_pass_plays=qb_pass_plays.merge(df_teams[["team_abbr", "team_color"]], left_on="posteam", right_on="team_abbr")
  # get total passing yards and touchdowns by week
  qb_pass_agg = (
      qb_pass_plays.groupby(["player_name","player_id","position","team_abbr", "team_color", "week", "game_half"], as_index=False)
      .agg({"passing_yards": "sum", "pass_touchdown": "sum","interception": "sum","two_point_conversions":"sum"})
  )
  return qb_pass_agg

def get_receiving_df(pbp, df_players, df_teams):
  pass_plays=pbp[pbp['play_type']=='pass']
  two_point_mapping={None:0,'failure':0,'success':1}
  pass_plays['two_point_conversions']=pass_plays['two_point_conv_result'].map(two_point_mapping)
  receive_plays=pass_plays.merge(df_players[["player_id", "player_name","position"]], left_on="receiver_player_id", right_on="player_id")
  receive_plays=receive_plays.merge(df_teams[["team_abbr", "team_color"]], left_on="posteam", right_on="team_abbr")
  gb=["player_name","player_id","position", "team_abbr", "team_color", "week", "game_half"]
  aggregation={"receiving_yards": "sum", "touchdown": "sum","complete_pass": "count","two_point_conversions":"sum"}
  receive_agg=receive_plays.query("complete_pass==1.").groupby(gb, as_index=False).agg(aggregation)
  receive_agg=receive_agg.rename(columns={'touchdown':'receiving_touchdown'})
  return receive_agg

def get_rushing_df(pbp, df_players, df_teams):
  run_plays=pbp[pbp['play_type']=='run']
  two_point_mapping={None:0,'failure':0,'success':1}
  run_plays['two_point_conversions']=run_plays['two_point_conv_result'].map(two_point_mapping)
  run_plays=run_plays.merge(df_players[["player_id", "player_name","position"]], left_on="rusher_player_id", right_on="player_id")
  run_plays=run_plays.merge(df_teams[["team_abbr", "team_color"]], left_on="posteam", right_on="team_abbr")
  run_agg = (
      run_plays.groupby(["player_name","player_id","position", "team_abbr", "team_color", "week", "game_half"], as_index=False)
      .agg({"rushing_yards": "sum", "rush_touchdown": "sum","two_point_conversions":"sum"})
  )
  return run_agg

def fumbles(pbp):
  fumble_plays=pbp[pbp['fumble_lost']==1]
  return fumble_plays.groupby(['fumbled_1_player_id','fumbled_1_player_name','week','game_half']).agg({'fumble':'count'}).reset_index()

def get_all_stats_df(throwing_df,receiving_df,rushing_df,fumble_df):
  merge_on_columns=['player_name','player_id','position','team_abbr','team_color','week','game_half']
  throw_and_receive=throwing_df.merge(receiving_df,how='outer',on=merge_on_columns,suffixes=['_pass','_rec'])
  all_stats=throw_and_receive.merge(rushing_df,how='outer',on=merge_on_columns,suffixes=['_non_rush','_rush'])
  fumble_df['player_id']=fumble_df['fumbled_1_player_id']
  new_merge_columns=['player_id','week','game_half']
  all_stats=all_stats.merge(fumble_df,how='left',on=new_merge_columns,suffixes=['_stats','_fumb'])
  return all_stats.fillna(0)

#These take the reformatted pbp data and convert to fantasy points
def stats_to_fantasy_points(row):
  pass_points=.04*row['passing_yards']+4*row['pass_touchdown']
  rec_points=row['complete_pass']+0.1*row['receiving_yards']+6*row['receiving_touchdown']
  rush_points=.1*row['rushing_yards']+6*row['rush_touchdown']
  turnover_points=-2*(row['interception']+row['fumble'])
  two_point_conversions=2*(row['two_point_conversions_pass']+row['two_point_conversions_rec']+row['two_point_conversions'])
  return pass_points+rec_points+rush_points+turnover_points+two_point_conversions

def add_fantasy_points_to_df(all_stats_df):
  all_stats_df['fantasy_pts']=all_stats_df.apply(stats_to_fantasy_points,axis=1)
  return all_stats_df

def get_stats_and_fantasy_df(fantasy_df,all_stats_df):
  all_stats_df=all_stats_df.rename(columns={'player_name': 'Player', 'week': 'Week'})
  merged=fantasy_df.merge(all_stats_df)
  merged.drop(labels=['Slot','Pos','Team','player_id','team_color','fumbled_1_player_id', 'fumbled_1_player_name'],axis=1,inplace=True)
  merged.rename({'complete_pass':'receptions','Actual':'Fantasy: Actual','fantasy_pts':'Fantasy: Calculated'},inplace=True,axis=1)
  return merged

def first_half_df(stats_and_fantasy_df):
  first_half=stats_and_fantasy_df[stats_and_fantasy_df['game_half']=='Half1']
  first_half['Fantasy: Second Half']=first_half['Fantasy: Actual']-first_half['Fantasy: Calculated']
  first_half.rename({'Fantasy: Actual': 'Fantasy: Full Game', 'Fantasy: Calculated': 'Fantasy: First Half'},inplace=True,axis=1)
  return first_half

def complete_first_half_df(pbp, first_half):
  pbp_first_half = pbp[pbp['game_half'] == 'Half1']
  team_stats = pbp_first_half.groupby(['week','posteam']).agg({'rushing_yards': 'sum', 
  'rush_touchdown': 'sum','passing_yards': 'sum', 'pass_touchdown': 'sum','interception': 'sum', 
  'fumble': 'sum', 'receiving_yards': 'sum', 'complete_pass': 'sum', 'touchdown': 'sum'})
  all_half = first_half.merge(team_stats.reset_index(), left_on=['Week', 'team_abbr'],
  right_on=['week', 'posteam'], how='outer').drop(columns=['week', 'posteam'])
  renamed_half = all_half.rename({
    'passing_yards_x':'Passing_Yards',
    'pass_touchdown_x':'Pass_TDs',
    'rushing_yards_x':'Rushing_Yards',
    'rush_touchdown_x':'Rushing_TDs',
    'interception_x':'Interceptions',
    'receiving_yards_x':'Receiving_Yards',
    'fumble_x':'Fumbles',
    'rushing_yards_y':'Team_Rushing_Yards',
    'rush_touchdown_y':'Team_Rushing_TDs',
    'passing_yards_y':'Team_Passing_Yards',
    'pass_touchdown_y':'Team_Passing_TDs',
    'interception_y':'Team_Interceptions',
    'receiving_yards_y':'Team_Receiving_Yards',
    'fumble_y':'Team_Fumbles',
    'complete_pass':'Team_Completions',
    'touchdown':'Team_Receiving_TDs'}, axis=1)
  return renamed_half

#Function used to accumulate weekly stats up until a given week (i.e. a player's average target share over the first 6 weeks ahead of week 7).
def calculate_avg_to_week(df, stat_name, week, player):
  return df.loc[(df['player_display_name'] == player) & (df['week']<week) & df[stat_name], stat_name].mean()
#Same as above just for team stats
def calculate_team_avg_to_week(df, stat_name, week, team):
  return df.loc[(df['recent_team'] == team) & (df['week']<week) & df[stat_name], stat_name].mean()

qb_features=['Passing_Yards', 'Pass_TDs',
        'Interceptions', 'two_point_conversions_pass',
        'Rushing_Yards', 'Rushing_TDs', 'two_point_conversions', 'Fumbles','Fantasy: First Half','team_cum_avg_passing_yards',
        'team_cum_avg_passing_tds', 'team_cum_avg_attempts',
        'team_cum_avg_completions', 'team_cum_avg_rushing_yards',
        'team_cum_avg_rushing_tds', 'team_cum_avg_sacks',
        'team_cum_avg_interceptions']
rb_features=['Receiving_Yards',
      'receiving_touchdown', 'receptions', 'two_point_conversions_rec',
      'Rushing_Yards', 'Rushing_TDs', 'Fumbles',
      'Fantasy: First Half', 'Team_Rushing_Yards',
      'Team_Rushing_TDs', 'Team_Passing_Yards', 'Team_Passing_TDs',
      'Team_Interceptions', 'Team_Fumbles', 'Team_Receiving_Yards',
      'Team_Completions', 'Team_Receiving_TDs', 'cum_avg_targets',
      'cum_avg_receptions', 'cum_avg_carries', 'cum_avg_target_share',
      'cum_avg_air_yards_share',
      'cum_avg_passing_yards_after_catch',
    'cum_avg_rushing_yards',
      'cum_avg_rushing_fumbles', 'team_cum_avg_passing_yards',
      'team_cum_avg_passing_tds', 'team_cum_avg_attempts',
      'team_cum_avg_completions', 'team_cum_avg_rushing_yards',
      'team_cum_avg_rushing_tds', 'team_cum_avg_rushing_fumbles',
      'team_cum_avg_sack_fumbles']
wr_features=['Receiving_Yards',
      'receiving_touchdown', 'receptions', 'two_point_conversions_rec',
    'Rushing_Yards', 'Rushing_TDs', 'Fumbles',
      'Fantasy: First Half', 'Team_Rushing_Yards',
      'Team_Rushing_TDs', 'Team_Passing_Yards', 'Team_Passing_TDs',
      'Team_Interceptions', 'Team_Fumbles', 'Team_Receiving_Yards',
      'Team_Completions', 'Team_Receiving_TDs', 'cum_avg_targets',
      'cum_avg_receptions', 'cum_avg_carries', 'cum_avg_target_share',
      'cum_avg_air_yards_share',
      'cum_avg_passing_yards_after_catch',
      'cum_avg_rushing_yards',
      'cum_avg_rushing_fumbles', 'team_cum_avg_passing_yards',
      'team_cum_avg_passing_tds', 'team_cum_avg_attempts',
      'team_cum_avg_completions', 'team_cum_avg_rushing_yards',
      'team_cum_avg_rushing_tds', 'team_cum_avg_rushing_fumbles',
      'team_cum_avg_sack_fumbles']
te_features=wr_features
pos_to_features={'QB': qb_features,'RB': rb_features,'WR': wr_features,'TE': te_features}
#Returns a dictionary with the RMSE of the ESPN second half projections
#Input: half plus historical dataframe
def espn_errors_dict(df):
  errors=dict()
  positions=['QB','RB','WR','TE']
  for p in positions:
    pos_df=df[df['position']==p]
    test_data=pos_df['Fantasy: Second Half']
    espn_proj=pos_df['Proj']/2.0 #NOTE: ESPN projects each player half of their pregame projection for the 2nd half
    errors[p]=mean_squared_error(test_data,espn_proj,squared=False)
  return errors

def split_df_by_pos (df):
  X_by_pos = dict()
  y_by_pos = dict()
  for pos in ['RB', 'QB', 'WR', 'TE']:
    pos_df=df[df['position']==pos]  
    y_by_pos[pos]=pos_df['Fantasy: Second Half'].values
    y_by_pos[pos]=y_by_pos[pos].reshape(-1,1)
    features=pos_to_features[pos]
    X_by_pos[pos]=pos_df[features].values
  return X_by_pos, y_by_pos
#Neural network functions:
#standardizes data for use in network
#returns data' (scaled),fitted scaler
def standardize_data(data):
  PredictorScaler=StandardScaler()
  PredictorScalerFit=PredictorScaler.fit(data)
  scaled_data=PredictorScalerFit.transform(data)
  return scaled_data,PredictorScalerFit

#Builds a simple neural network
#Input: half_plus_historical dataframe, position string (ie 'QB')
#Outputs: a model, a training dataframe, a testing dataframe
def build_baseline_nn(df,pos):
  X, y = split_df_by_pos (df)
  X,y=X[pos],y[pos]
  X,x_scaler_fit=standardize_data(X)
  y,y_scaler_fit=standardize_data(y)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
  features=pos_to_features[pos]
  #ANN model
  model=Sequential()
  model.add(Dense(units=64,input_dim=len(features),kernel_initializer='normal',activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(1, kernel_initializer='normal'))
  model.compile(loss='mean_squared_error',optimizer='adam')
  model.fit(X_train, y_train ,batch_size = 20, epochs = 50, verbose=0,validation_data=(X_test,y_test))

  X_train_orig=x_scaler_fit.inverse_transform(X_train)
  y_train_orig=y_scaler_fit.inverse_transform(y_train)
  trainingdata=pd.DataFrame(data=X_train_orig,columns=features)
  trainingdata['Fantasy: Second Half']=y_train_orig

  predictions=model.predict(X_test)
  predictions=y_scaler_fit.inverse_transform(predictions)
  y_test_orig=y_scaler_fit.inverse_transform(y_test)
  X_test_orig=x_scaler_fit.inverse_transform(X_test)

  testingdata=pd.DataFrame(data=X_test_orig, columns=features)
  testingdata['Fantasy: Second Half']=y_test_orig
  testingdata['Second Half Projection']=predictions
  return model, trainingdata, testingdata

#Builds a tuned neural network, based on hyperparameter tuning with kerastuner below
#Input: half_plus_historical dataframe, position string (ie 'QB')
#Outputs: a model, a training dataframe, a testing dataframe
def build_tuned_nn(df,pos):
  X, y = split_df_by_pos (df)
  X,y=X[pos],y[pos]
  X,x_scaler_fit=standardize_data(X)
  y,y_scaler_fit=standardize_data(y)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
  features=pos_to_features[pos]

  #All of the following are based on extensive tuning
  qb_params={'units1':40,'units2':32,'dense_activation':'sigmoid','dropout':.1} 
  rb_params={'units1':28,'units2':32,'dense_activation':'tanh','dropout':.13}
  wr_params={'units1':24,'units2':28,'dense_activation':'relu','dropout':.12}
  te_params={'units1':64,'dense_activation':'relu','dropout':.2}
  params_dict={'QB':qb_params,'WR':wr_params,'RB':rb_params,'TE':te_params}
  #ANN model
  model = Sequential()
  model.add(Dense(input_dim=len(features),units=params_dict[pos]['units1'],
          activation=params_dict[pos]['dense_activation'],
          kernel_initializer='zeros', bias_initializer='zeros'))
  if pos !='TE':
    model.add(Dense(units=params_dict[pos]['units1'],
          activation=params_dict[pos]['dense_activation'],
          kernel_initializer='zeros', bias_initializer='zeros'))
  model.add(Dropout(params_dict[pos]['dropout']))
  model.add(Dense(1, kernel_initializer='zeros', bias_initializer='zeros'))
  model.compile(
      optimizer='rmsprop',loss='mse',metrics=['mse'])
  model.fit(X_train, y_train ,batch_size = 20, epochs = 50, verbose=0,validation_data=(X_test,y_test))

  X_train_orig=x_scaler_fit.inverse_transform(X_train)
  y_train_orig=y_scaler_fit.inverse_transform(y_train)
  trainingdata=pd.DataFrame(data=X_train_orig,columns=features)
  trainingdata['Fantasy: Second Half']=y_train_orig

  predictions=model.predict(X_test)
  predictions=y_scaler_fit.inverse_transform(predictions)
  y_test_orig=y_scaler_fit.inverse_transform(y_test)
  X_test_orig=x_scaler_fit.inverse_transform(X_test)

  testingdata=pd.DataFrame(data=X_test_orig, columns=features)
  testingdata['Fantasy: Second Half']=y_test_orig
  testingdata['Second Half Projection']=predictions
  return model, trainingdata, testingdata

def baseline_nn_error_dict(df):
  errors=dict()
  positions=['QB','RB','WR','TE']
  for p in positions:
    model,trainingdata,testingdata=build_baseline_nn(df,p)
    y_test=testingdata['Fantasy: Second Half']
    yhat=testingdata['Second Half Projection']
    X_test=testingdata.drop(['Fantasy: Second Half','Second Half Projection'],axis=1).values
    rmse=mean_squared_error(y_test,yhat,squared=False)
    errors[p]=rmse
  return errors

def tuned_nn_error_dict(df):
  errors=dict()
  positions=['QB','RB','WR','TE']
  for p in positions:
    model,trainingdata,testingdata=build_tuned_nn(df,p)
    y_test=testingdata['Fantasy: Second Half']
    yhat=testingdata['Second Half Projection']
    X_test=testingdata.drop(['Fantasy: Second Half','Second Half Projection'],axis=1).values
    rmse=mean_squared_error(y_test,yhat,squared=False)
    errors[p]=rmse
  return errors

#The below code was used to tune the neural network hyperparameters. 
#It has been ommited from this program's main function for runtime purposes.
#The values it found have been hardcoded into the tuned model.
#A similar approach was used for the other two models.
class RegressionHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()
        model.add(Dense(units=hp.Int('units', 8, 72, 4, default=12),
                activation=hp.Choice(
                    'dense_activation',
                    values=['sigmoid','relu', 'tanh'],
                    default='relu'),
                input_shape=input_shape,
                kernel_initializer='zeros', bias_initializer='zeros'))
        model.add(Dense(units=hp.Int('units', 16, 72, 4, default=16),
                activation=hp.Choice(
                    'dense_activation',
                    values=['sigmoid','relu', 'tanh'],
                    default='relu'),
                kernel_initializer='zeros', bias_initializer='zeros'))
        model.add(Dropout(hp.Float(
                    'dropout',
                    min_value=0.0,
                    max_value=0.2,
                    default=0.005,
                    step=0.01)))
        model.add(Dense(1, kernel_initializer='zeros', bias_initializer='zeros'))
        model.compile(
            optimizer='rmsprop',loss='mse',metrics=['mse'])
        return model

def get_bayesian_opt_tuner(trainingdata):
  tuner_bo = BayesianOptimization(
              hypermodel,
              objective='mse',
              max_trials=10,
              executions_per_trial=2,
              directory='b_o',overwrite=True
          )
  y_train=trainingdata['Fantasy: Second Half'].values.reshape(-1,1)
  X_train=trainingdata.drop('Fantasy: Second Half',axis=1).values

  X,x_scaler=standardize_data(X_train)
  y,y_scaler=standardize_data(y_train)

  tuner_bo.search(X,y, epochs=10, validation_split=0.2, verbose=0)
  return tuner_bo,x_scaler,y_scaler

def get_rs_tuner(trainingdata):
  tuner_rs = RandomSearch(
              hypermodel,
              objective='mse',
              max_trials=10,
              executions_per_trial=2, overwrite=True
          )
  y_train=trainingdata['Fantasy: Second Half'].values.reshape(-1,1)
  X_train=trainingdata.drop('Fantasy: Second Half',axis=1).values

  X,x_scaler=standardize_data(X_train)
  y,y_scaler=standardize_data(y_train)

  tuner_rs.search(X,y, epochs=10, validation_split=0.2, verbose=0)
  return tuner_rs,x_scaler,y_scaler

def get_hyperband_tuner(trainingdata):
  tuner_hb = Hyperband(
              hypermodel,
            max_epochs=5,
            objective='mse',
            executions_per_trial=2, overwrite=True
          )
  y_train=trainingdata['Fantasy: Second Half'].values.reshape(-1,1)
  X_train=trainingdata.drop('Fantasy: Second Half',axis=1).values

  X,x_scaler=standardize_data(X_train)
  y,y_scaler=standardize_data(y_train)

  tuner_hb.search(X,y, epochs=10, validation_split=0.2, verbose=0)
  return tuner_hb,x_scaler,y_scaler

def nn_tuning_code():
  #Code below was used for hyperparameter tuning

  just_features=testingdata.drop(['Fantasy: Second Half','Second Half Projection'],axis=1)
  input_shape = (just_features.shape[1],) #last 2 columns of testingdata were not features
  hypermodel = RegressionHyperModel(input_shape)
  tuner_bo,x_scaler,y_scaler=get_bayesian_opt_tuner(trainingdata)
  best_model = tuner_bo.get_best_models(num_models=1)[0]
  best_hps=tuner_bo.get_best_hyperparameters(1)[0]
  print(best_hps.space)
  print(best_hps.values)

#Random Forest Regression code below
def build_forest (df, pos):
  X, y = split_df_by_pos (df)
  X_train, X_test, y_train, y_test = train_test_split(X[pos], y[pos], test_size=0.3, random_state=1)
  rgr = RandomForestRegressor(n_estimators = 100)
  rgr.fit(X_train, y_train.ravel())
  y_pred = rgr.predict(X_test)
  features=pos_to_features[pos]
  testingdata=pd.DataFrame(data=X_test, columns=features)
  testingdata['Fantasy: Second Half']=y_test
  testingdata['Second Half Projection']=y_pred
  return rgr, testingdata

def build_tuned_forest(df, pos):
  X, y = split_df_by_pos (df)
  X_train, X_test, y_train, y_test = train_test_split(X[pos], y[pos], test_size=0.3, random_state=1)
  if(pos == 'RB'):
    rgr = RandomForestRegressor(n_estimators=120, min_samples_split=4, max_leaf_nodes=7, max_features='sqrt', min_samples_leaf=25)
  elif(pos == 'QB'):
    rgr = RandomForestRegressor(n_estimators=50, min_samples_split=3, max_leaf_nodes=5, max_features='sqrt', min_samples_leaf=29)
  elif(pos == 'WR'):
    rgr = RandomForestRegressor(n_estimators=16, min_samples_split=4, max_leaf_nodes=5, max_features='sqrt', min_samples_leaf=25)
  elif(pos == 'TE'):
    rgr = RandomForestRegressor(n_estimators=13, max_leaf_nodes=4, max_features='sqrt', min_samples_leaf=22)
  rgr.fit(X_train, y_train.ravel())
  y_pred = rgr.predict(X_test)
  features=pos_to_features[pos]
  testingdata=pd.DataFrame(data=X_test, columns=features)
  testingdata['Fantasy: Second Half']=y_test
  testingdata['Second Half Projection']=y_pred
  return rgr, testingdata

def forest_errors_dict(df):
  errors=dict()
  positions=['QB','RB','WR','TE']
  for p in positions:
    rgr, testingdata=build_forest(df,p)
    y_test=testingdata['Fantasy: Second Half']
    yhat=testingdata['Second Half Projection']
    X_test=testingdata.drop(['Fantasy: Second Half','Second Half Projection'],axis=1).values
    rmse=mean_squared_error(y_test,yhat,squared=False)
    errors[p]=rmse
  return errors

def tuned_forest_errors_dict(df):
  errors=dict()
  positions=['QB','RB','WR','TE']
  for p in positions:
    rgr, testingdata=build_tuned_forest(df,p)
    y_test=testingdata['Fantasy: Second Half']
    yhat=testingdata['Second Half Projection']
    X_test=testingdata.drop(['Fantasy: Second Half','Second Half Projection'],axis=1).values
    rmse=mean_squared_error(y_test,yhat,squared=False)
    errors[p]=rmse
  return errors

#Hyperparameter tuning code (unused in this program):
def rf_tuning(df):
  #param_grid = {'n_estimators': range(50,120,20), 'max_depth' : range(5, 30, 2), 'min_samples_split' : range(2,10), 'max_leaf_nodes' : range(10, 50, 9), 'max_features' : ['sqrt']}
  pos = 'TE'
  param_grid = {'n_estimators' : range(5,15,2), 'min_samples_split': range(2,5), 'max_leaf_nodes' : range(3,7), 'max_features' : ['sqrt'], 'min_samples_leaf' : range(20,30,2)}
  X, y = split_df_by_pos (df)
  X_train, X_test, y_train, y_test = train_test_split(X[pos], y[pos], test_size=0.3, random_state=1)
  grid_search = GridSearchCV(RandomForestRegressor(),
                          param_grid=param_grid)
  grid_search.fit(X_train, y_train.ravel())
  print(grid_search.best_estimator_)

#Ridge regression code below
def build_rr_model(df, pos):
    X, y = split_df_by_pos(df)
    X_train, X_test, y_train, y_test = train_test_split(X[pos], y[pos], test_size=0.3, random_state=1)
    rr = Ridge().fit(X_train, y_train.ravel())
    predictions = rr.predict(X_test)
    features = pos_to_features[pos]
    testingdata = pd.DataFrame(data=X_test, columns=features)
    testingdata['Fantasy: Second Half'] = y_test
    testingdata['Second Half Projection'] = predictions
    return rr, testingdata

def rr_errors(df):
    errors = dict()
    pos = ['QB', 'RB', 'WR', 'TE']
    for p in pos:
        model,testingdata=build_rr_model(df,p)
        y_test=testingdata['Fantasy: Second Half']
        yhat=testingdata['Second Half Projection']
        X_test=testingdata.drop(['Fantasy: Second Half','Second Half Projection'],axis=1).values
        rmse=mean_squared_error(y_test,yhat,squared=False)
        errors[p] = rmse
    return errors

def build_tuned_rr_model(df, pos):
    X, y = split_df_by_pos(df)
    X_train, X_test, y_train, y_test = train_test_split(X[pos], y[pos], test_size=0.3, random_state=1)
    # Build Tuned Model
    X, y = split_df_by_pos(df)
    X_train, X_test, y_train, y_test = train_test_split(X[pos], y[pos], test_size=0.3, random_state=1)
    if(pos == 'QB'):
      kpca_tuner = KernelPCA(n_components = 2, degree = 1)
      X_reduced = kpca_tuner.fit_transform(X_train)
      X_test_reduced = kpca_tuner.fit_transform(X_test)
      rr_tuned = Ridge(alpha = 10, solver = 'sag').fit(X_reduced, y_train.ravel())
    if(pos == 'RB'):
      kpca_tuner = KernelPCA(n_components = 5, degree = 4)
      X_reduced = kpca_tuner.fit_transform(X_train)
      X_test_reduced = kpca_tuner.fit_transform(X_test)
      rr_tuned = Ridge(alpha = 200, solver = 'lsqr').fit(X_reduced, y_train.ravel())
    if(pos == 'WR'):
      X_reduced = X_train
      X_test_reduced = X_test
      rr_tuned = Ridge().fit(X_train, y_train.ravel())
    if(pos == 'TE'):
      kpca_tuner = KernelPCA(n_components = 13, degree = 1)
      X_reduced = kpca_tuner.fit_transform(X_train)
      X_test_reduced = kpca_tuner.fit_transform(X_test)
      rr_tuned = Ridge(alpha = 10, solver = 'sparse_cg').fit(X_reduced, y_train.ravel())
    # pca_tuner = PCA(num_components)
    # X_reduced = pca_tuner.fit_transform(X_train)
    # rr_tuned = Ridge(alpha = best_alpha, solver = best_solver).fit(X_reduced, y_train.ravel())
    
    predictions = rr_tuned.predict(X_test_reduced)
    features = pos_to_features[pos]
    testingdata = pd.DataFrame(data=X_test, columns=features)
    testingdata['Fantasy: Second Half'] = y_test
    testingdata['Second Half Projection'] = predictions
    return rr_tuned, testingdata

def tuned_errors(df):
    errors = dict()
    pos = ['QB', 'RB', 'WR', 'TE']
    for p in pos:
        model,testingdata=build_tuned_rr_model(df,p)
        y_test=testingdata['Fantasy: Second Half']
        yhat=testingdata['Second Half Projection']
        X_test=testingdata.drop(['Fantasy: Second Half','Second Half Projection'],axis=1).values
        rmse=mean_squared_error(y_test,yhat,squared=False)
        errors[p] = rmse
    return errors

#Ridge regression hyperparameter tuning (unused):
def find_hypers(df):
    #Find Hypers
    scalar = StandardScaler()
    kpca = decomposition.KernelPCA(kernel= 'poly')
    rr = Ridge()
    pipeline = Pipeline(steps = [("std_slc", scalar), ("kpca", kpca), ("ridge", rr)])
    components = list(range(1,X_train.shape[1]+1, 1))
    alphas = [10, 50, 100, 150, 200, 500, 1000]
    degrees = [1, 2, 3, 4]
    solver = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    hyper_parameters = dict(ridge__alpha = alphas, kpca__degree = degrees, kpca__n_components = components, ridge__solver = solver)
    grid_search = GridSearchCV(pipeline, hyper_parameters, refit= True, return_train_score=True, scoring= 'neg_root_mean_squared_error')
    grid_search_final = grid_search.fit(X_train, y_train.ravel())
    num_components = grid_search_final.best_estimator_.get_params()["kpca__n_components"]
    best_degree = grid_search_final.best_estimator_.get_params()["kpca__degree"]
    best_solver = grid_search_final.best_estimator_.get_params()["ridge__solver"]
    best_alpha = grid_search_final.best_estimator_.get_params()["ridge__alpha"]
    print(num_components, best_solver, best_alpha, best_degree)

def graph_errors(errors, title, positions):
  x = np.arange(len(positions))  # the label locations
  width = 0.25  # the width of the bars
  multiplier = 0

  fig, ax = plt.subplots(layout='constrained')

  for attribute, measurement in errors.items():
      offset = width * multiplier
      rects = ax.bar(x + offset, measurement, width, label=attribute)
      ax.bar_label(rects, padding=3)
      multiplier += 1

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('RMSE, Second Half Projections')
  ax.set_title(title)
  ax.set_xticks(x + width, positions)
  ax.legend(loc='upper left', ncols=3)
  ax.set_ylim(0, 7.5)

  plt.show()

def main():
  #Need an existing ESPN league to retrieve fantasy statistics. This is one of Alec's old ones.
  league_id = 184643294
  season=2022
  swid='0'
  espn_s2='AEAm18SEnQdytJFeAH88JXY%2FGtGbjTOG0SfHCwguMTgicol3cYL3ncOz5mdzs9YsO7Ao9jxNXlbGykAHh6I4WBzXWJ99EDJPN7yO829ABcmCu02%2B4Cnwnvde17PT8Wjs2nVJBoYNoHOfq87ZppPSyUGCGEm6dXyQXNBklK31hh5lcTKQ0FfIeM%2B9DETdIzJUIl6oEniDwZMGbbLjwcsToh26rxhqw4Yi2YwnWqdJKMbhkAax7tOwg03C6NbIeG%2BXFvdO0Ik%2F9E776vJaJ18hfrvj'
  data=get_fantasy_season_df(league_id,season,swid,espn_s2)
  print("Fantasy data received")
  df_players=nfl.import_rosters([2022])
  df_teams=nfl.import_team_desc()
  pbp = nfl.import_pbp_data([2022])
  pbp.to_csv("2022_pbp.csv")
  print("Play-by-play data received")
  fumble=fumbles(pbp)
  qb=get_throwing_df(pbp, df_players, df_teams)
  wr=get_receiving_df(pbp, df_players, df_teams)
  rb=get_rushing_df(pbp, df_players, df_teams)
  all_stats_df=get_all_stats_df(qb,wr,rb,fumble)
  all_stats_df=add_fantasy_points_to_df(all_stats_df)
  stats_and_fantasy_df=get_stats_and_fantasy_df(data,all_stats_df)
  print("Play-by-play data transformed")
  wk = nfl.import_weekly_data([2022])
  print("Weekly data received")
  cum_team_stats = pd.DataFrame({'team': wk.recent_team.unique()})
  relevant_team_cum_stats = ['passing_yards', 'passing_tds', 'attempts', 'completions', 'rushing_yards', 'rushing_tds', 'sacks', 'interceptions', 'rushing_fumbles', 'sack_fumbles']
  temp_dict = dict()
  for x in relevant_team_cum_stats:
    temp_dict[x] = 'sum'
  team_cum_stats = wk.groupby(['week', 'recent_team']).agg(temp_dict)
  team_cum_stats = team_cum_stats.reset_index()
  for s in relevant_team_cum_stats:
    t = "team_cum_avg_"+s
    team_cum_stats[t] = team_cum_stats.apply(lambda x: calculate_team_avg_to_week(team_cum_stats, s, x.week, x.recent_team), axis=1)
  relevant_player_cum_stats = ['targets', 'receptions', 'carries', 'target_share', 'air_yards_share', 'interceptions', 'passing_yards_after_catch', 'completions', 'pacr', 'racr', 'rushing_yards', 'rushing_fumbles', 'wopr']
  for s in relevant_player_cum_stats:
    t = "cum_avg_"+s
    wk[t] = wk.apply(lambda x: calculate_avg_to_week(wk, s, x.week, x.player_display_name), axis=1)
  print("Weekly data transformed")
  final_weekly = pd.merge(wk, team_cum_stats, how='inner', on=['week', 'recent_team'])
  xtra_columns = []
  for s in final_weekly.columns:
    if s != 'player_display_name' and s != 'week' and not s.startswith('cum') and not s.startswith('team_cum'):
      xtra_columns.append(s)
  trimmed_final_weekly = final_weekly.drop(columns=xtra_columns)
  first_half=first_half_df(stats_and_fantasy_df)
  complete_half = complete_first_half_df(pbp, first_half)
  complete_half_plus_historical = pd.merge(complete_half, trimmed_final_weekly, how='left', right_on=['week', 'player_display_name'], left_on=['Week', 'Player'])
  complete_half_plus_historical = complete_half_plus_historical.drop(columns=['week', 'player_display_name'])
  complete_half_plus_historical.fillna(0,inplace=True)
  print("Pbp and weekly data combined")
  complete_half_plus_historical.to_csv('half_plus_historical.csv')
  df=pd.read_csv('half_plus_historical.csv')
  espn_errors=espn_errors_dict(df)
  print("ESPN's errors: " + str(espn_errors))
  baseline_nn_dict=baseline_nn_error_dict(df)
  print("Untuned neural network errors: " + str(baseline_nn_dict))
  tuned_nn_dict=tuned_nn_error_dict(df)
  print("Tuned neural network errors: " + str(baseline_nn_dict))
  static_fed = forest_errors_dict(df)
  print("Untuned random forest errors: " + str(static_fed))
  static_tfed = tuned_forest_errors_dict(df)
  print("Tuned random forest errors: " + str(static_tfed))
  base_errors = rr_errors(df)
  print("Untuned ridge regression errors: " + str(base_errors))
  final_errors = tuned_errors(df)
  print("Tuned ridge regression errors: " + str(final_errors))
  positions = ("QB", "RB", "WR", "TE")
  errors_by_pos = {
    'ESPN': (espn_errors['QB'], espn_errors['RB'], espn_errors['WR'], espn_errors['TE']),
    'Baseline NN': (baseline_nn_dict['QB'],baseline_nn_dict['RB'],baseline_nn_dict['WR'],baseline_nn_dict['TE']),
    'Tuned NN': (tuned_nn_dict['QB'],tuned_nn_dict['RB'],tuned_nn_dict['WR'],tuned_nn_dict['TE'])
  }
  graph_errors (errors_by_pos, 'Second Half Projection Errors by Position, Neural Network vs ESPN', positions)
  errors_by_pos = {
    'ESPN': (espn_errors['QB'], espn_errors['RB'], espn_errors['WR'], espn_errors['TE']),
    'Baseline RR': (base_errors['QB'],base_errors['RB'],base_errors['WR'],base_errors['TE']),
    'Tuned RR': (final_errors['QB'],final_errors['RB'],final_errors['WR'],final_errors['TE'])
  }
  graph_errors (errors_by_pos, 'Second Half Projection Errors by Position, Ridge Regression vs ESPN', positions)
  errors_by_pos = {
    'ESPN': (espn_errors['QB'], espn_errors['RB'], espn_errors['WR'], espn_errors['TE']),
    'Untuned RF': (round(static_fed['QB'],2),round(static_fed['RB'],2),round(static_fed['WR'],2),round(static_fed['TE'],2)),
    'Tuned RF': (round(static_tfed['QB'],2),round(static_tfed['RB'],2),round(static_tfed['WR'],2),round(static_tfed['TE'],2))
  }
  graph_errors (errors_by_pos, 'Second Half Projection Errors by Position, Random Forest vs ESPN', positions)
  

main()



  

