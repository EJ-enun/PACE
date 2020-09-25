#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 03:10:39 2020

@author: Enun Enun Bassey J
"""
import requests
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix 
from sklearn.model_selection import KFold, cross_val_score,RandomizedSearchCV,GridSearchCV
from scipy.stats import uniform
from sklearn.preprocessing import normalize, OrdinalEncoder
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
from lxml import html
from lxml.cssselect import CSSSelector
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
#datasets go here.
df_offensive_stats_all_players = pd.read_excel('/Users/cinema/Desktop/PACE/all_player_offensive_stat_2019_2020_season.xlsx')
df_total_stats_all_teams = pd.read_excel('/Users/cinema/Desktop/PACE/fbref_2019_2020_season.xlsx')
df_home_and_away_table = pd.read_excel('/Users/cinema/Desktop/PACE/league_table_home_away_2019_2020.xlsx')
df_league_table = pd.read_excel('/Users/cinema/Desktop/PACE/league_table_overall_2019_2020.xlsx')
df_mid_underlying_performance = pd.read_excel('/Users/cinema/Desktop/PACE/MID_FWD_2019_20_Underlying_Performance.xlsx')
df_miscellaneous_stats = pd.read_excel('/Users/cinema/Desktop/PACE/miscellaneous_stats_2019_2020_season_teams.xlsx')
df_squad_advanced_goalkeeping = pd.read_excel('/Users/cinema/Desktop/PACE/squad_advanced_goalkeeping_2019_2020.xlsx')
df_squad_defensive_actions = pd.read_excel('/Users/cinema/Desktop/PACE/squad_defensive_actions_2019_2020.xlsx')
df_squad_goal_and_shot_creation = pd.read_excel('/Users/cinema/Desktop/PACE/squad_goal_and_shot_creation.xlsx')
df_squad_goalkeeping = pd.read_excel('/Users/cinema/Desktop/PACE/squad_goalkeeping_2019_2020.xlsx')
df_squad_pass_types = pd.read_excel('/Users/cinema/Desktop/PACE/squad_pass_types.xlsx')
df_squad_passing = pd.read_excel('/Users/cinema/Desktop/PACE/squad_passing_2019_2020.xlsx')
df_squad_possession = pd.read_excel('/Users/cinema/Desktop/PACE/squad_possession_2019_2020.xlsx')
df_squad_shooting = pd.read_excel('/Users/cinema/Desktop/PACE/squad_shototing_2019_2020.xlsx')
df_squad_standard_stats = pd.read_excel('/Users/cinema/Desktop/PACE/squad_standard_starts.xlsx')
df_team_play_time = pd.read_excel('/Users/cinema/Desktop/PACE/team_play_time_2019_2020.xlsx')
df_team_overall_xG_xA = pd.read_excel('/Users/cinema/Desktop/PACE/epl_overall_xG_xA.xlsx')
df_team_home_xG_xA = pd.read_excel('/Users/cinema/Desktop/PACE/epl_home_xG_xA.xlsx')
df_team_away_xG_xA = pd.read_excel('/Users/cinema/Desktop/PACE/epl_away_xG_xA.xlsx')
df_fpl_players = pd.read_excel('/Users/cinema/Desktop/PACE/cleaned_players.xlsx')
#df = pd.read_pdf('/Users/cinema/Desktop/PACE/')

#Abstract Methods


def find(data_frame, array_val, search_term):
    df = pd.DataFrame(data_frame)
    for result in df[array_val]:
        if re.search(result, search_term):
            return df[df[array_val].isin(search_term)]
        
        
#df_all_players_offense = pd.DataFrame(df_offensive_stats_all_players)
#df_gk = find(df_offensive_stats_all_players,'Pos', 'GK' )


#housing.dropna(subset=["total_bedrooms"]) 
# option 1 housing.drop("total_bedrooms", axis=1)
 # option 2 median = housing["total_bedrooms"].median() 
# option 3 housing["total_bedrooms"].fillna(median, inplace=True)
"""
url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
url_2 = 'https://fantasy.premierleague.com/api/bootstrap-dynamic/'
url_3 = 'https://fantasy.premierleague.com/api/bootstrap-dynamic/'
url_4 = 'https://fantasy.premierleague.com/api/bootstrap-dynamic/'
r = requests.get(url)
json = r.json()
print(json.keys())

elements_df = pd.DataFrame(json['elements'])
elements_types_df = pd.DataFrame(json['element_types'])
teams_df = pd.DataFrame(json['teams'])
phases_df = pd.DataFrame(json['phases'])
element_stats = pd.DataFrame(json['element_stats'])
events = pd.DataFrame(json['events'])
slim_elements_df = elements_df[['second_name','team','element_type','selected_by_percent','now_cost','minutes','transfers_in', 'transfers_out', 'value_season','total_points', 'value_form', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded', 'penalties_saved', 'own_goals', 'yellow_cards', 'saves', 'bonus', 'bps', 'influence', 'creativity', 'threat', 'influence_rank', 'ict_index' ]]


slim_elements_df['position'] = slim_elements_df.element_type.map(elements_types_df.set_index('id').singular_name)
slim_elements_df['team'] = slim_elements_df.team.map(teams_df.set_index('id').name)
slim_elements_df['value'] = slim_elements_df.value_season.astype(float)
print(slim_elements_df.sort_values('value',ascending=False).head(10))
#slim_elements_df.pivot_table(index='position',values='value',aggfunc=np.mean).reset_index()
#print(pivot.sort_values('value',ascending=False))
slim_elements_df = slim_elements_df.loc[slim_elements_df.value > 0]
pivot = slim_elements_df.pivot_table(index='position',values='value',aggfunc=np.mean).reset_index()
print(pivot.sort_values('value',ascending=False))
team_pivot = slim_elements_df.pivot_table(index='team',values='value',aggfunc=np.mean).reset_index()
team_pivot.sort_values('value',ascending=False)
fwd_df = slim_elements_df.loc[slim_elements_df.position == 'Forward']
mid_df = slim_elements_df.loc[slim_elements_df.position == 'Midfielder']
def_df = slim_elements_df.loc[slim_elements_df.position == 'Defender']
goal_df = slim_elements_df.loc[slim_elements_df.position == 'Goalkeeper']
print(goal_df.value.hist())
"""
#sos features =  team rating, wins/losses/draws at home, goals scored, shots, chances created, chances conceded, goals conceded.  
#This method defines 
#select 2 GKs, 5 def, 5 mid, 3 fwd = 15 total players. 

#persistence: A player can miss 9 shots and be redeemed with one goal. 
#df_offensive_stats_all_players.info()
attackers_fpl = df_fpl_players.query("element_type == 'FWD'")
defenders_fpl = df_fpl_players.query("element_type == 'DEF'")
midfielders_fpl = df_fpl_players.query("element_type == 'MID'")
goalkeepers_fpl = df_fpl_players.query("element_type == 'GK'")

df_fpl = df_fpl_players.query("minutes >= 1000")
df_fpl_creativity = df_fpl_players.query("element_type != 'DEF'")['creativity'] / 38
df_fpl_influence = df_fpl_players.query("element_type != 'DEF'")['influence'] / 38
df_fpl_threat = df_fpl_players.query("element_type != 'DEF'")['threat'] / 38
th = pd.DataFrame(df_fpl_threat)
df_fpl_ict_index = df_fpl_players.query("element_type != 'DEF'")['ict_index'] / 38
df_fpl_total_points = df_fpl_players.query("element_type != 'DEF'")['total_points'] 
df_fpl_goals_scored = df_fpl_players.query("element_type != 'DEF'")['goals_scored']
grp_fpl = np.stack([df_fpl_creativity, df_fpl_influence, df_fpl_threat, df_fpl_ict_index], axis=1)

df_corr_fpl = df_fpl_players.corr()
df_off_corr = df_offensive_stats_all_players.corr()

df_offensive_stats_all_players.describe()
df_fwd = df_offensive_stats_all_players.query("Pos == 'FW'")
df_mid = df_offensive_stats_all_players.query("Pos == 'MF'")
df_def = df_offensive_stats_all_players.query("Pos == 'DF'")
df_xGI = df_mid_underlying_performance['xGI']
df_xA = df_mid_underlying_performance['xA']
df_xG = df_mid_underlying_performance['xG']

model_linear = LinearRegression()
model_k_nearest = KNeighborsRegressor(n_neighbors=3)

squad_xG_per_gw = df_squad_shooting['xG']/38
squad_goals_per_gw = df_squad_shooting['Gls']/38
squad_g_xg_per_gw = df_squad_shooting['G-xG']/38
#min_df = df_offensive_stats_all_players.loc(df['Min'] >= 750)

df_ = df_offensive_stats_all_players.query("Gls >= 2") 
y_goals = np.array((df_['Gls']/38).fillna(0)).reshape(-1, 1)
y_ast = df_offensive_stats_all_players['Ast']
y = (df_offensive_stats_all_players['Gls'] + df_offensive_stats_all_players['Ast'])/38 
x = df_offensive_stats_all_players['xG+xA_per90']
xa = np.array(df_offensive_stats_all_players['xA_per90'].fillna(0)).reshape(-1, 1)
#xg = np.array(x_g
xg =  np.array((df_['xG_expected']/38).fillna(0)).reshape(-1, 1)
min_per_gm = df_offensive_stats_all_players['Min']/df_offensive_stats_all_players['MP']
gm_min = np.array(min_per_gm)
#grp = np.stack([x, xg, xa, min_per_gm], axis=1)

fig = plt.figure()
axis = fig.add_subplot(1,1,1)
axis.plot(xg, y_goals, "o", color="pink")
axis.set_xlabel("XG")
axis.set_ylabel("GOALS")
axis.set_title("Fresh Per GameWeek Plot")

x, x_test, y, y_test = train_test_split(xg, y_goals, test_size=0.3)
model_linear.fit(x,y)
pred = model_linear.predict(x_test)
lin_gl_mse = mean_squared_error(y_test, pred)
 
lin_gl_rmse = np.sqrt(lin_gl_mse)

fig_ = plt.figure()
axis_1 = fig_.add_subplot(1,1,1)
axis_1.plot(y_test, pred, "x", color="black")
axis_1.set_xlabel("Y_REAL")
axis_1.set_ylabel("Y_PRED")
axis_1.set_title("Y Plot")

def mse(y_test, pred):
    n = len(y_test) #find total number of items in our array.
    summ = 0 #var to store summation of differences.
    for i in range(0,n): #looping through each element on the list.
        diff = y_test[i] - pred[i] #finding the difference between observed and predicted value. 
        squared_diff = diff**2 #taking square of the difference. 
        summ = summ + squared_diff #taking sum of all the differences. 
    return summ/n #dividing summation by total values to obtain average

mse_ = mse(y_test, pred)

#pipeline for my estimators and implement the weights in the transformer
num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler())
        ])

#tct = num_pipeline.fit_transform(df_fpl_threat, df_fpl_goals_scored)
lmr = LinearRegression()
x_a, x_t, y_a, y_t = train_test_split(grp_fpl, df_fpl_total_points, test_size = 0.3)
threat_model = lmr.fit(x_a,y_a)
goal_prediction = threat_model.predict(x_t)
lin_mse = mean_squared_error(y_t, goal_prediction) 
lin_rmse = np.sqrt(lin_mse)

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

tree_reg = DecisionTreeRegressor()
tree_reg.fit(x_t,y_t)
#cross vallidation eith 10 cv folds. 
scores = cross_val_score(tree_reg, x_a, y_a, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

#n = len(y_test)
#for x in range(0, n): 
#    y_curr = np.sqrt((x+6)^2 + 25) + np.sqrt((x-6)^2 + 121)
#    print(y_curr)

def din(n,a): #create method head and add parameters in parenthesis
    #for i in range(0,a):
     #   for h in range(0, n):    
            g = (a **((n - 1)/2)) + 1
            #K = (2**n)  + 1
            #p = K*(2**n) + 1
            #N = h * 2**k + 1 #where h < 2**k
            #p = 2**n
            ho = int((g/n)) 
            #ho != float :
            #if g == -1 :
            #    return n, "is a prime number."
            return g #n, "is not a prime number"
print(din(5,3))

#n = h\cdot 2^k + 1, h < 2^k, 2\nmid h$
#if n is prime,then any nonsquare a will satisfy  a^\frac{n-1}{2}\equiv -1\pmod{n}$.
#It remains to show the converse

def isPowerOfTwo(n):
    #g = (5**((n-1)/2)) + 1
    k = 3
    #floor_divisor = n//k
    
    #if(n % k == 0):
    #    return "yup it does divide without remainder."
    #else:
    #    return "it doesn't divide without a remainder. :("
    
    #return (n and (not(n & (n - 1))))#perform binary operation to check power 
import random
def prothNumber(n,k): #For any 2 random positive integers n and k where k is odd, N is our proth number. 
    while(k % 2 != 0 and 2**n > k): #checks if k is odd and if 2^n is greater than k. 
        N = (k*2**n) + 1 #Get Proth Number.
        value_list = list(range(1,10)) #create list in the range of 1 - 10.
        print(N) #Print Proth number
        power = (N-1)/2 #get raised power after relevant computations.
        power_a = [a **power + 1 for a in value_list] #get all values of a raised to power. 
        prime_check = [x % N == 0 for x in power_a] #iterate through all values in range 1 - 10 and check if they are divisible by N(our proth number) without a remainder.
        if(any(prime_check) == True): #If any value in the range returns as true then N is a Proth Prime number.
            return N, "Is a Proth Prime Number."
        else :
            return N, "Is not a Proth Prime Number."
    return "Conditions for proth number generation not satisfied, check your parameters."
def prothPrime(N):
        a = list(range(0,10))#create list in the range of 1 - 10.
        power = (N-1)/2 #get power after relevant computations.
        #a = 5**power + 1#(N-1)/2) + 1
        power_a = [i **power + 1 for i in a]#get all values of a raised to power.
        prime_check = [x % N == 0 for x in power_a]#iterate through all values in range 1 - 10 and check if they are divisible by N(our proth number) without a remainder.
        if(any(prime_check) == True): #If any value in the range returns as true then N is a Proth Prime number.
            return N, "Is a Proth Prime Number."
        else :
            return N, "Is not a Proth Prime Number."
print(prothPrime(17))
          #[(values**(N-1)/2) + 1 for values in a]
        #return N
#def isProthNumberandIsPrime(p,k):
    #n = n - 1
        #while(k % 2 == 0 and 2**n > k): #checks if k is odd and if 2^n is greater than k. 
        #N = (k*2**n) + 1
        #n_div_k = n//k #Use floor divisor to get absolute value of n/k. 
        #while (k != 0 and n_div_k > k): #Check if proth number. 
         #   if (n % k == 0):#use modulo operator to check if k divides n without a remainder. 
          #      if (n_div_k and (not(n_div_k & (n_div_k - 1)))):  #perform boolean and binary operation to check power.          
          #      #for i in range(0, 100):
                       #values = list(range(0,100))
                       #a = random.randint(1,100)
                       #g = []
                       #randlist.append(a)
                       #print(randlist)
                       #g = [ ((a**(N-1)/2) + 1) for a in values]
                       #if (g % n == 0):
           #            return g
            #    else: return "balse."      
            #k = k + 2
            
        #return False #always return False if condition is not satisfied.
#print(isPowerOfTwo(13))
#print(isProthNumberandIsPrime(13,3))
print(prothNumber(1, 1))

"""
A proth number is a number N of the form N = k*2**n + 1 where k and n are both positive integers,
k is odd and 2**n > K. A proth prime is a proth number that is prime(divisible by only itself and 1).
Named after french mathematician Francois Proth, the first few proth primes are:3,5,13,17,41,97,113.

conditions:
    1- if p is a proth number of the form (k*2**n) + 1 
    2- if k is odd and k (is less than) 2**n (k<(2**n)) - if these 2 are satisfied then we have a pro
    th  number. 
    3- if there exists an integer a for which a**(p-1)/2 == -1(mod p)
    
    then P is prime. 
    
    therefore a values are iterated separately, n values are same as p value
    e.g p = 3: where a[1,2,3], n = 3, 

Definition:
    A proth number takes the form N = k*2**n + 1 where k and n are both positive integers, k is odd 
    and 2**n > k. A proth prime is a proth number that is prime.
    
    Without the condition that 2**n > k, all odd integers larger than1 would be proth numbers. 


"""
import scipy.stats as si
def blackScholesMersonMethod(option_type, volatiility, risk_free_rate,time_to_expiry,strike_price,spot_price):
    T = time_to_expiry/365 #Time to maturity expressed in years.  
    S = spot_price #Spot price of underlying asset at time t.
    K = strike_price #Strike price for each share in our options contract.
    r = risk_free_rate #Risk free rate assumed to be a value between our T and t.
    D = volatiility #Volatility of returns of the underlying asset ,SD of datapoints in assets returns. 
    N = si.norm.cdf #Cumulative distribution function of a standard normal distribution with a mean = 0 and SD of 1. 
    deriv = (D**2)/2
    d1 =  np.log(S/K) + ((r+ 0.5 * deriv) * T) / (D * np.sqrt(T))
    d2 = d1 - (D * np.sqrt(T))
    power_PV = -r*T
    PV = K*np.exp(power_PV)
    if (option_type == "call"):
        first_half_eqn = N(d1,0,1) * S #normal distribution * spot price. 
        second_half_eqn = N(d2,0,1) * PV #normal distribution wth d2 multiplied by PV.
        C = first_half_eqn - second_half_eqn #get final call value.
        import scipy.stats as sc
        call = ((S * sc.norm.cdf(d1, 0,1)) - K * np.exp(power_PV) * sc.norm.cdf(d2,0,1)) 
        print("This is call:",call)
        C = first_half_eqn - second_half_eqn
        return C
    elif(option_type == "put"):
        P = PV * N(-d2,0,1) - S * N(-d1,0,1)
        put = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
        print("This is put:",put)
        return P
    
def rN():
    values = list(range(0,10))
    y = [np.sqrt(((x+6)**2) + 25) + np.sqrt(((x - 6)**2) + 121) for x in values]
    return y


print(blackScholesMersonMethod("call",0.32,0.04,40,60,62))
"""


#fwd_xg = np.array(df_fwd['xG'])
#fwd_sh = np.array(df_fwd['Sh'])
#goal_involvements = np.stack([fwd_xg, fwd_sh], axis=1)
"""
diff_gi_xgi = np.array(df_mid_underlying_performance['Difference (GI-xGI)'])
gi_per_gw = np.stack([df_xGI,df_xG, df_xA, diff_gi_xgi], axis=1)/38
gi = np.stack([df_xGI, df_xG, df_xA, diff_gi_xgi], axis=1)
y_GI_per_gw = np.array(df_mid_underlying_performance['Goal Involvements'])/38
y_GI = np.array(df_mid_underlying_performance['Goal Involvements'])


X, X_test, Y, Y_test = train_test_split(gi_per_gw,y_GI_per_gw,test_size=0.3)
x, x_test, y, y_test = train_test_split(gi, y_GI, test_size=0.3)

fig = plt.figure()
fpl_model_1 = fig.add_subplot(1,1,1)
fpl_model_1.plot(X, Y, "x", color="pink")
fpl_model_1.set_xlabel("XGI,XG,XA,DIFFXGI_XG")
fpl_model_1.set_ylabel("GI")
fpl_model_1.set_title("Fresh Per GameWeek Plot")
fig_model_2 = plt.figure()
fpl_model_2 = fig_model_2.add_subplot(1,1,1)
fpl_model_2.plot(x, y, "o", color="cyan")
fpl_model_2.set_title("Total Stats Plot")
fpl_model_2.set_xlabel("XGI,XG,XA,DIFF")
fpl_model_2.set_ylabel("GI")
model_linear.fit(X,Y)
model_k_nearest.fit(x,y)

lm = model_linear.predict(X_test)
knn = model_k_nearest.predict(x_test)
lin_mse = mean_squared_error(Y_test, lm) 
lin_rmse = np.sqrt(lin_mse)



show_lm = np.stack([lm, Y_test], axis=1)
show_knn = np.stack([knn, y_test], axis=1)



"""


#scores = cross_val_score(model_linear, df_fpl_threat, df_fpl_goals_scored, 
#                            scoring="neg_mean_squared_error", cv=10)
#tree_rmse_scores = np.sqrt(-scores)
   
ordinal_encoder = OrdinalEncoder()
array_val = pd.DataFrame([events['is_current'],events['id']])
ret = events.query("is_current == 1")['id']
is_curr = np.array(events['is_current'])
id_val = np.array(events['id'])
val_arr = np.stack([is_curr, id_val], axis=1)
"""
#df.loc[df['a'] == 1, 'b']
#vala = array_val.loc[array_val['is_current'] == 1]




#fpl model

"""



#Abstract Methods
def find(data_frame, column, search_term):
    df = data_frame
    for result in df[column]:
        if re.search(result, search_term):
            return df[df[column].isin(search_term)]

#vala = find(events, "is_current", 1)
    
#if(events['is_current'].bool()):
 #   print("yes")
#show = ordinal_encoder.fit_transform(array_val)
#print(array_val[0])
#for val in array_val:
    #val.loc['True']
    #print(id_val[val])
#val.loc[val[''] == some_value]
"""
#val = num_pipeline.fit_transform(grp)
val_corr = df_offensive_stats_all_players.corr()
overall_corr = df_team_overall_xG_xA.corr()
home_corr = df_team_home_xG_xA.corr()
away_corr = df_team_away_xG_xA.corr()
#x_norm = normalize(gm_min.reshape(-1,1))

df_goals_against = df_team_overall_xG_xA['GA']/38
df_xGA = df_team_overall_xG_xA['xGA']/38
df_losses = df_team_overall_xG_xA['L']+ df_team_home_xG_xA['D']
grp_team_defense_ovr = np.stack([df_losses, df_goals_against, df_xGA], axis=1)

df_goals_against_at_home = df_team_home_xG_xA['GA']/19
df_xGA_home = df_team_home_xG_xA['xGA']/19
df_losses_home = df_team_home_xG_xA['L'] + df_team_home_xG_xA['D']
grp_team_defense_home = np.stack([df_losses_home, df_goals_against_at_home, df_xGA_home], axis=1)

df_goals_against_away = df_team_away_xG_xA['GA']/19
df_xGA_away = df_team_away_xG_xA['xGA']/19
df_losses_away = df_team_away_xG_xA['L']+ df_team_home_xG_xA['D']
grp_team_defense_away = np.stack([df_losses_away, df_goals_against_away, df_xGA_away], axis=1)





#clean sheet and save models
gk_corr = df_squad_goalkeeping.corr()


"""

#Take site and structure html
page = requests.get('https://www.premierleague.com/clubs')
tree = html.fromstring(page.content)

#Using the page's CSS classes, extract all links pointing to a team
linkLocation = tree.cssselect('.indexItem')


#Create an empty list for us to send each team's link to
teamLinks = []

#For each link...
for i in range(0,20):
    
    #...Find the page the link is going to...
    temp = linkLocation[i].attrib['href']
    
    #...Add the link to the website domain...
    temp = "http://www.premierleague.com/" + temp
    
    #...Change the link text so that it points to the squad list, not the page overview...
    temp = temp.replace("overview", "squad")
    
    #...Add the finished link to our teamLinks list...
    teamLinks.append(temp)

#Create empty lists for player links
playerLink1 = []
playerLink2 = []

#For each team link page...
for i in range(len(teamLinks)):
    
    #...Download the team page and process the html code...
    squadPage = requests.get(teamLinks[i])
    squadTree = html.fromstring(squadPage.content)
    
    #...Extract the player links...
    playerLocation = squadTree.cssselect('.playerOverviewCard')

    #...For each player link within the team page...
    for i in range(len(playerLocation)):
        
        #...Save the link, complete with domain...
        playerLink1.append("http://www.premierleague.com/" + playerLocation[i].attrib['href'])
        
        #...For the second link, change the page from player overview to stats
        playerLink2.append(playerLink1[i].replace("overview", "stats"))

#Create lists for each variable
Name = []
Team = []
Age = []
Apps = []
HeightCM = []
WeightKG = []


#Populate lists with each player

#For each player...
for i in range(len(playerLink1)):

    #...download and process the two pages collected earlier...
    playerPage1 = requests.get(playerLink1[i])
    playerTree1 = html.fromstring(playerPage1.content)
    playerPage2 = requests.get(playerLink2[i])
    playerTree2 = html.fromstring(playerPage2.content)

    #...find the relevant datapoint for each player, starting with name...
    tempName = str(playerTree1.cssselect('div.name')[0].text_content())
    
    #...and team, but if there isn't a team, return "BLANK"...
    try:
        tempTeam = str(playerTree1.cssselect('.table:nth-child(1) .long')[0].text_content())
    except IndexError:
        tempTeam = str("BLANK")
    
    #...and age, but if this isn't there, leave a blank 'no number' number...
    try:  
        tempAge = int(playerTree1.cssselect('.pdcol2 li:nth-child(1) .info')[0].text_content())
    except IndexError:
        tempAge = float('NaN')

    #...and appearances. This is a bit of a mess on the page, so tidy it first...
    try:
        tempApps = playerTree2.cssselect('.statappearances')[0].text_content()
        tempApps = int(re.search(r'\d+', tempApps).group())
    except IndexError:
        tempApps = float('NaN')

    #...and height. Needs tidying again...
    try:
        tempHeight = playerTree1.cssselect('.pdcol3 li:nth-child(1) .info')[0].text_content()
        tempHeight = int(re.search(r'\d+', tempHeight).group())
    except IndexError:
        tempHeight = float('NaN')

    #...and weight. Same with tidying and returning blanks if it isn't there
    try:
        tempWeight = playerTree1.cssselect('.pdcol3 li+ li .info')[0].text_content()
        tempWeight = int(re.search(r'\d+', tempWeight).group())
    except IndexError:
        tempWeight = float('NaN')


    #Now that we have a player's full details - add them all to the lists
    Name.append(tempName)
    Team.append(tempTeam)
    Age.append(tempAge)
    Apps.append(tempApps)
    HeightCM.append(tempHeight)
    WeightKG.append(tempWeight)


#Create data frame from lists
df = pd.DataFrame(
    {'Name':Name,
     'Team':Team,
     'Age':Age,
     'Apps':Apps,
     'HeightCM':HeightCM,
     'WeightKG':WeightKG})


def get_highest_performer(squad_xG_per_gw, squad_goals_per_gw, squad_g_xg_per_gw):
    df_highest = np.stack([squad_xG_per_gw, squad_goals_per_gw, squad_g_xg_per_gw], axis=1) 
    
    return df_



#nid


def persistence():
   
    return 
#consistency: A CB can clear the ball 9 times and be crucified for one mistake.
def consistency():
    df_gk = df_offensive_stats_all_players.query("Pos == 'GK'")
    df_def = df_offensive_stats_all_players.query("Pos == 'DF'")
    return
    



Persistence and Consistency estimator (PACE)

- Break season into multiple sections by months and look at goals scored and conceded across seasons. 

- Break goalscorers, mids and defenders seasons by multiple sections of months and see the periods of most activity in terms of returns. 

- Weight correctly, account for strength of schedule using simple ratings system and other active systems out there. 

- Strength of schedule +  Season Activity sections + Team Quality (Goals Scored/Chances Created/Defense) 

- Assume a 3 - 6 game scale for form. A player in-form is likely to return high values in likely 3, at most 6 games before a return to their mean. 

- Track how many games in a season a player returns on average, as a percentage of total games on a scale of P(0 - 1).

- Track player mean return in a season, Track team mean return in a season.

- Set a team for 3 weeks at a stretch, only tinker after week 2.


- Make sure your features correlate strongly with your target which is points generated.
- Account for strength of schedule, it is a significant determinant of player return.
- I don’t want to go too deep into it because i am building my own model

A striker can miss 9 shots and be redeemed with one goal. A CB can clear the ball 9 times and be crucified for one mistake.
Strikers a lauded for persistency, defenders for consistency.




During the season, your fantasy football players will be allocated points based on their performance in the Premier League.
Action	Points
For playing up to 60 minutes	1
For playing 60 minutes or more (excluding stoppage time)	2
For each goal scored by a goalkeeper or defender	6
For each goal scored by a midfielder	5
For each goal scored by a forward	4
For each goal assist	3
For a clean sheet by a goalkeeper or defender	4
For a clean sheet by a midfielder	1
For every 3 shot saves by a goalkeeper	1
For each penalty save	5
For each penalty miss	-2
Bonus points for the best players in a match	1-3
For every 2 goals conceded by a goalkeeper or defender	-1
For each yellow card	-1
For each red card	-3
For each own goal	-2
Clean sheets

A clean sheet is awarded for not conceding a goal whilst on the pitch and playing at least 60 minutes (excluding stoppage time).

If a player has been substituted when a goal is conceded this will not affect any clean sheet bonus.
Red Cards

If a player receives a red card, they will continue to be penalised for goals conceded by their team.

Red card deductions include any points deducted for yellow cards.
Assists

Assists are awarded to the player from the goal scoring team, who makes the final pass before a goal is scored. An assist is awarded whether the pass was intentional (that it actually creates the chance) or unintentional (that the player had to dribble the ball or an inadvertent touch or shot created the chance).

If an opposing player touches the ball after the final pass before a goal is scored, significantly altering the intended destination of the ball, then no assist is awarded. Should a touch by an opposing player be followed by a defensive error by another opposing outfield player then no assist will be awarded. If the goal scorer loses and then regains possession, then no assist is awarded.
Rebounds

If a shot on goal is blocked by an opposition player, is saved by a goalkeeper or hits the woodwork, and a goal is scored from the rebound, then an assist is awarded.
Own Goals

If a player shoots or passes the ball and forces an opposing player to put the ball in his own net, then an assist is awarded.
Penalties and Free-Kicks

In the event of a penalty or free-kick, the player earning the penalty or free-kick is awarded an assist if a goal is directly scored, but not if he takes it himself, in which case no assist is given.
Finalising Assists

Assist points awarded by Opta within Fantasy Premier League are calculated using additional stats which may differ from other websites. For example, some other sites would not show an assist where a player has won a penalty.

For the avoidance of doubt, points awarded in-game are subject to change up until one hour after the final whistle of the last match of any given day. Once the points have all been updated on that day, no further adjustments to points will be made.
Bonus Points

The Bonus Points System (BPS) utilises a range of statistics to create a BPS score for every player. The three best performing players in each match will be awarded bonus points. 3 points will be awarded to the highest scoring player, 2 to the second best and 1 to the third.

Examples of how bonus point ties will be resolved are as follows:

    If there is a tie for first place, Players 1 & 2 will receive 3 points each and Player 3 will receive 1 point.
    If there is a tie for second place, Player 1 will receive 3 points and Players 2 and 3 will receive 2 points each.
    If there is a tie for third place, Player 1 will receive 3 points, Player 2 will receive 2 points and Players 3 & 4 will receive 1 point each.

How is the BPS score calculated?

Players score BPS points based on the following statistics (one point for each unless otherwise stated):
Action	BPS
Playing 1 to 60 minutes	3
Playing over 60 minutes	6
Goalkeepers and defenders scoring a goal	12
Midfielders scoring a goal	18
Forwards scoring a goal	24
Assists	9
Goalkeepers and defenders keeping a clean sheet	12
Saving a penalty	15
Save	2
Successful open play cross	1
Creating a big chance (a chance where the receiving player should score)	3
For every 2 clearances, blocks and interceptions (total)	1
For every 3 recoveries	1
Key pass	1
Successful tackle (net*)	2
Successful dribble	1
Scoring the goal that wins a match	3
70 to 79% pass completion (at least 30 passes attempted)	2
80 to 89% pass completion (at least 30 passes attempted)	4
90%+ pass completion (at least 30 passes attempted)	6
Conceding a penalty	-3
Missing a penalty	-6
Yellow card	-3
Red card	-9
Own goal	-6
Missing a big chance	-3
Making an error which leads to a goal	-3
Making an error which leads to an attempt at goal	-1
Being tackled	-1
Conceding a foul	-1
Being caught offside	-1
Shot off target	-1

*Net successful tackles is the total of all successful tackles minus any unsuccessful tackles. Players will not be awarded negative BPS points for this statistic.

Data is supplied by Opta and once it has been marked as final will not be changed. We will not enter into discussion around any of the statistics used to calculate this score for any individual match.





The opposition's team rating and home ground advantage are the main factors in grading the difficulty of the fixture.
 The 'Next 5' Schedule Difficulty Rating is an aggregation of the team’s next 5 fixtures.

Note: An FPL season is a marathon not a sprint - 38 gameweeks spread typically across 9 months.
We all enjoy immediate rewards and to be doing well at something right from the start, but it's important to take a long-term view with almost every decision in FPL



Now let’s look at this another way. To win FPL you typically need ~2500 pts. To do extremely well (to finish in the top 1k) you typically need ~2400 pts. Here’s how that could realistically be broken down:
• 2 super elite players - 240 pts each
• Captaincy (incl. TC)
• Keeper
• 1 elite defender (e.g. Trent / Robbo)
• Bench Boost and Free Hit
• 2 cheap but effective outfield players
480 pts 280 pts 150 pts 200 pts
40 pts 250 pts
At this point we have 6 players and we are at 1400 pts and need another 1000 pts to make our 2400 pts target for a fantastic season. I think all the above figures are realistically achievable. That leaves us needing 5 other players that average 200 pts each! That is where things get tricky as the budget will not stretch to these 5 being all premium expensive assets.
• 5 remaining players 1000 pts
To achieve this then, we need to nail some cheaper upside picks (e.g. Jimenez 18/19, Ings 19/20, Robertson 18/19, Mahrez 15/16 etc). That sounds extremely difficult, but we don’t necessarily need to nail them first time, we just need to hit the ground running with our initial picks and then shift to better assets in a timely fashion as and when they emerge.
What I’m circling back around to is that if you pick 5 safe players who should be worth their prices, but don’t really have an upside case in which they get close to 180- 200 pts, you are almost locking yourself into a moderate season.

"""





