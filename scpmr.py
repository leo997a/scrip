# Import Packages
import json
import re
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
import seaborn as sns
import requests
import matplotlib.patches as patches
from mplsoccer import Pitch, VerticalPitch, add_image
# from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# from matplotlib.patheffects import withStroke, Normal
from matplotlib.colors import LinearSegmentedColormap
# from mplsoccer.utils import FontManager
import matplotlib.patheffects as path_effects
# from sklearn.cluster import KMeans
from highlight_text import ax_text, fig_text
from PIL import Image
from urllib.request import urlopen
from unidecode import unidecode
from scipy.spatial import ConvexHull
import streamlit as st
import os

green = '#69f900'
red = '#ff4b44'
blue = '#00a0de'
violet = '#a369ff'
bg_color= '#f5f5f5'
line_color= '#000000'
col1 = '#ff4b44'
col2 = '#00a0de'

# Set up session state for selected values
if 'confirmed' not in st.session_state:
    st.session_state.confirmed = False
    
def reset_confirmed():
    st.session_state['confirmed'] = False

uploaded_files = st.file_uploader(
    "Choose a CSV file", accept_multiple_files=False
)

    
if uploaded_files is not None:

    # Can be used wherever a "file-like" object is accepted:
    df = pd.read_csv(uploaded_files)
    df.loc[(df['type'] == 'Carry') & (df['name'].isna()) & (df['playerId']==df['playerId'].shift(-1)), 'name'] = df['name'].shift(-1)
    
    

    home_away_df = df.head(2)
    home_away_df = home_away_df[['teamName', 'h_a']]
    home_away_df = home_away_df.sort_values(by='h_a', ascending=True).reset_index(drop=True)
    hteamName = home_away_df['teamName'][1]
    ateamName = home_away_df['teamName'][0]
    
    homedf = df[df['teamName']==hteamName]
    awaydf = df[df['teamName']==ateamName]
    
    if 'type_value_Own goal' not in df.columns:
        df['type_value_Own goal'] = 0
        
    score_df = df[['type', 'minute', 'type_value_Own goal', 'name', 'teamName']]
    score_df = score_df[score_df['type']=='Goal'].reset_index(drop=True)
    score_df['type_value_Own goal'] = score_df['type_value_Own goal'].fillna(0)
    
    h_goal = score_df[(score_df['teamName']==hteamName) & (score_df['type_value_Own goal']==0)]
    h_og = score_df[(score_df['teamName']==hteamName) & (score_df['type_value_Own goal']!=0)]
    a_goal = score_df[(score_df['teamName']==ateamName) & (score_df['type_value_Own goal']==0)]
    a_og = score_df[(score_df['teamName']==ateamName) & (score_df['type_value_Own goal']!=0)]
    
    hgoal_count = len(h_goal) + len(a_og)
    agoal_count = len(a_goal) + len(h_og)
    
    hpnames = homedf['name'].unique()
    apnames = awaydf['name'].unique()
    
    home_Forward = st.selectbox("Select Home Team's Forward name:", hpnames, key='home_fwd', index=None, on_change=reset_confirmed)
    away_Forward = st.selectbox("Select Away Team's Forward name:", apnames, key='away_fwd', index=None, on_change=reset_confirmed)
    if home_Forward and away_Forward:
        homeGK = st.selectbox("Select Home Team's Goal Keeper name:", hpnames, key='home_gk', index=None, on_change=reset_confirmed)
        awayGK = st.selectbox("Select Away Team's Goal Keeper name:", apnames, key='away_gk', index=None, on_change=reset_confirmed)
        if homeGK and awayGK:
            league_name = st.text_area('Write Competition Name', on_change=reset_confirmed)
            match_input = st.button('Confirm', on_click=lambda: st.session_state.update({'confirmed': True}))
   
    

    if home_Forward and away_Forward and homeGK and awayGK and league_name and st.session_state.confirmed:
        
        # ShortName s
         
        def get_short_name(full_name):
            if pd.isna(full_name):
                return full_name
            parts = full_name.split()
            if len(parts) == 1:
                return full_name  # No need for short name if there's only one word
            elif len(parts) == 2:
                return parts[0][0] + ". " + parts[1]
            else:
                return parts[0][0] + ". " + parts[1][0] + ". " + " ".join(parts[2:])
            
        # Pass Network
            
        df['pass_receiver'] = df.loc[(df['type'] == 'Pass') & (df['outcomeType'] == 'Successful'), 'name'].shift(-1)
        df['pass_receiver'] = df['pass_receiver'].fillna('No')
        
        def pass_network(ax, team_name, col):
            pass_df_full = df[(df['type']=='Pass') & (df['outcomeType']=='Successful') & (df['teamName']==team_name) &
                         (df['type_value_Corner taken']!=6) & (df['type_value_Free kick taken']!=5)]
            pass_df = pass_df_full[['type', 'name', 'pass_receiver']].reset_index(drop=True)
        
            pass_counts_df = pass_df.groupby(['name', 'pass_receiver']).size().reset_index(name='pass_count').sort_values(by='pass_count', ascending=False)
            pass_counts_df = pass_counts_df.reset_index(drop=True)
        
            team_df = df[(df['teamName']==team_name) & (df['type']!='OffsidePass') & (df['type']!='CornerAwarded') & (df['type']!='SubstitutionOff') &
                         (df['type']!='SubstitutionOff') & (df['type']!='SubstitutionOn') & (df['type']!='Card')]
            team_df = team_df[['name', 'x', 'y']].reset_index(drop=True)
            avg_locs_df = team_df.groupby('name').agg(avg_x=('x', 'median'), avg_y=('y', 'median')).reset_index()
        
            pass_counts_df = pd.merge(pass_counts_df, avg_locs_df, on='name', how='left')
            pass_counts_df.rename(columns={'avg_x': 'avg_x', 'avg_y': 'avg_y'}, inplace=True)
            pass_counts_df = pd.merge(pass_counts_df, avg_locs_df, left_on='pass_receiver', right_on='name', how='left', suffixes=('', '_receiver'))
            pass_counts_df.drop(columns=['name_receiver'], inplace=True)
            pass_counts_df.rename(columns={'avg_x_receiver': 'receiver_avg_x', 'avg_y_receiver': 'receiver_avg_y'}, inplace=True)
            pass_counts_df = pass_counts_df.sort_values(by='pass_count', ascending=False).reset_index(drop=True)
            
            # avg_locs_df['shortName'] = avg_locs_df['name'].apply(get_short_name)
            avg_locs_df['short_name'] = avg_locs_df['name'].apply(lambda x: ''.join([name[0] + '' for name in x.split()]))
        
            MAX_LINE_WIDTH = 15
            MAX_MARKER_SIZE = 3000
            pass_counts_df['width'] = (pass_counts_df.pass_count / pass_counts_df.pass_count.max() *MAX_LINE_WIDTH)
            # avg_locs_df['marker_size'] = (avg_locs_df['count']/ avg_locs_df['count'].max() * MAX_MARKER_SIZE) #You can plot variable size of each player's node according to their passing volume, in the plot using this
            MIN_TRANSPARENCY = 0.05
            MAX_TRANSPARENCY = 0.85
            color = np.array(to_rgba(col))
            color = np.tile(color, (len(pass_counts_df), 1))
            c_transparency = pass_counts_df.pass_count / pass_counts_df.pass_count.max()
            c_transparency = (c_transparency * (MAX_TRANSPARENCY - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
            color[:, 3] = c_transparency
        
            pitch = Pitch(pitch_type='uefa', line_color=line_color, pitch_color=bg_color, linewidth=2)
            pitch.draw(ax=ax)
            ax.set_xlim(-0.5, 105.5)
            # ax.set_ylim(-0.5, 68.5)
        
            # Plotting those lines between players
            pass_lines = pitch.lines(pass_counts_df.avg_x, pass_counts_df.avg_y, pass_counts_df.receiver_avg_x, pass_counts_df.receiver_avg_y,
                                     lw=pass_counts_df.width, color=color, zorder=2, ax=ax)
            
            # Plotting the player nodes
            sub_list = df[(df['type']=='SubstitutionOn') & (df['teamName']==team_name)].name.to_list()
            for index, row in avg_locs_df.iterrows():
                if row['name'] in sub_list:
                    pass_nodes = pitch.scatter(row['avg_x'], row['avg_y'], s=750, marker='s', color=bg_color, edgecolor=line_color, zorder=3, linewidth=2, ax=ax)
                else:
                    pass_nodes = pitch.scatter(row['avg_x'], row['avg_y'], s=1000, marker='o', color=bg_color, edgecolor=line_color, zorder=3, linewidth=2, ax=ax)
        
            # Plotting the shirt no. of each player
            for index, row in avg_locs_df.iterrows():
                player_initials = row["short_name"]
                pitch.annotate(player_initials, xy=(row.avg_x, row.avg_y), c=col, ha='center', va='center', size=10, ax=ax)
        
            # Plotting a vertical line to show the median vertical position of all passes
            avgph = round(avg_locs_df['avg_x'].median(), 2)
            # avgph_show = round((avgph*1.05),2)
            avgph_show = avgph
            ax.vlines(x=avgph, ymin=0, ymax=68, color='gray', linestyle='--', zorder=1, alpha=0.75, linewidth=2)
        
            if team_name == hteamName:
                ax.text(52.5, -5, f"avg. passing height: {avgph_show}m", fontsize=15, color=line_color, ha='center')
        
            else:
                ax.invert_xaxis()
                ax.invert_yaxis()
                ax.text(52.5, 73, f"avg. passing height: {avgph_show}m", fontsize=15, color=line_color, ha='center')
        
            # ax.text(2,66, "circle = starter\nbox = sub", color=col, size=12, ha='left', va='top')
            ax.set_title(f"{team_name}\nPassing Network", color=line_color, size=25, fontweight='bold')
        
            
            return 
        
        # Defensive Heatmap
        
        def defensive_heatmap(ax, team_name, col):
            defensive_actions_ids = df.index[((df['type'] == 'Aerial') & (df['type_value_Defensive'] == 285)) |
                                             (df['type'] == 'BallRecovery') |
                                             (df['type'] == 'BlockedPass') |
                                             (df['type'] == 'Challenge') |
                                             (df['type'] == 'Clearance') |
                                             (df['type'] == 'Error') |
                                             ((df['type'] == 'Foul') & (df['outcomeType']=='Unsuccessful')) |
                                             (df['type'] == 'Interception') |
                                             (df['type'] == 'Tackle')]
            df_defensive_actions = df.loc[defensive_actions_ids, ["x", "y", "teamName", "name", "type", "outcomeType"]]
            df_defensive_actions = df_defensive_actions[df_defensive_actions['teamName']==team_name].reset_index(drop=True)
        
            average_locs_and_count_df = (df_defensive_actions.groupby('name').agg({'x': ['median'], 'y': ['median', 'count']}))
            average_locs_and_count_df.columns = ['x', 'y', 'count']
            average_locs_and_count_df = average_locs_and_count_df.reset_index()
            average_locs_and_count_df['short_name'] = average_locs_and_count_df['name'].apply(lambda x: ''.join([name[0] + '' for name in x.split()]))
        
        
            pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, line_zorder=2, corner_arcs=True)
            pitch.draw(ax=ax)
            ax.set_facecolor(bg_color)
            ax.set_xlim(-0.5, 105.5)
            # ax.set_ylim(-0.5, 68.5)
        
            # using variable marker size for each player according to their defensive engagements
            MAX_MARKER_SIZE = 3500
            average_locs_and_count_df['marker_size'] = (average_locs_and_count_df['count']/ average_locs_and_count_df['count'].max() * MAX_MARKER_SIZE)
            # plotting the heatmap of the team defensive actions
            color = np.array(to_rgba(col))
            flamingo_cmap = LinearSegmentedColormap.from_list("Flamingo - 100 colors", [bg_color, col], N=500)
            kde = pitch.kdeplot(df_defensive_actions.x, df_defensive_actions.y, ax=ax, fill=True, levels=5000, thresh=0.02, cut=4, cmap=flamingo_cmap)
        
            # using different node marker for starting and substitute players
            sub_list = df[(df['type']=='SubstitutionOn') & (df['teamName']==team_name)].name.to_list()
            # average_locs_and_count_df = average_locs_and_count_df.reset_index(drop=True)
            for index, row in average_locs_and_count_df.iterrows():
                if row['name'] in sub_list:
                    da_nodes = pitch.scatter(row['x'], row['y'], s=row['marker_size']+75, marker='s', color=bg_color, edgecolor=line_color, linewidth=1, 
                                          zorder=3, ax=ax)
                else:
                    da_nodes = pitch.scatter(row['x'], row['y'], s=row['marker_size']+100, marker='o', color=bg_color, edgecolor=line_color, linewidth=1, 
                                              zorder=3, ax=ax)
            # plotting very tiny scatterings for the defensive actions
            da_scatter = pitch.scatter(df_defensive_actions.x, df_defensive_actions.y, s=10, marker='x', color='yellow', alpha=0.2, ax=ax)
        
            # Plotting the shirt no. of each player
            for index, row in average_locs_and_count_df.iterrows():
                player_initials = row["short_name"]
                pitch.annotate(player_initials, xy=(row.x, row.y), c=line_color, ha='center', va='center', size=10, ax=ax)
        
            # Plotting a vertical line to show the median vertical position of all defensive actions, which is called Defensive Actions Height
            dah = round(average_locs_and_count_df['x'].mean(), 2)
            dah_show = round((dah*1.05), 2)
            ax.vlines(x=dah, ymin=0, ymax=68, color='gray', linestyle='--', alpha=0.75, linewidth=2)
        
            if team_name == hteamName:
                ax.text(52.5, -5, f"avg. def. action height: {dah}m", fontsize=15, color=line_color, ha='center')
        
            else:
                ax.invert_xaxis()
                ax.invert_yaxis()
                ax.text(52.5, 73, f"avg. def. action height: {dah}m", fontsize=15, color=line_color, ha='center')
        
            # ax.text(2,66, "circle = starter\nbox = sub", color=col, size=12, ha='left', va='top')
            ax.set_title(f"{team_name}\nDefensive Action Heatmap", color=line_color, size=25, fontweight='bold')
        
            return
        
        
        # Progressive Pass
        
        def draw_progressive_pass_map(ax, team_name, col):
            dfpro = df[(df['teamName']==team_name) & (df['prog_pass']>=9.11) & (df['type_value_Corner taken']!=6) & (df['type_value_Free kick taken']!=5) & 
            (df['x']>=35) & (df['outcomeType']=='Successful')]
            
            pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2, corner_arcs=True)
            pitch.draw(ax=ax)
            ax.set_xlim(-0.5, 105.5)
            # ax.set_ylim(-0.5, 68.5)
        
            if team_name == ateamName:
                ax.invert_xaxis()
                ax.invert_yaxis()
        
            pro_count = len(dfpro)
        
            # calculating the counts
            left_pro = len(dfpro[dfpro['y']>=45.33])
            mid_pro = len(dfpro[(dfpro['y']>=22.67) & (dfpro['y']<45.33)])
            right_pro = len(dfpro[(dfpro['y']>=0) & (dfpro['y']<22.67)])
            left_percentage = round((left_pro/pro_count)*100)
            mid_percentage = round((mid_pro/pro_count)*100)
            right_percentage = round((right_pro/pro_count)*100)
        
            ax.hlines(22.67, xmin=0, xmax=105, colors=line_color, linestyle='dashed', alpha=0.35)
            ax.hlines(45.33, xmin=0, xmax=105, colors=line_color, linestyle='dashed', alpha=0.35)
        
            # showing the texts in the pitch
            bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="None", facecolor=bg_color, alpha=0.75)
            if col == col1:
                ax.text(8, 11.335, f'{right_pro}\n({right_percentage}%)', color=col1, fontsize=24, va='center', ha='center', bbox=bbox_props)
                ax.text(8, 34, f'{mid_pro}\n({mid_percentage}%)', color=col1, fontsize=24, va='center', ha='center', bbox=bbox_props)
                ax.text(8, 56.675, f'{left_pro}\n({left_percentage}%)', color=col1, fontsize=24, va='center', ha='center', bbox=bbox_props)
            else:
                ax.text(8, 11.335, f'{right_pro}\n({right_percentage}%)', color=col2, fontsize=24, va='center', ha='center', bbox=bbox_props)
                ax.text(8, 34, f'{mid_pro}\n({mid_percentage}%)', color=col2, fontsize=24, va='center', ha='center', bbox=bbox_props)
                ax.text(8, 56.675, f'{left_pro}\n({left_percentage}%)', color=col2, fontsize=24, va='center', ha='center', bbox=bbox_props)
        
            # plotting the passes
            pro_pass = pitch.lines(dfpro.x, dfpro.y, dfpro.endX, dfpro.endY, lw=3.5, comet=True, color=col, ax=ax, alpha=0.5)
            # plotting some scatters at the end of each pass
            pro_pass_end = pitch.scatter(dfpro.endX, dfpro.endY, s=35, edgecolor=col, linewidth=1, color=bg_color, zorder=2, ax=ax)
        
            counttext = f"{pro_count} Progressive Passes"
        
            # Heading and other texts
            if col == col1:
                ax.set_title(f"{hteamName}\n{counttext}", color=line_color, fontsize=25, fontweight='bold')
            else:
                ax.set_title(f"{ateamName}\n{counttext}", color=line_color, fontsize=25, fontweight='bold')
        
            return 
        
        
        # Progressive Carry
        
        def draw_progressive_carry_map(ax, team_name, col):
            dfpro = df[(df['teamName']==team_name) & (df['prog_carry']>=9.11) & (df['endX']>=35)]
            
            pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2,
                                  corner_arcs=True)
            pitch.draw(ax=ax)
            ax.set_xlim(-0.5, 105.5)
            # ax.set_ylim(-5, 68.5)
        
            if team_name == ateamName:
                ax.invert_xaxis()
                ax.invert_yaxis()
        
            pro_count = len(dfpro)
        
            # calculating the counts
            left_pro = len(dfpro[dfpro['y']>=45.33])
            mid_pro = len(dfpro[(dfpro['y']>=22.67) & (dfpro['y']<45.33)])
            right_pro = len(dfpro[(dfpro['y']>=0) & (dfpro['y']<22.67)])
            left_percentage = round((left_pro/pro_count)*100)
            mid_percentage = round((mid_pro/pro_count)*100)
            right_percentage = round((right_pro/pro_count)*100)
        
            ax.hlines(22.67, xmin=0, xmax=105, colors=line_color, linestyle='dashed', alpha=0.35)
            ax.hlines(45.33, xmin=0, xmax=105, colors=line_color, linestyle='dashed', alpha=0.35)
        
            # showing the texts in the pitch
            bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="None", facecolor=bg_color, alpha=0.75)
            if col == col1:
                ax.text(8, 11.335, f'{right_pro}\n({right_percentage}%)', color=col1, fontsize=24, va='center', ha='center', bbox=bbox_props)
                ax.text(8, 34, f'{mid_pro}\n({mid_percentage}%)', color=col1, fontsize=24, va='center', ha='center', bbox=bbox_props)
                ax.text(8, 56.675, f'{left_pro}\n({left_percentage}%)', color=col1, fontsize=24, va='center', ha='center', bbox=bbox_props)
            else:
                ax.text(8, 11.335, f'{right_pro}\n({right_percentage}%)', color=col2, fontsize=24, va='center', ha='center', bbox=bbox_props)
                ax.text(8, 34, f'{mid_pro}\n({mid_percentage}%)', color=col2, fontsize=24, va='center', ha='center', bbox=bbox_props)
                ax.text(8, 56.675, f'{left_pro}\n({left_percentage}%)', color=col2, fontsize=24, va='center', ha='center', bbox=bbox_props)
        
            # plotting the carries
            for index, row in dfpro.iterrows():
                arrow = patches.FancyArrowPatch((row['x'], row['y']), (row['endX'], row['endY']), arrowstyle='->', color=col, zorder=4, mutation_scale=20, 
                                                alpha=0.9, linewidth=2, linestyle='--')
                ax.add_patch(arrow)
        
            counttext = f"{pro_count} Progressive Carries"
        
            # Heading and other texts
            if col == col1:
                ax.set_title(f"{hteamName}\n{counttext}", color=line_color, fontsize=25, fontweight='bold')
            else:
                ax.set_title(f"{ateamName}\n{counttext}", color=line_color, fontsize=25, fontweight='bold')
        
            return 
        
        
        # ShotMap
        
        Shotsdf = df[(df['type'] == 'Goal') | (df['type'] == 'MissedShots') | (df['type'] == 'SavedShot') | (df['type'] == 'ShotOnPost')]
        Shotsdf = Shotsdf.reset_index(drop=True)
        
        # filtering according to the types of shots
        hShotsdf = Shotsdf[Shotsdf['teamName']==hteamName]
        aShotsdf = Shotsdf[Shotsdf['teamName']==ateamName]
        hSavedf = hShotsdf[(hShotsdf['type']=='SavedShot') & (hShotsdf['type_value_Blocked']!=82)]
        aSavedf = aShotsdf[(aShotsdf['type']=='SavedShot') & (aShotsdf['type_value_Blocked']!=82)]
        hogdf = hShotsdf[(hShotsdf['teamName']==hteamName) & (hShotsdf['type_value_Own goal']==28)]
        aogdf = aShotsdf[(aShotsdf['teamName']==ateamName) & (aShotsdf['type_value_Own goal']==28)]
        
        # Center Goal point
        given_point = (105, 34)
        # Calculate distances
        home_shot_distances = np.sqrt((hShotsdf['x'] - given_point[0])**2 + (hShotsdf['y'] - given_point[1])**2)
        home_average_shot_distance = round(home_shot_distances.mean(),2)
        away_shot_distances = np.sqrt((aShotsdf['x'] - given_point[0])**2 + (aShotsdf['y'] - given_point[1])**2)
        away_average_shot_distance = round(away_shot_distances.mean(),2)
        
        def plot_shotmap(ax):
            pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, linewidth=2, line_color=line_color)
            pitch.draw(ax=ax)
            ax.set_ylim(-0.5,68.5)
            ax.set_xlim(-0.5,105.5)
            
            #shooting stats
            hTotalShots = len(hShotsdf)
            aTotalShots = len(aShotsdf)
            hShotsOnT = len(hSavedf) + hgoal_count
            aShotsOnT = len(aSavedf) + agoal_count
            
            # without big chances for home team
            hGoalData = Shotsdf[(Shotsdf['teamName'] == hteamName) & (Shotsdf['type'] == 'Goal') & (Shotsdf['type_value_Big Chance']!=214)]
            hPostData = Shotsdf[(Shotsdf['teamName'] == hteamName) & (Shotsdf['type'] == 'ShotOnPost') & (Shotsdf['type_value_Big Chance']!=214)]
            hSaveData = Shotsdf[(Shotsdf['teamName'] == hteamName) & (Shotsdf['type'] == 'SavedShot') & (Shotsdf['type_value_Big Chance']!=214)]
            hMissData = Shotsdf[(Shotsdf['teamName'] == hteamName) & (Shotsdf['type'] == 'MissedShots') & (Shotsdf['type_value_Big Chance']!=214)]
            # only big chances of home team
            Big_C_hGoalData = Shotsdf[(Shotsdf['teamName'] == hteamName) & (Shotsdf['type'] == 'Goal') & (Shotsdf['type_value_Big Chance']==214)]
            Big_C_hPostData = Shotsdf[(Shotsdf['teamName'] == hteamName) & (Shotsdf['type'] == 'ShotOnPost') & (Shotsdf['type_value_Big Chance']==214)]
            Big_C_hSaveData = Shotsdf[(Shotsdf['teamName'] == hteamName) & (Shotsdf['type'] == 'SavedShot') & (Shotsdf['type_value_Big Chance']==214)]
            Big_C_hMissData = Shotsdf[(Shotsdf['teamName'] == hteamName) & (Shotsdf['type'] == 'MissedShots') & (Shotsdf['type_value_Big Chance']==214)]
            total_bigC_home = len(Big_C_hGoalData) + len(Big_C_hPostData) + len(Big_C_hSaveData) + len(Big_C_hMissData)
            bigC_miss_home = len(Big_C_hPostData) + len(Big_C_hSaveData) + len(Big_C_hMissData)
            # normal shots scatter of home team
            sc2 = pitch.scatter((105-hPostData.x), (68-hPostData.y), s=200, edgecolors=col1, c=col1, marker='o', ax=ax)
            sc3 = pitch.scatter((105-hSaveData.x), (68-hSaveData.y), s=200, edgecolors=col1, c='None', hatch='///////', marker='o', ax=ax)
            sc4 = pitch.scatter((105-hMissData.x), (68-hMissData.y), s=200, edgecolors=col1, c='None', marker='o', ax=ax)
            sc1 = pitch.scatter((105-hGoalData.x), (68-hGoalData.y), s=350, edgecolors='green', linewidths=0.6, c='None', marker='football', zorder=3, ax=ax)
            sc1_og = pitch.scatter((105-hogdf.x), (68-hogdf.y), s=350, edgecolors='orange', linewidths=0.6, c='None', marker='football', zorder=3, ax=ax)
            # big chances bigger scatter of home team
            bc_sc2 = pitch.scatter((105-Big_C_hPostData.x), (68-Big_C_hPostData.y), s=500, edgecolors=col1, c=col1, marker='o', ax=ax)
            bc_sc3 = pitch.scatter((105-Big_C_hSaveData.x), (68-Big_C_hSaveData.y), s=500, edgecolors=col1, c='None', hatch='///////', marker='o', ax=ax)
            bc_sc4 = pitch.scatter((105-Big_C_hMissData.x), (68-Big_C_hMissData.y), s=500, edgecolors=col1, c='None', marker='o', ax=ax)
            bc_sc1 = pitch.scatter((105-Big_C_hGoalData.x), (68-Big_C_hGoalData.y), s=650, edgecolors='green', linewidths=0.6, c='None', marker='football', ax=ax)
        
            # without big chances for away team
            aGoalData = Shotsdf[(Shotsdf['teamName'] == ateamName) & (Shotsdf['type'] == 'Goal') & (Shotsdf['type_value_Big Chance']!=214)]
            aPostData = Shotsdf[(Shotsdf['teamName'] == ateamName) & (Shotsdf['type'] == 'ShotOnPost') & (Shotsdf['type_value_Big Chance']!=214)]
            aSaveData = Shotsdf[(Shotsdf['teamName'] == ateamName) & (Shotsdf['type'] == 'SavedShot') & (Shotsdf['type_value_Big Chance']!=214)]
            aMissData = Shotsdf[(Shotsdf['teamName'] == ateamName) & (Shotsdf['type'] == 'MissedShots') & (Shotsdf['type_value_Big Chance']!=214)]
            # only big chances of away team
            Big_C_aGoalData = Shotsdf[(Shotsdf['teamName'] == ateamName) & (Shotsdf['type'] == 'Goal') & (Shotsdf['type_value_Big Chance']==214)]
            Big_C_aPostData = Shotsdf[(Shotsdf['teamName'] == ateamName) & (Shotsdf['type'] == 'ShotOnPost') & (Shotsdf['type_value_Big Chance']==214)]
            Big_C_aSaveData = Shotsdf[(Shotsdf['teamName'] == ateamName) & (Shotsdf['type'] == 'SavedShot') & (Shotsdf['type_value_Big Chance']==214)]
            Big_C_aMissData = Shotsdf[(Shotsdf['teamName'] == ateamName) & (Shotsdf['type'] == 'MissedShots') & (Shotsdf['type_value_Big Chance']==214)]
            total_bigC_away = len(Big_C_aGoalData) + len(Big_C_aPostData) + len(Big_C_aSaveData) + len(Big_C_aMissData)
            bigC_miss_away = len(Big_C_aPostData) + len(Big_C_aSaveData) + len(Big_C_aMissData)
            # normal shots scatter of away team
            sc6 = pitch.scatter(aPostData.x, aPostData.y, s=200, edgecolors=col2, c=col2, marker='o', ax=ax)
            sc7 = pitch.scatter(aSaveData.x, aSaveData.y, s=200, edgecolors=col2, c='None', hatch='///////', marker='o', ax=ax)
            sc8 = pitch.scatter(aMissData.x, aMissData.y, s=200, edgecolors=col2, c='None', marker='o', ax=ax)
            sc5 = pitch.scatter(aGoalData.x, aGoalData.y, s=350, edgecolors='green', linewidths=0.6, c='None', marker='football', zorder=3, ax=ax)
            sc5_og = pitch.scatter((aogdf.x), (aogdf.y), s=350, edgecolors='orange', linewidths=0.6, c='None', marker='football', zorder=3, ax=ax)
            # big chances bigger scatter of away team
            bc_sc6 = pitch.scatter(Big_C_aPostData.x, Big_C_aPostData.y, s=700, edgecolors=col2, c=col2, marker='o', ax=ax)
            bc_sc7 = pitch.scatter(Big_C_aSaveData.x, Big_C_aSaveData.y, s=700, edgecolors=col2, c='None', hatch='///////', marker='o', ax=ax)
            bc_sc8 = pitch.scatter(Big_C_aMissData.x, Big_C_aMissData.y, s=700, edgecolors=col2, c='None', marker='o', ax=ax)
            bc_sc5 = pitch.scatter(Big_C_aGoalData.x, Big_C_aGoalData.y, s=850, edgecolors='green', linewidths=0.6, c='None', marker='football', ax=ax)
        
            # sometimes the both teams ends the match 0-0, then normalizing the data becomes problem, thats why this part of the code
            if hgoal_count+agoal_count == 0:
               hgoal = 10
               agoal = 10
            else:
               hgoal = (hgoal_count/(hgoal_count+agoal_count))*20
               agoal = (agoal_count/(hgoal_count+agoal_count))*20
                
            if total_bigC_home+total_bigC_away == 0:
               total_bigC_home_n = 10
               total_bigC_away_n = 10
            else:
               total_bigC_home_n = (total_bigC_home/(total_bigC_home+total_bigC_away))*20
               total_bigC_away_n = (total_bigC_away/(total_bigC_home+total_bigC_away))*20
                
            if bigC_miss_home+bigC_miss_away == 0:
               bigC_miss_home_n = 10
               bigC_miss_away_n = 10
            else:
               bigC_miss_home_n = (bigC_miss_home/(bigC_miss_home+bigC_miss_away))*20
               bigC_miss_away_n = (bigC_miss_away/(bigC_miss_home+bigC_miss_away))*20
        
            if hShotsOnT+aShotsOnT == 0:
               hShotsOnT_n = 10
               aShotsOnT_n = 10
            else:
               hShotsOnT_n = (hShotsOnT/(hShotsOnT+aShotsOnT))*20
               aShotsOnT_n = (aShotsOnT/(hShotsOnT+aShotsOnT))*20
        
            # Stats bar diagram
            shooting_stats_title = [51, 51-(1*7), 51-(2*7), 51-(3*7), 51-(4*7), 51-(5*7)]
            shooting_stats_home = [hgoal_count, hTotalShots, hShotsOnT, total_bigC_home, bigC_miss_home, home_average_shot_distance]
            shooting_stats_away = [agoal_count, aTotalShots, aShotsOnT, total_bigC_away, bigC_miss_away, away_average_shot_distance]
        
            # normalizing the stats
            shooting_stats_normalized_home = [hgoal, (hTotalShots/(hTotalShots+aTotalShots))*20, hShotsOnT_n,
                                              total_bigC_home_n, bigC_miss_home_n, 
                                              (home_average_shot_distance/(home_average_shot_distance+away_average_shot_distance))*20]
            shooting_stats_normalized_away = [agoal, (aTotalShots/(hTotalShots+aTotalShots))*20, aShotsOnT_n,
                                              total_bigC_away_n, bigC_miss_away_n,
                                              (away_average_shot_distance/(home_average_shot_distance+away_average_shot_distance))*20]
        
            # definig the start point
            start_x = 42.5
            start_x_for_away = [x + 42.5 for x in shooting_stats_normalized_home]
            ax.barh(shooting_stats_title, shooting_stats_normalized_home, height=5, color=col1, left=start_x)
            ax.barh(shooting_stats_title, shooting_stats_normalized_away, height=5, left=start_x_for_away, color=col2)
            # Turn off axis-related elements
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
            ax.set_xticks([])
            ax.set_yticks([])
        
            # plotting the texts
            ax.text(52.5, 51, "Goals", color=bg_color, fontsize=18, ha='center', va='center', fontweight='bold')
            ax.text(52.5, 51-(1*7), "Shots", color=bg_color, fontsize=18, ha='center', va='center', fontweight='bold')
            ax.text(52.5, 51-(2*7), "On Target", color=bg_color, fontsize=18, ha='center', va='center', fontweight='bold')
            ax.text(52.5, 51-(3*7), "BigChance", color=bg_color, fontsize=18, ha='center', va='center', fontweight='bold')
            ax.text(52.5, 51-(4*7), "BigC.Miss", color=bg_color, fontsize=18, ha='center', va='center', fontweight='bold')
            ax.text(52.5, 51-(5*7), "Avg.Dist.", color=bg_color, fontsize=18, ha='center', va='center', fontweight='bold')
        
            ax.text(41.5, 51, f"{hgoal_count}", color=line_color, fontsize=18, ha='right', va='center', fontweight='bold')
            ax.text(41.5, 51-(1*7), f"{hTotalShots}", color=line_color, fontsize=18, ha='right', va='center', fontweight='bold')
            ax.text(41.5, 51-(2*7), f"{hShotsOnT}", color=line_color, fontsize=18, ha='right', va='center', fontweight='bold')
            ax.text(41.5, 51-(3*7), f"{total_bigC_home}", color=line_color, fontsize=18, ha='right', va='center', fontweight='bold')
            ax.text(41.5, 51-(4*7), f"{bigC_miss_home}", color=line_color, fontsize=18, ha='right', va='center', fontweight='bold')
            ax.text(41.5, 51-(5*7), f"{home_average_shot_distance}", color=line_color, fontsize=18, ha='right', va='center', fontweight='bold')
        
            ax.text(63.5, 51, f"{agoal_count}", color=line_color, fontsize=18, ha='left', va='center', fontweight='bold')
            ax.text(63.5, 51-(1*7), f"{aTotalShots}", color=line_color, fontsize=18, ha='left', va='center', fontweight='bold')
            ax.text(63.5, 51-(2*7), f"{aShotsOnT}", color=line_color, fontsize=18, ha='left', va='center', fontweight='bold')
            ax.text(63.5, 51-(3*7), f"{total_bigC_away}", color=line_color, fontsize=18, ha='left', va='center', fontweight='bold')
            ax.text(63.5, 51-(4*7), f"{bigC_miss_away}", color=line_color, fontsize=18, ha='left', va='center', fontweight='bold')
            ax.text(63.5, 51-(5*7), f"{away_average_shot_distance}", color=line_color, fontsize=18, ha='left', va='center', fontweight='bold')
        
            # Heading and other texts
            ax.text(0, 70, f"{hteamName}\n<---shots", color=col1, size=25, ha='left', fontweight='bold')
            ax.text(105, 70, f"{ateamName}\nshots--->", color=col2, size=25, ha='right', fontweight='bold')
            
            return
        
        
        # Goal Post Viz
        
        def plot_goalPost(ax):
            hShotsdf = Shotsdf[Shotsdf['teamName']==hteamName]
            aShotsdf = Shotsdf[Shotsdf['teamName']==ateamName]
            # converting the datapoints according to the pitch dimension, because the goalposts are being plotted inside the pitch using pitch's dimension
            hShotsdf['goalMouthZ_custom'] = hShotsdf['value_Goal mouth z coordinate']*0.75
            aShotsdf['goalMouthZ_custom'] = (aShotsdf['value_Goal mouth z coordinate']*0.75) + 38
        
            # hShotsdf['goalMouthY_custom'] = ((44 - hShotsdf['value_Goal mouth y coordinate'])*12.295) + 7.5
            # aShotsdf['goalMouthY_custom'] = ((44 - aShotsdf['value_Goal mouth y coordinate'])*12.295) + 7.5
        
            hShotsdf['goalMouthY_custom'] = ((55.5 - hShotsdf['value_Goal mouth y coordinate'])*8.5) + 7.5
            aShotsdf['goalMouthY_custom'] = ((55.5 - aShotsdf['value_Goal mouth y coordinate'])*8.5) + 7.5
        
            # plotting an invisible pitch using the pitch color and line color same color, because the goalposts are being plotted inside the pitch using pitch's dimension
            pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=bg_color, linewidth=2)
            pitch.draw(ax=ax)
            ax.set_ylim(-0.5,68.5)
            ax.set_xlim(-0.5,105.5)
            # ax.set_ylim(-200,200)
            # ax.set_xlim(-200,150)
        
            # away goalpost bars
            ax.plot([7.5, 7.5], [0, 30], color=line_color, linewidth=5)
            ax.plot([7.5, 97.5], [30, 30], color=line_color, linewidth=5)
            ax.plot([97.5, 97.5], [30, 0], color=line_color, linewidth=5)
            ax.plot([0, 105], [0, 0], color=line_color, linewidth=3)
            # plotting the away net
            y_values = np.arange(0, 6) * 6
            for y in y_values:
                ax.plot([7.5, 97.5], [y, y], color=line_color, linewidth=2, alpha=0.2)
            x_values = (np.arange(0, 11) * 9) + 7.5
            for x in x_values:
                ax.plot([x, x], [0, 30], color=line_color, linewidth=2, alpha=0.2)
            # home goalpost bars
            ax.plot([7.5, 7.5], [38, 68], color=line_color, linewidth=5)
            ax.plot([7.5, 97.5], [68, 68], color=line_color, linewidth=5)
            ax.plot([97.5, 97.5], [68, 38], color=line_color, linewidth=5)
            ax.plot([0, 105], [38, 38], color=line_color, linewidth=3)
            # plotting the home net
            y_values = (np.arange(0, 6) * 6) + 38
            for y in y_values:
                ax.plot([7.5, 97.5], [y, y], color=line_color, linewidth=2, alpha=0.2)
            x_values = (np.arange(0, 11) * 9) + 7.5
            for x in x_values:
                ax.plot([x, x], [38, 68], color=line_color, linewidth=2, alpha=0.2)
        
            # filtering different types of shots without BigChance
            hSavedf = hShotsdf[(hShotsdf['type']=='SavedShot') & (hShotsdf['type_value_Blocked']!=82) & (hShotsdf['type_value_Big Chance']!=214)]
            hGoaldf = hShotsdf[(hShotsdf['type']=='Goal') & (hShotsdf['type_value_Own goal']!=28) & (hShotsdf['type_value_Big Chance']!=214)]
            hPostdf = hShotsdf[(hShotsdf['type']=='ShotOnPost') & (hShotsdf['type_value_Big Chance']!=214)]
            aSavedf = aShotsdf[(aShotsdf['type']=='SavedShot') & (aShotsdf['type_value_Blocked']!=82) & (aShotsdf['type_value_Big Chance']!=214)]
            aGoaldf = aShotsdf[(aShotsdf['type']=='Goal') & (aShotsdf['type_value_Own goal']!=28) & (aShotsdf['type_value_Big Chance']!=214)]
            aPostdf = aShotsdf[(aShotsdf['type']=='ShotOnPost') & (aShotsdf['type_value_Big Chance']!=214)]
            # filtering different types of shots with BigChance
            hSavedf_bc = hShotsdf[(hShotsdf['type']=='SavedShot') & (hShotsdf['type_value_Blocked']!=82) & (hShotsdf['type_value_Big Chance']==214)]
            hGoaldf_bc = hShotsdf[(hShotsdf['type']=='Goal') & (hShotsdf['type_value_Own goal']!=28) & (hShotsdf['type_value_Big Chance']==214)]
            hPostdf_bc = hShotsdf[(hShotsdf['type']=='ShotOnPost') & (hShotsdf['type_value_Big Chance']==214)]
            aSavedf_bc = aShotsdf[(aShotsdf['type']=='SavedShot') & (aShotsdf['type_value_Blocked']!=82) & (aShotsdf['type_value_Big Chance']==214)]
            aGoaldf_bc = aShotsdf[(aShotsdf['type']=='Goal') & (aShotsdf['type_value_Own goal']!=28) & (aShotsdf['type_value_Big Chance']==214)]
            aPostdf_bc = aShotsdf[(aShotsdf['type']=='ShotOnPost') & (aShotsdf['type_value_Big Chance']==214)]
        
            # scattering those shots without BigChance
            sc1 = pitch.scatter(hSavedf.goalMouthY_custom, hSavedf.goalMouthZ_custom, marker='o', c=bg_color, zorder=3, edgecolor=col2, hatch='/////', s=350, ax=ax)
            sc2 = pitch.scatter(hGoaldf.goalMouthY_custom, hGoaldf.goalMouthZ_custom, marker='football', c=bg_color, zorder=3, edgecolors='green', s=350, ax=ax)
            sc3 = pitch.scatter(hPostdf.goalMouthY_custom, hPostdf.goalMouthZ_custom, marker='o', c=bg_color, zorder=3, edgecolors='orange', hatch='/////', s=350, ax=ax)
            sc4 = pitch.scatter(aSavedf.goalMouthY_custom, aSavedf.goalMouthZ_custom, marker='o', c=bg_color, zorder=3, edgecolor=col1, hatch='/////', s=350, ax=ax)
            sc5 = pitch.scatter(aGoaldf.goalMouthY_custom, aGoaldf.goalMouthZ_custom, marker='football', c=bg_color, zorder=3, edgecolors='green', s=350, ax=ax)
            sc6 = pitch.scatter(aPostdf.goalMouthY_custom, aPostdf.goalMouthZ_custom, marker='o', c=bg_color, zorder=3, edgecolors='orange', hatch='/////', s=350, ax=ax)
            # scattering those shots with BigChance
            sc1_bc = pitch.scatter(hSavedf_bc.goalMouthY_custom, hSavedf_bc.goalMouthZ_custom, marker='o', c=bg_color, zorder=3, edgecolor=col2, hatch='/////', s=1000, ax=ax)
            sc2_bc = pitch.scatter(hGoaldf_bc.goalMouthY_custom, hGoaldf_bc.goalMouthZ_custom, marker='football', c=bg_color, zorder=3, edgecolors='green', s=1000, ax=ax)
            sc3_bc = pitch.scatter(hPostdf_bc.goalMouthY_custom, hPostdf_bc.goalMouthZ_custom, marker='o', c=bg_color, zorder=3, edgecolors='orange', hatch='/////', s=1000, ax=ax)
            sc4_bc = pitch.scatter(aSavedf_bc.goalMouthY_custom, aSavedf_bc.goalMouthZ_custom, marker='o', c=bg_color, zorder=3, edgecolor=col1, hatch='/////', s=1000, ax=ax)
            sc5_bc = pitch.scatter(aGoaldf_bc.goalMouthY_custom, aGoaldf_bc.goalMouthZ_custom, marker='football', c=bg_color, zorder=3, edgecolors='green', s=1000, ax=ax)
            sc6_bc = pitch.scatter(aPostdf_bc.goalMouthY_custom, aPostdf_bc.goalMouthZ_custom, marker='o', c=bg_color, zorder=3, edgecolors='orange', hatch='/////', s=1000, ax=ax)
        
            # Headlines and other texts
            ax.text(52.5, 70, f"{hteamName} GK saves", color=col1, fontsize=30, ha='center', fontweight='bold')
            ax.text(52.5, -2, f"{ateamName} GK saves", color=col2, fontsize=30, ha='center', va='top', fontweight='bold')
        
            ax.text(100, 68, f"Saves = {len(aSavedf)+len(aSavedf_bc)}",
                            color=col1, fontsize=16, va='top', ha='left')
            ax.text(100, 2, f"Saves = {len(hSavedf)+len(hSavedf_bc)}",
                            color=col2, fontsize=16, va='bottom', ha='left')
        
            return
        
        
        # Match Momentum
        
        u_df = df.copy()
        u_df = u_df[(u_df['type_value_Corner taken']!=6)]
        u_df = u_df[['x', 'minute', 'period', 'type', 'teamName']]
        u_df = u_df[~u_df['type'].isin(['Start', 'OffsidePass', 'OffsideProvoked', 'Card', 'CornerAwarded', 'End', 
                                        'OffsideGiven', 'SubstitutionOff', 'SubstitutionOn', 'FormationChange', 'FormationSet'])].reset_index(drop=True)
        u_df.loc[u_df['teamName'] == ateamName, 'x'] = 105 - u_df.loc[u_df['teamName'] == ateamName, 'x']
        
        Momentumdf = u_df.groupby('minute')['x'].mean()
        Momentumdf = Momentumdf.reset_index()
        Momentumdf.columns = ['minute', 'average_x']
        Momentumdf['average_x'] = Momentumdf['average_x'] - 52.5
        
        u_df_1 = u_df[u_df['period']=='FirstHalf']
        u_df_2 = u_df[u_df['period']=='SecondHalf']
        
        Momentumdf1 = u_df_1.groupby('minute')['x'].mean()
        Momentumdf1 = Momentumdf1.reset_index()
        Momentumdf1.columns = ['minute', 'average_x']
        Momentumdf1['average_x'] = Momentumdf1['average_x'] - 52.5
        
        Momentumdf2 = u_df_2.groupby('minute')['x'].mean()
        Momentumdf2 = Momentumdf2.reset_index()
        Momentumdf2.columns = ['minute', 'average_x']
        Momentumdf2['average_x'] = Momentumdf2['average_x'] - 52.5
        
        def plot_Momentum(ax):
            # Set colors based on positive or negative values
            colors1 = [col1 if x > 0 else col2 for x in Momentumdf1['average_x']]
            colors2 = [col1 if x > 0 else col2 for x in Momentumdf2['average_x']]
        
            homedf = df[df['teamName']==hteamName]
            awaydf = df[df['teamName']==ateamName]
            hxT = homedf['xT'].sum().round(2)
            axT = awaydf['xT'].sum().round(2)
            # making a list of munutes when goals are scored
            hgoal_list = homedf[(homedf['type'] == 'Goal') & (homedf['type_value_Own goal']!=28)]['minute'].tolist()
            agoal_list = awaydf[(awaydf['type'] == 'Goal') & (awaydf['type_value_Own goal']!=28)]['minute'].tolist()
            hog_list = homedf[(homedf['type'] == 'Goal') & (homedf['type_value_Own goal']==28)]['minute'].tolist()
            aog_list = awaydf[(awaydf['type'] == 'Goal') & (awaydf['type_value_Own goal']==28)]['minute'].tolist()
            # hred_list = homedf[homedf['qualifiers'].str.contains('Red|SecondYellow')]['minute'].tolist()
            # ared_list = awaydf[awaydf['qualifiers'].str.contains('Red|SecondYellow')]['minute'].tolist()
        
            # plotting scatters when goals are scored
            highest_x = Momentumdf['average_x'].max()
            lowest_x = Momentumdf['average_x'].min()
            highest_minute = Momentumdf['minute'].max()
            hscatter_y = [highest_x]*len(hgoal_list)
            ascatter_y = [lowest_x]*len(agoal_list)
            hogscatter_y = [highest_x]*len(aog_list)
            aogscatter_y = [lowest_x]*len(hog_list)
            # hred_y = [highest_x]*len(hred_list)
            # ared_y = [lowest_x]*len(ared_list)
            extra_time = Momentumdf1['minute'].max() - 45
            
        
            ax.text((45/2), lowest_x, 'First Half', color='gray', fontsize=20, alpha=0.25, va='center', ha='center')
            ax.text((45+(45/2)), lowest_x, 'Second Half', color='gray', fontsize=20, alpha=0.25, va='center', ha='center')
        
            ax.scatter(hgoal_list, hscatter_y, s=250, c='None', edgecolor='green', hatch='////', marker='o')
            ax.scatter(agoal_list, ascatter_y, s=250, c='None', edgecolor='green', hatch='////', marker='o')
            ax.scatter(hog_list, aogscatter_y, s=250, c='None', edgecolor='orange', hatch='////', marker='o')
            ax.scatter(aog_list, hogscatter_y, s=250, c='None', edgecolor='orange', hatch='////', marker='o')
            # ax.scatter(hred_list, hred_y, s=250, c='None', edgecolor='red', hatch='////', marker='s')
            # ax.scatter(ared_list, ared_y, s=250, c='None', edgecolor='red', hatch='////', marker='s')
        
            # Creating the bar plot
            ax.bar(Momentumdf1['minute'], Momentumdf1['average_x'], width=1, color=colors1)
            ax.bar(Momentumdf2['minute']+extra_time, Momentumdf2['average_x'], width=1, color=colors2)
            ax.set_xticks(range(0, len(Momentumdf['minute']), 5))
            ax.axvline(45, color='gray', linewidth=2, linestyle='dotted')
            # ax.axvline(90, color='gray', linewidth=2, linestyle='dotted')
            ax.set_facecolor(bg_color)
            # Hide spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            # # Hide ticks
            ax.tick_params(axis='both', which='both', length=0)
            ax.tick_params(axis='x', colors=line_color)
            ax.tick_params(axis='y', colors=bg_color)
            # Add labels and title
            ax.set_xlabel('Minute', color=line_color, fontsize=20)
            # ax.set_ylabel('Avg. xT per minute', color=line_color, fontsize=20)
            ax.axhline(y=0, color=line_color, alpha=1, linewidth=2)
        
            ax.text(highest_minute+1,highest_x, f"{hteamName}\nxT: {hxT}", color=col1, fontsize=20, va='bottom', ha='left')
            ax.text(highest_minute+1,lowest_x,  f"{ateamName}\nxT: {axT}", color=col2, fontsize=20, va='top', ha='left')
        
            ax.set_title('Match Momentum', color=line_color, fontsize=30, fontweight='bold')
            
            return
        
        
        # Stats
        
        #Possession%
        hpossdf = df[(df['teamName']==hteamName) & (df['type']=='Pass')]
        apossdf = df[(df['teamName']==ateamName) & (df['type']=='Pass')]
        hposs = round((len(hpossdf)/(len(hpossdf)+len(apossdf)))*100,2)
        aposs = round((len(apossdf)/(len(hpossdf)+len(apossdf)))*100,2)
        
        #Field Tilt%
        hftdf = df[(df['teamName']==hteamName) & (df['isTouch']==1) & (df['endX']>=70)]
        aftdf = df[(df['teamName']==ateamName) & (df['isTouch']==1) & (df['endX']>=70)]
        hft = round((len(hftdf)/(len(hftdf)+len(aftdf)))*100,2)
        aft = round((len(aftdf)/(len(hftdf)+len(aftdf)))*100,2)
        
        #Total Passes
        htotalPass = len(df[(df['teamName']==hteamName) & (df['type']=='Pass')])
        atotalPass = len(df[(df['teamName']==ateamName) & (df['type']=='Pass')])
        
        #Accurate Pass
        hAccPass = len(df[(df['teamName']==hteamName) & (df['type']=='Pass') & (df['outcomeType']=='Successful')])
        aAccPass = len(df[(df['teamName']==ateamName) & (df['type']=='Pass') & (df['outcomeType']=='Successful')])
        
        #LongBall
        hLongB = len(df[(df['teamName']==hteamName) & (df['type']=='Pass') & (df['type_value_Long ball']==1) & (df['type_value_Corner taken']!=6) & (df['type_value_Cross']!=1)])
        aLongB = len(df[(df['teamName']==ateamName) & (df['type']=='Pass') & (df['type_value_Long ball']==1) & (df['type_value_Corner taken']!=6) & (df['type_value_Cross']!=1)])
        #Accurate LongBall
        hAccLongB = len(df[(df['teamName']==hteamName) & (df['type']=='Pass') & (df['type_value_Long ball']==1) & (df['outcomeType']=='Successful') & (df['type_value_Corner taken']!=6) & (df['type_value_Cross']!=1)])
        aAccLongB = len(df[(df['teamName']==ateamName) & (df['type']=='Pass') & (df['type_value_Long ball']==1) & (df['outcomeType']=='Successful') & (df['type_value_Corner taken']!=6) & (df['type_value_Cross']!=1)])
        
        # Defensive Stats
        #Tackles
        htkl = len(df[(df['teamName']==hteamName) & (df['type']=='Tackle')])
        atkl = len(df[(df['teamName']==ateamName) & (df['type']=='Tackle')])
        
        #Tackles Won
        htklw = len(df[(df['teamName']==hteamName) & (df['type']=='Tackle') & (df['outcomeType']=='Successful')])
        atklw = len(df[(df['teamName']==ateamName) & (df['type']=='Tackle') & (df['outcomeType']=='Successful')])
        
        #Interceptions
        hintc= len(df[(df['teamName']==hteamName) & (df['type']=='Interception')])
        aintc= len(df[(df['teamName']==ateamName) & (df['type']=='Interception')])
        
        #Clearances
        hclr= len(df[(df['teamName']==hteamName) & (df['type']=='Clearance')])
        aclr= len(df[(df['teamName']==ateamName) & (df['type']=='Clearance')])
        
        #Aerials
        harl= len(df[(df['teamName']==hteamName) & (df['type']=='Aerial')])
        aarl= len(df[(df['teamName']==ateamName) & (df['type']=='Aerial')])
        
        #Aerials Wins
        harlw= len(df[(df['teamName']==hteamName) & (df['type']=='Aerial') & (df['outcomeType']=='Successful')])
        aarlw= len(df[(df['teamName']==ateamName) & (df['type']=='Aerial') & (df['outcomeType']=='Successful')])
        
        # PPDA
        home_def_acts = df[(df['teamName']==hteamName) & (df['type'].str.contains('Interception|Foul|Challenge|BlockedPass|Tackle')) & (df['x']>35)]
        away_def_acts = df[(df['teamName']==ateamName) & (df['type'].str.contains('Interception|Foul|Challenge|BlockedPass|Tackle')) & (df['x']>35)]
        home_pass = df[(df['teamName']==hteamName) & (df['type']=='Pass') & (df['outcomeType']=='Successful') & (df['x']<70)]
        away_pass = df[(df['teamName']==ateamName) & (df['type']=='Pass') & (df['outcomeType']=='Successful') & (df['x']<70)]
        home_ppda = round((len(away_pass)/len(home_def_acts)), 2)
        away_ppda = round((len(home_pass)/len(away_def_acts)), 2)
        
        # Average Passes per Sequence
        pass_df_home = df[(df['type'] == 'Pass') & (df['teamName']==hteamName)]
        pass_counts_home = pass_df_home.groupby('possession_id').size()
        PPS_home = pass_counts_home.mean().round()
        pass_df_away = df[(df['type'] == 'Pass') & (df['teamName']==ateamName)]
        pass_counts_away = pass_df_away.groupby('possession_id').size()
        PPS_away = pass_counts_away.mean().round()
        
        # Number of Sequence with 10+ Passes
        possessions_with_10_or_more_passes = pass_counts_home[pass_counts_home >= 10]
        pass_seq_10_more_home = possessions_with_10_or_more_passes.count()
        possessions_with_10_or_more_passes = pass_counts_away[pass_counts_away >= 10]
        pass_seq_10_more_away = possessions_with_10_or_more_passes.count()
        
        path_eff1 = [path_effects.Stroke(linewidth=1.5, foreground=line_color), path_effects.Normal()]
        path_eff = [path_effects.Stroke(linewidth=3, foreground=bg_color), path_effects.Normal()]
        
        def plotting_match_stats(ax):
            pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=bg_color, linewidth=2)
            pitch.draw(ax=ax)
            ax.set_xlim(-0.5, 105.5)
            ax.set_ylim(-5, 68.5)
        
            # plotting the headline box
            head_y = [62,68,68,62]
            head_x = [0,0,105,105]
            ax.fill(head_x, head_y, 'orange')
            ax.text(52.5,64.5, "Match Stats", ha='center', va='center', color=line_color, fontsize=25, fontweight='bold', path_effects=path_eff)
        
            # Stats bar diagram
            stats_title = [58, 58-(1*6), 58-(2*6), 58-(3*6), 58-(4*6), 58-(5*6), 58-(6*6), 58-(7*6), 58-(8*6), 58-(9*6), 58-(10*6)] # y co-ordinate values of the bars
            stats_home = [hposs, hft, htotalPass, hLongB, htkl, hintc, hclr, harl, home_ppda, PPS_home, pass_seq_10_more_home]
            stats_away = [aposs, aft, atotalPass, aLongB, atkl, aintc, aclr, aarl, away_ppda, PPS_away, pass_seq_10_more_away]
        
            stats_normalized_home = [-(hposs/(hposs+aposs))*50, -(hft/(hft+aft))*50, -(htotalPass/(htotalPass+atotalPass))*50,
                                     -(hLongB/(hLongB+aLongB))*50, -(htkl/(htkl+atkl))*50,       # put a (-) sign before each value so that the
                                     -(hintc/(hintc+aintc))*50, -(hclr/(hclr+aclr))*50, -(harl/(harl+aarl))*50, -(home_ppda/(home_ppda+away_ppda))*50,
                                     -(PPS_home/(PPS_home+PPS_away))*50, -(pass_seq_10_more_home/(pass_seq_10_more_home+pass_seq_10_more_away))*50]          # home stats bar shows in the opposite of away
            stats_normalized_away = [(aposs/(hposs+aposs))*50, (aft/(hft+aft))*50, (atotalPass/(htotalPass+atotalPass))*50,
                                     (aLongB/(hLongB+aLongB))*50, (atkl/(htkl+atkl))*50,
                                     (aintc/(hintc+aintc))*50, (aclr/(hclr+aclr))*50, (aarl/(harl+aarl))*50, (away_ppda/(home_ppda+away_ppda))*50,
                                     (PPS_away/(PPS_home+PPS_away))*50, (pass_seq_10_more_away/(pass_seq_10_more_home+pass_seq_10_more_away))*50]
        
            start_x = 52.5
            ax.barh(stats_title, stats_normalized_home, height=4, color=col1, left=start_x)
            ax.barh(stats_title, stats_normalized_away, height=4, left=start_x, color=col2)
            # Turn off axis-related elements
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
            ax.set_xticks([])
            ax.set_yticks([])
        
            # Plotting the texts
            ax.text(52.5, 58, "Possession", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
            ax.text(52.5, 58-(1*6), "Field Tilt", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
            ax.text(52.5, 58-(2*6), "Passes (Acc.)", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
            ax.text(52.5, 58-(3*6), "LongBalls (Acc.)", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
            ax.text(52.5, 58-(4*6), "Tackles (Wins)", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
            ax.text(52.5, 58-(5*6), "Interceptions", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
            ax.text(52.5, 58-(6*6), "Clearence", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
            ax.text(52.5, 58-(7*6), "Aerials (Wins)", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
            ax.text(52.5, 58-(8*6), "PPDA", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
            ax.text(52.5, 58-(9*6), "Pass/Sequence", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
            ax.text(52.5, 58-(10*6), "10+Pass Seq.", color=bg_color, fontsize=17, ha='center', va='center', fontweight='bold', path_effects=path_eff1)
        
            ax.text(0, 58, f"{round(hposs)}%", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
            ax.text(0, 58-(1*6), f"{round(hft)}%", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
            ax.text(0, 58-(2*6), f"{htotalPass}({hAccPass})", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
            ax.text(0, 58-(3*6), f"{hLongB}({hAccLongB})", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
            ax.text(0, 58-(4*6), f"{htkl}({htklw})", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
            ax.text(0, 58-(5*6), f"{hintc}", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
            ax.text(0, 58-(6*6), f"{hclr}", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
            ax.text(0, 58-(7*6), f"{harl}({harlw})", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
            ax.text(0, 58-(8*6), f"{home_ppda}", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
            ax.text(0, 58-(9*6), f"{int(PPS_home)}", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
            ax.text(0, 58-(10*6), f"{pass_seq_10_more_home}", color=line_color, fontsize=20, ha='right', va='center', fontweight='bold')
        
            ax.text(105, 58, f"{round(aposs)}%", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
            ax.text(105, 58-(1*6), f"{round(aft)}%", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
            ax.text(105, 58-(2*6), f"{atotalPass}({aAccPass})", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
            ax.text(105, 58-(3*6), f"{aLongB}({aAccLongB})", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
            ax.text(105, 58-(4*6), f"{atkl}({atklw})", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
            ax.text(105, 58-(5*6), f"{aintc}", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
            ax.text(105, 58-(6*6), f"{aclr}", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
            ax.text(105, 58-(7*6), f"{aarl}({aarlw})", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
            ax.text(105, 58-(8*6), f"{away_ppda}", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
            ax.text(105, 58-(9*6), f"{int(PPS_away)}", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
            ax.text(105, 58-(10*6), f"{pass_seq_10_more_away}", color=line_color, fontsize=20, ha='left', va='center', fontweight='bold')
            
            
        # Final Third Entry
        
        def Final_third_entry(ax, team_name, col):
            dfpass = df[(df['teamName']==team_name) & (df['type']=='Pass') & (df['x']<70) & (df['endX']>=70) & (df['outcomeType']=='Successful') &
                        (df['type_value_Free kick taken']!=5)]
            dfcarry = df[(df['teamName']==team_name) & (df['type']=='Carry') & (df['x']<70) & (df['endX']>=70)]
            pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color, linewidth=2,
                                  corner_arcs=True)
            pitch.draw(ax=ax)
            ax.set_xlim(-0.5, 105.5)
            # ax.set_ylim(-0.5, 68.5)
        
            if team_name == ateamName:
                ax.invert_xaxis()
                ax.invert_yaxis()
        
            pass_count = len(dfpass) + len(dfcarry)
        
            # calculating the counts
            left_entry = len(dfpass[dfpass['y']>=45.33]) + len(dfcarry[dfcarry['y']>=45.33])
            mid_entry = len(dfpass[(dfpass['y']>=22.67) & (dfpass['y']<45.33)]) + len(dfcarry[(dfcarry['y']>=22.67) & (dfcarry['y']<45.33)])
            right_entry = len(dfpass[(dfpass['y']>=0) & (dfpass['y']<22.67)]) + len(dfcarry[(dfcarry['y']>=0) & (dfcarry['y']<22.67)])
            left_percentage = round((left_entry/pass_count)*100)
            mid_percentage = round((mid_entry/pass_count)*100)
            right_percentage = round((right_entry/pass_count)*100)
        
            ax.hlines(22.67, xmin=0, xmax=70, colors=line_color, linestyle='dashed', alpha=0.35)
            ax.hlines(45.33, xmin=0, xmax=70, colors=line_color, linestyle='dashed', alpha=0.35)
            ax.vlines(70, ymin=-2, ymax=70, colors=line_color, linestyle='dashed', alpha=0.55)
        
            # showing the texts in the pitch
            bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="None", facecolor=bg_color, alpha=0.75)
            if col == col1:
                ax.text(8, 11.335, f'{right_entry}\n({right_percentage}%)', color=col1, fontsize=24, va='center', ha='center', bbox=bbox_props)
                ax.text(8, 34, f'{mid_entry}\n({mid_percentage}%)', color=col1, fontsize=24, va='center', ha='center', bbox=bbox_props)
                ax.text(8, 56.675, f'{left_entry}\n({left_percentage}%)', color=col1, fontsize=24, va='center', ha='center', bbox=bbox_props)
            else:
                ax.text(8, 11.335, f'{right_entry}\n({right_percentage}%)', color=col2, fontsize=24, va='center', ha='center', bbox=bbox_props)
                ax.text(8, 34, f'{mid_entry}\n({mid_percentage}%)', color=col2, fontsize=24, va='center', ha='center', bbox=bbox_props)
                ax.text(8, 56.675, f'{left_entry}\n({left_percentage}%)', color=col2, fontsize=24, va='center', ha='center', bbox=bbox_props)
        
            # plotting the passes
            pro_pass = pitch.lines(dfpass.x, dfpass.y, dfpass.endX, dfpass.endY, lw=3.5, comet=True, color=col, ax=ax, alpha=0.5)
            # plotting some scatters at the end of each pass
            pro_pass_end = pitch.scatter(dfpass.endX, dfpass.endY, s=35, edgecolor=col, linewidth=1, color=bg_color, zorder=2, ax=ax)
            # plotting carries
            for index, row in dfcarry.iterrows():
                arrow = patches.FancyArrowPatch((row['x'], row['y']), (row['endX'], row['endY']), arrowstyle='->', color=col, zorder=4, mutation_scale=20, 
                                                alpha=1, linewidth=2, linestyle='--')
                ax.add_patch(arrow)
        
            counttext = f"{pass_count} Final Third Entries"
        
            # Heading and other texts
            if col == col1:
                ax.set_title(f"{hteamName}\n{counttext}", color=line_color, fontsize=25, fontweight='bold', path_effects=path_eff)
                ax.text(87.5, 70, '<-------------- Final third -------------->', color=line_color, ha='center', va='center')
                pitch.lines(53, -2, 73, -2, lw=3, transparent=True, comet=True, color=col, ax=ax, alpha=0.5)
                ax.scatter(73,-2, s=35, edgecolor=col, linewidth=1, color=bg_color, zorder=2)
                arrow = patches.FancyArrowPatch((83, -2), (103, -2), arrowstyle='->', color=col, zorder=4, mutation_scale=20, 
                                                alpha=1, linewidth=2, linestyle='--')
                ax.add_patch(arrow)
                ax.text(63, -5, f'Entry by Pass: {len(dfpass)}', fontsize=15, color=line_color, ha='center', va='center')
                ax.text(93, -5, f'Entry by Carry: {len(dfcarry)}', fontsize=15, color=line_color, ha='center', va='center')
                
            else:
                ax.set_title(f"{ateamName}\n{counttext}", color=line_color, fontsize=25, fontweight='bold', path_effects=path_eff)
                ax.text(87.5, -2, '<-------------- Final third -------------->', color=line_color, ha='center', va='center')
                pitch.lines(53, 70, 73, 70, lw=3, transparent=True, comet=True, color=col, ax=ax, alpha=0.5)
                ax.scatter(73,70, s=35, edgecolor=col, linewidth=1, color=bg_color, zorder=2)
                arrow = patches.FancyArrowPatch((83, 70), (103, 70), arrowstyle='->', color=col, zorder=4, mutation_scale=20, 
                                                alpha=1, linewidth=2, linestyle='--')
                ax.add_patch(arrow)
                ax.text(63, 73, f'Entry by Pass: {len(dfpass)}', fontsize=15, color=line_color, ha='center', va='center')
                ax.text(93, 73, f'Entry by Carry: {len(dfcarry)}', fontsize=15, color=line_color, ha='center', va='center')
        
            return 
        
        
        # Zone14 & Half-Spaces
        
        def zone14hs(ax, team_name, col):
            dfhp = df[(df['teamName']==team_name) & (df['type']=='Pass') & (df['outcomeType']=='Successful') & 
                      (df['type_value_Corner taken']!=6) & (df['type_value_Free kick taken']!=5)]
            
            pitch = Pitch(pitch_type='uefa', pitch_color=bg_color, line_color=line_color,  linewidth=2,
                                  corner_arcs=True)
            pitch.draw(ax=ax)
            ax.set_xlim(-0.5, 105.5)
            ax.set_facecolor(bg_color)
            if team_name == ateamName:
              ax.invert_xaxis()
              ax.invert_yaxis()
        
            # setting the count varibale
            z14 = 0
            hs = 0
            lhs = 0
            rhs = 0
        
            path_eff = [path_effects.Stroke(linewidth=3, foreground=bg_color), path_effects.Normal()]
            # iterating ecah pass and according to the conditions plotting only zone14 and half spaces passes
            for index, row in dfhp.iterrows():
                if row['endX'] >= 70 and row['endX'] <= 88.54 and row['endY'] >= 22.66 and row['endY'] <= 45.32:
                    pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color='orange', comet=True, lw=3, zorder=3, ax=ax, alpha=0.75)
                    ax.scatter(row['endX'], row['endY'], s=35, linewidth=1, color=bg_color, edgecolor='orange', zorder=4)
                    z14 += 1
                if row['endX'] >= 70 and row['endY'] >= 11.33 and row['endY'] <= 22.66:
                    pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=col, comet=True, lw=3, zorder=3, ax=ax, alpha=0.75)
                    ax.scatter(row['endX'], row['endY'], s=35, linewidth=1, color=bg_color, edgecolor=col, zorder=4)
                    hs += 1
                    rhs += 1
                if row['endX'] >= 70 and row['endY'] >= 45.32 and row['endY'] <= 56.95:
                    pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=col, comet=True, lw=3, zorder=3, ax=ax, alpha=0.75)
                    ax.scatter(row['endX'], row['endY'], s=35, linewidth=1, color=bg_color, edgecolor=col, zorder=4)
                    hs += 1
                    lhs += 1
        
            # coloring those zones in the pitch
            y_z14 = [22.66, 22.66, 45.32, 45.32]
            x_z14 = [70, 88.54, 88.54, 70]
            ax.fill(x_z14, y_z14, 'orange', alpha=0.2, label='Zone14')
        
            y_rhs = [11.33, 11.33, 22.66, 22.66]
            x_rhs = [70, 105, 105, 70]
            ax.fill(x_rhs, y_rhs, col, alpha=0.2, label='HalfSpaces')
        
            y_lhs = [45.32, 45.32, 56.95, 56.95]
            x_lhs = [70, 105, 105, 70]
            ax.fill(x_lhs, y_lhs, col, alpha=0.2, label='HalfSpaces')
        
            # showing the counts in an attractive way
            z14name = "Zone14"
            hsname = "HalfSp"
            z14count = f"{z14}"
            hscount = f"{hs}"
            ax.scatter(16.46, 13.85, color=col, s=15000, edgecolor=line_color, linewidth=2, alpha=1, marker='h')
            ax.scatter(16.46, 54.15, color='orange', s=15000, edgecolor=line_color, linewidth=2, alpha=1, marker='h')
            ax.text(16.46, 13.85-4, hsname, fontsize=20, color=line_color, ha='center', va='center', path_effects=path_eff)
            ax.text(16.46, 54.15-4, z14name, fontsize=20, color=line_color, ha='center', va='center', path_effects=path_eff)
            ax.text(16.46, 13.85+2, hscount, fontsize=40, color=line_color, ha='center', va='center', path_effects=path_eff)
            ax.text(16.46, 54.15+2, z14count, fontsize=40, color=line_color, ha='center', va='center', path_effects=path_eff)
        
            # Headings and other texts
            if col == col1:
              ax.set_title(f"{hteamName}\nZone14 & Halfsp. Pass", color=line_color, fontsize=25, fontweight='bold')
            else:
              ax.set_title(f"{ateamName}\nZone14 & Halfsp. Pass", color=line_color, fontsize=25, fontweight='bold')
        
            return 
        
        # Pass End Zone
        
        # setting the custom colormap
        pearl_earring_cmaph = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",  [bg_color, col1], N=20)
        pearl_earring_cmapa = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",  [bg_color, col2], N=20)
        
        path_eff = [path_effects.Stroke(linewidth=3, foreground=bg_color), path_effects.Normal()]
        
        def Pass_end_zone(ax, team_name, cm):
            pez = df[(df['teamName'] == team_name) & (df['type'] == 'Pass') & (df['outcomeType'] == 'Successful')]
            pitch = Pitch(pitch_type='uefa', line_color=line_color, goal_type='box', goal_alpha=.5, corner_arcs=True, line_zorder=2, pitch_color=bg_color, linewidth=2)
            pitch.draw(ax=ax)
            ax.set_xlim(-0.5, 105.5)
            if team_name == ateamName:
              ax.invert_xaxis()
              ax.invert_yaxis()
        
            pearl_earring_cmap = cm
            # binning the data points
            # bin_statistic = pitch.bin_statistic_positional(df.endX, df.endY, statistic='count', positional='full', normalize=True)
            bin_statistic = pitch.bin_statistic(pez.endX, pez.endY, bins=(6, 5), normalize=True)
            pitch.heatmap(bin_statistic, ax=ax, cmap=pearl_earring_cmap, edgecolors=bg_color)
            pitch.scatter(pez.endX, pez.endY, c='gray', alpha=0.5, s=5, ax=ax)
            labels = pitch.label_heatmap(bin_statistic, color=line_color, fontsize=25, ax=ax, ha='center', va='center', str_format='{:.0%}', path_effects=path_eff)
        
            # Headings and other texts
            if team_name == hteamName:
              ax.set_title(f"{hteamName}\nPass Target Zone", color=line_color, fontsize=25, fontweight='bold', path_effects=path_eff)
            else:
              ax.set_title(f"{ateamName}\nPass Target Zone", color=line_color, fontsize=25, fontweight='bold', path_effects=path_eff)
              
              
        # Chances Creating Zones
        
        # setting the custom colormap
        pearl_earring_cmaph = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors", [bg_color, col1], N=20)
        pearl_earring_cmapa = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors", [bg_color, col2], N=20)
        
        path_eff = [path_effects.Stroke(linewidth=3, foreground=bg_color), path_effects.Normal()]
        
        def Chance_creating_zone(ax, team_name, cm, col):
            ccp = df[(df['type_value_Assist']==210) & (df['teamName']==team_name) & (df['type'].str.contains('Pass|BallTouch'))]
            pitch = Pitch(pitch_type='uefa', line_color=line_color, corner_arcs=True, line_zorder=2, pitch_color=bg_color, linewidth=2)
            pitch.draw(ax=ax)
            ax.set_xlim(-0.5, 105.5)
            if team_name == ateamName:
              ax.invert_xaxis()
              ax.invert_yaxis()
        
            cc = 0
            pearl_earring_cmap = cm
            # bin_statistic = pitch.bin_statistic_positional(df.x, df.y, statistic='count', positional='full', normalize=False)
            bin_statistic = pitch.bin_statistic(ccp.x, ccp.y, bins=(6,5), statistic='count', normalize=False)
            pitch.heatmap(bin_statistic, ax=ax, cmap=pearl_earring_cmap, edgecolors='#f8f8f8')
            # pitch.scatter(ccp.x, ccp.y, c='gray', s=5, ax=ax)
            for index, row in ccp.iterrows():
              if row['assist']==1:
                pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=green, comet=True, lw=3, zorder=3, ax=ax)
                ax.scatter(row['endX'], row['endY'], s=35, linewidth=1, color=bg_color, edgecolor=green, zorder=4)
                cc += 1
              else :
                pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=violet, comet=True, lw=3, zorder=3, ax=ax)
                ax.scatter(row['endX'], row['endY'], s=35, linewidth=1, color=bg_color, edgecolor=violet, zorder=4)
                cc += 1
            labels = pitch.label_heatmap(bin_statistic, color=line_color, fontsize=25, ax=ax, ha='center', va='center', str_format='{:.0f}', path_effects=path_eff)
            teamName = team_name
        
            # Headings and other texts
            if col == col1:
              ax.text(105,-3.5, "violet = key pass\ngreen = assist", color=col1, size=15, ha='right', va='center')
              ax.text(52.5,70, f"Total Chances Created = {cc}", color=col, fontsize=15, ha='center', va='center')
              ax.set_title(f"{hteamName}\nChance Creating Zone", color=line_color, fontsize=25, fontweight='bold', path_effects=path_eff)
            else:
              ax.text(105,71.5, "violet = key pass\ngreen = assist", color=col2, size=15, ha='left', va='center')
              ax.text(52.5,-2, f"Total Chances Created = {cc}", color=col, fontsize=15, ha='center', va='center')
              ax.set_title(f"{ateamName}\nChance Creating Zone", color=line_color, fontsize=25, fontweight='bold', path_effects=path_eff)
        
            return 
        
        
        # Box Entry
        
        def box_entry(ax):
            bentry = df[((df['type']=='Pass')|(df['type']=='Carry')) & (df['outcomeType']=='Successful') & (df['endX']>=88.5) &
                        ~((df['x']>=88.5) & (df['y']>=13.6) & (df['y']<=54.6)) & (df['endY']>=13.6) & (df['endY']<=54.4) &
                        (df['type_value_Corner taken']!=6) & (df['type_value_Free kick taken']!=5)]
            hbentry = bentry[bentry['teamName']==hteamName]
            abentry = bentry[bentry['teamName']==ateamName]
        
            hrigt = hbentry[hbentry['y']<68/3]
            hcent = hbentry[(hbentry['y']>=68/3) & (hbentry['y']<=136/3)]
            hleft = hbentry[hbentry['y']>136/3]
        
            arigt = abentry[(abentry['y']<68/3)]
            acent = abentry[(abentry['y']>=68/3) & (abentry['y']<=136/3)]
            aleft = abentry[(abentry['y']>136/3)]
        
            pitch = Pitch(pitch_type='uefa', line_color=line_color, corner_arcs=True, line_zorder=2, pitch_color=bg_color, linewidth=2)
            pitch.draw(ax=ax)
            ax.set_xlim(-0.5, 105.5)
            ax.set_ylim(-0.5, 68.5)
        
            for index, row in bentry.iterrows():
                if row['teamName'] == ateamName:
                    color = col2
                    x, y, endX, endY = row['x'], row['y'], row['endX'], row['endY']
                elif row['teamName'] == hteamName:
                    color = col1
                    x, y, endX, endY = 105 - row['x'], 68 - row['y'], 105 - row['endX'], 68 - row['endY']
                else:
                    continue  # Skip rows that don't match either team name
        
                if row['type'] == 'Pass':
                    pitch.lines(x, y, endX, endY, lw=3.5, comet=True, color=color, ax=ax, alpha=0.5)
                    pitch.scatter(endX, endY, s=35, edgecolor=color, linewidth=1, color=bg_color, zorder=2, ax=ax)
                elif row['type'] == 'Carry':
                    arrow = patches.FancyArrowPatch((x, y), (endX, endY), arrowstyle='->', color=color, zorder=4, mutation_scale=20, 
                                                    alpha=1, linewidth=2, linestyle='--')
                    ax.add_patch(arrow)
        
            
            ax.text(0, 69, f'{hteamName}\nBox Entries: {len(hbentry)}', color=col1, fontsize=25, fontweight='bold', ha='left', va='bottom')
            ax.text(105, 69, f'{ateamName}\nBox Entries: {len(abentry)}', color=col2, fontsize=25, fontweight='bold', ha='right', va='bottom')
        
            ax.scatter(46, 6, s=2000, marker='s', color=col1, zorder=3)
            ax.scatter(46, 34, s=2000, marker='s', color=col1, zorder=3)
            ax.scatter(46, 62, s=2000, marker='s', color=col1, zorder=3)
            ax.text(46, 6, f'{len(hleft)}', fontsize=30, fontweight='bold', color=bg_color, ha='center', va='center')
            ax.text(46, 34, f'{len(hcent)}', fontsize=30, fontweight='bold', color=bg_color, ha='center', va='center')
            ax.text(46, 62, f'{len(hrigt)}', fontsize=30, fontweight='bold', color=bg_color, ha='center', va='center')
        
            ax.scatter(59.5, 6, s=2000, marker='s', color=col2, zorder=3)
            ax.scatter(59.5, 34, s=2000, marker='s', color=col2, zorder=3)
            ax.scatter(59.5, 62, s=2000, marker='s', color=col2, zorder=3)
            ax.text(59.5, 6, f'{len(arigt)}', fontsize=30, fontweight='bold', color=bg_color, ha='center', va='center')
            ax.text(59.5, 34, f'{len(acent)}', fontsize=30, fontweight='bold', color=bg_color, ha='center', va='center')
            ax.text(59.5, 62, f'{len(aleft)}', fontsize=30, fontweight='bold', color=bg_color, ha='center', va='center')
            
            return 
        
        
        # Crosses
        
        def Crosses(ax):
            pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
            pitch.draw(ax=ax)
            ax.set_ylim(-0.5,68.5)
            ax.set_xlim(-0.5,105.5)
        
            home_cross = df[(df['teamName']==hteamName) & (df['type']=='Pass') & (df['type_value_Cross']==2) & (df['type_value_Corner taken']!=6) & (df['type_value_Free kick taken']!=5)]
            away_cross = df[(df['teamName']==ateamName) & (df['type']=='Pass') & (df['type_value_Cross']==2) & (df['type_value_Corner taken']!=6) & (df['type_value_Free kick taken']!=5)]
        
            hsuc = 0
            hunsuc = 0
            asuc = 0
            aunsuc = 0
        
            # iterating through each pass and coloring according to successful or not
            for index, row in home_cross.iterrows():
                if row['outcomeType'] == 'Successful':
                    arrow = patches.FancyArrowPatch((105-row['x'], 68-row['y']), (105-row['endX'], 68-row['endY']), arrowstyle='->', mutation_scale=15, color=col1, linewidth=1.5, alpha=1)
                    ax.add_patch(arrow)
                    hsuc += 1
                else:
                    arrow = patches.FancyArrowPatch((105-row['x'], 68-row['y']), (105-row['endX'], 68-row['endY']), arrowstyle='->', mutation_scale=10, color=line_color, linewidth=1.5, alpha=.25)
                    ax.add_patch(arrow)
                    hunsuc += 1
        
            for index, row in away_cross.iterrows():
                if row['outcomeType'] == 'Successful':
                    arrow = patches.FancyArrowPatch((row['x'], row['y']), (row['endX'], row['endY']), arrowstyle='->', mutation_scale=15, color=col2, linewidth=1.5, alpha=1)
                    ax.add_patch(arrow)
                    asuc += 1
                else:
                    arrow = patches.FancyArrowPatch((row['x'], row['y']), (row['endX'], row['endY']), arrowstyle='->', mutation_scale=10, color=line_color, linewidth=1.5, alpha=.25)
                    ax.add_patch(arrow)
                    aunsuc += 1
        
            # Headlines and other texts
            home_left = len(home_cross[home_cross['y']>=34])
            home_right = len(home_cross[home_cross['y']<34])
            away_left = len(away_cross[away_cross['y']>=34])
            away_right = len(away_cross[away_cross['y']<34])
        
            ax.text(51, 2, f"Crosses from\nLeftwing: {home_left}", color=col1, fontsize=15, va='bottom', ha='right')
            ax.text(51, 66, f"Crosses from\nRightwing: {home_right}", color=col1, fontsize=15, va='top', ha='right')
            ax.text(54, 66, f"Crosses from\nLeftwing: {away_left}", color=col2, fontsize=15, va='top', ha='left')
            ax.text(54, 2, f"Crosses from\nRightwing: {away_right}", color=col2, fontsize=15, va='bottom', ha='left')
        
            ax.text(0,-2, f"Successful: {hsuc}", color=col1, fontsize=20, ha='left', va='top')
            ax.text(0,-5.5, f"Unsuccessful: {hunsuc}", color=line_color, fontsize=20, ha='left', va='top')
            ax.text(105,-2, f"Successful: {asuc}", color=col2, fontsize=20, ha='right', va='top')
            ax.text(105,-5.5, f"Unsuccessful: {aunsuc}", color=line_color, fontsize=20, ha='right', va='top')
        
            ax.text(0, 70, f"{hteamName}\n<---Crosses", color=col1, size=25, ha='left', fontweight='bold')
            ax.text(105, 70, f"{ateamName}\nCrosses--->", color=col2, size=25, ha='right', fontweight='bold')
            
            return
        
        
        # High Turnovers
        
        def HighTO(ax):
            pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
            pitch.draw(ax=ax)
            ax.set_ylim(-0.5,68.5)
            ax.set_xlim(-0.5,105.5)
        
            highTO = df.copy()
            highTO['Distance'] = ((highTO['x'] - 105)**2 + (highTO['y'] - 34)**2)**0.5
        
            agoal_count = 0
            # Iterate through the DataFrame
            for i in range(len(highTO)):
                if ((highTO.loc[i, 'type'] in ['BallRecovery', 'Interception']) and 
                    (highTO.loc[i, 'teamName'] == ateamName) and 
                    (highTO.loc[i, 'Distance'] <= 40)):
                    
                    possession_id = highTO.loc[i, 'possession_id']
                    
                    # Check the following rows within the same possession
                    j = i + 1
                    while j < len(highTO) and highTO.loc[j, 'possession_id'] == possession_id and highTO.loc[j, 'teamName']==ateamName:
                        if highTO.loc[j, 'type'] == 'Goal' and highTO.loc[j, 'teamName']==ateamName:
                            ax.scatter(highTO.loc[i, 'x'],highTO.loc[i, 'y'], s=600, marker='*', color='green', edgecolor='k', zorder=3)
                            # print(highTO.loc[i, 'type'])
                            agoal_count += 1
                            break
                        j += 1
        
            ashot_count = 0
            # Iterate through the DataFrame
            for i in range(len(highTO)):
                if ((highTO.loc[i, 'type'] in ['BallRecovery', 'Interception']) and 
                    (highTO.loc[i, 'teamName'] == ateamName) and 
                    (highTO.loc[i, 'Distance'] <= 40)):
                    
                    possession_id = highTO.loc[i, 'possession_id']
                    
                    # Check the following rows within the same possession
                    j = i + 1
                    while j < len(highTO) and highTO.loc[j, 'possession_id'] == possession_id and highTO.loc[j, 'teamName']==ateamName:
                        if ('Shot' in highTO.loc[j, 'type']) and (highTO.loc[j, 'teamName']==ateamName):
                            ax.scatter(highTO.loc[i, 'x'],highTO.loc[i, 'y'], s=150, color=col2, edgecolor=bg_color, zorder=2)
                            ashot_count += 1
                            break
                        j += 1
            
            aht_count = 0
            p_list = []
            # Iterate through the DataFrame
            for i in range(len(highTO)):
                if ((highTO.loc[i, 'type'] in ['BallRecovery', 'Interception']) and 
                    (highTO.loc[i, 'teamName'] == ateamName) and 
                    (highTO.loc[i, 'Distance'] <= 40)):
                    
                    # Check the following rows
                    j = i + 1
                    if ((highTO.loc[j, 'teamName']==ateamName) and
                        (highTO.loc[j, 'type']!='Dispossessed') and (highTO.loc[j, 'type']!='OffsidePass')):
                        ax.scatter(highTO.loc[i, 'x'],highTO.loc[i, 'y'], s=100, color='None', edgecolor=col2)
                        aht_count += 1
                        p_list.append(highTO.loc[i, 'shortName'])
        
        
        
        
        
            
            hgoal_count = 0
            # Iterate through the DataFrame
            for i in range(len(highTO)):
                if ((highTO.loc[i, 'type'] in ['BallRecovery', 'Interception']) and 
                    (highTO.loc[i, 'teamName'] == hteamName) and 
                    (highTO.loc[i, 'Distance'] <= 40)):
                    
                    possession_id = highTO.loc[i, 'possession_id']
                    
                    # Check the following rows within the same possession
                    j = i + 1
                    while j < len(highTO) and highTO.loc[j, 'possession_id'] == possession_id and highTO.loc[j, 'teamName']==hteamName:
                        if highTO.loc[j, 'type'] == 'Goal' and highTO.loc[j, 'teamName']==hteamName:
                            ax.scatter(105-highTO.loc[i, 'x'],68-highTO.loc[i, 'y'], s=600, marker='*', color='green', edgecolor='k', zorder=3)
                            # print(highTO.loc[i, 'name'])
                            hgoal_count += 1
                            break
                        j += 1
        
            hshot_count = 0
            # Iterate through the DataFrame
            for i in range(len(highTO)):
                if ((highTO.loc[i, 'type'] in ['BallRecovery', 'Interception']) and 
                    (highTO.loc[i, 'teamName'] == hteamName) and 
                    (highTO.loc[i, 'Distance'] <= 40)):
                    
                    possession_id = highTO.loc[i, 'possession_id']
                    
                    # Check the following rows within the same possession
                    j = i + 1
                    while j < len(highTO) and highTO.loc[j, 'possession_id'] == possession_id and highTO.loc[j, 'teamName']==hteamName:
                        if ('Shot' in highTO.loc[j, 'type']) and (highTO.loc[j, 'teamName']==hteamName):
                            ax.scatter(105-highTO.loc[i, 'x'],68-highTO.loc[i, 'y'], s=150, color=col1, edgecolor=bg_color, zorder=2)
                            hshot_count += 1
                            break
                        j += 1
            
            hht_count = 0
            p_list = []
            # Iterate through the DataFrame
            for i in range(len(highTO)):
                if ((highTO.loc[i, 'type'] in ['BallRecovery', 'Interception']) and 
                    (highTO.loc[i, 'teamName'] == hteamName) and 
                    (highTO.loc[i, 'Distance'] <= 40)):
                    
                    # Check the following rows
                    j = i + 1
                    if ((highTO.loc[j, 'teamName']==hteamName) and
                        (highTO.loc[j, 'type']!='Dispossessed') and (highTO.loc[j, 'type']!='OffsidePass')):
                        ax.scatter(105-highTO.loc[i, 'x'],68-highTO.loc[i, 'y'], s=100, color='None', edgecolor=col1)
                        hht_count += 1
                        p_list.append(highTO.loc[i, 'shortName'])
        
            # Plotting the half circle
            left_circle = plt.Circle((0,34), 40, color=col1, fill=True, alpha=0.25, linestyle='dashed')
            ax.add_artist(left_circle)
            right_circle = plt.Circle((105,34), 40, color=col2, fill=True, alpha=0.25, linestyle='dashed')
            ax.add_artist(right_circle)
            # Set the aspect ratio to be equal
            ax.set_aspect('equal', adjustable='box')
            # Headlines and other texts
            ax.text(0, 70, f"{hteamName}\nHigh Turnover: {hht_count}", color=col1, size=25, ha='left', fontweight='bold')
            ax.text(105, 70, f"{ateamName}\nHigh Turnover: {aht_count}", color=col2, size=25, ha='right', fontweight='bold')
            ax.text(0,  -3, '<---Attacking Direction', color=col1, fontsize=13, ha='left', va='center')
            ax.text(105,-3, 'Attacking Direction--->', color=col2, fontsize=13, ha='right', va='center')
            
            return 
        
        
        # Congestion
        
        def plot_congestion(ax):
            pcmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors",  [col2, 'gray', col1], N=20)
            df1 = df[(df['teamName']==hteamName) & (~df['type'].str.contains('SubstitutionOff|SubstitutionOn|Challenge|Card')) &
                     (df['type_value_Corner taken']!=6) & (df['type_value_Free kick taken']!=5)]
            df2 = df[(df['teamName']==ateamName) & (~df['type'].str.contains('SubstitutionOff|SubstitutionOn|Challenge|Card')) & 
                     (df['type_value_Corner taken']!=6) & (df['type_value_Free kick taken']!=5)]
            df2['x'] = 105-df2['x']
            df2['y'] =  68-df2['y']
            pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2, line_zorder=6)
            pitch.draw(ax=ax)
            ax.set_ylim(-0.5,68.5)
            ax.set_xlim(-0.5,105.5)
        
            bin_statistic1 = pitch.bin_statistic(df1.x, df1.y, bins=(6,5), statistic='count', normalize=False)
            bin_statistic2 = pitch.bin_statistic(df2.x, df2.y, bins=(6,5), statistic='count', normalize=False)
        
            # Assuming 'cx' and 'cy' are as follows:
            cx = np.array([[ 8.75, 26.25, 43.75, 61.25, 78.75, 96.25],
                       [ 8.75, 26.25, 43.75, 61.25, 78.75, 96.25],
                       [ 8.75, 26.25, 43.75, 61.25, 78.75, 96.25],
                       [ 8.75, 26.25, 43.75, 61.25, 78.75, 96.25],
                       [ 8.75, 26.25, 43.75, 61.25, 78.75, 96.25]])
        
            cy = np.array([[61.2, 61.2, 61.2, 61.2, 61.2, 61.2],
                       [47.6, 47.6, 47.6, 47.6, 47.6, 47.6],
                       [34.0, 34.0, 34.0, 34.0, 34.0, 34.0],
                       [20.4, 20.4, 20.4, 20.4, 20.4, 20.4],
                       [ 6.8,  6.8,  6.8,  6.8,  6.8,  6.8]])
        
            # Flatten the arrays
            cx_flat = cx.flatten()
            cy_flat = cy.flatten()
        
            # Create a DataFrame
            df_cong = pd.DataFrame({'cx': cx_flat, 'cy': cy_flat})
        
            hd_values = []
        
        
            # Loop through the 2D arrays
            for i in range(bin_statistic1['statistic'].shape[0]):
                for j in range(bin_statistic1['statistic'].shape[1]):
                    stat1 = bin_statistic1['statistic'][i, j]
                    stat2 = bin_statistic2['statistic'][i, j]
                
                    if (stat1 / (stat1 + stat2)) > 0.55:
                        hd_values.append(1)
                    elif (stat1 / (stat1 + stat2)) < 0.45:
                        hd_values.append(0)
                    else:
                        hd_values.append(0.5)
        
            df_cong['hd']=hd_values
            bin_stat = pitch.bin_statistic(df_cong.cx, df_cong.cy, bins=(6,5), values=df_cong['hd'], statistic='sum', normalize=False)
            pitch.heatmap(bin_stat, ax=ax, cmap=pcmap, edgecolors='#000000', lw=0, zorder=3, alpha=0.85)
        
            ax_text(52.5, 71, s=f"<{hteamName}>  |  Contested  |  <{ateamName }>", highlight_textprops=[{'color':col1}, {'color':col2}],
                    color='gray', fontsize=18, ha='center', va='center', ax=ax)
            ax.set_title("Team's Dominating Zone", color=line_color, fontsize=30, fontweight='bold', y=1.075)
            ax.text(0,  -3, 'Attacking Direction--->', color=col1, fontsize=13, ha='left', va='center')
            ax.text(105,-3, '<---Attacking Direction', color=col2, fontsize=13, ha='right', va='center')
        
            ax.vlines(1*(105/6), ymin=0, ymax=68, color=bg_color, lw=2, ls='--', zorder=5)
            ax.vlines(2*(105/6), ymin=0, ymax=68, color=bg_color, lw=2, ls='--', zorder=5)
            ax.vlines(3*(105/6), ymin=0, ymax=68, color=bg_color, lw=2, ls='--', zorder=5)
            ax.vlines(4*(105/6), ymin=0, ymax=68, color=bg_color, lw=2, ls='--', zorder=5)
            ax.vlines(5*(105/6), ymin=0, ymax=68, color=bg_color, lw=2, ls='--', zorder=5)
        
            ax.hlines(1*(68/5), xmin=0, xmax=105, color=bg_color, lw=2, ls='--', zorder=5)
            ax.hlines(2*(68/5), xmin=0, xmax=105, color=bg_color, lw=2, ls='--', zorder=5)
            ax.hlines(3*(68/5), xmin=0, xmax=105, color=bg_color, lw=2, ls='--', zorder=5)
            ax.hlines(4*(68/5), xmin=0, xmax=105, color=bg_color, lw=2, ls='--', zorder=5)
            
            return
            
            
        # Player Stats Counting
        
        # Get unique players
        home_unique_players = homedf['name'].unique()
        away_unique_players = awaydf['name'].unique()
        
        
        # Top Ball Progressor
        # Initialize an empty dictionary to store home players different type of pass counts
        home_progressor_counts = {'name': home_unique_players, 'Progressive Passes': [], 'Progressive Carries': []}
        for name in home_unique_players:
            home_progressor_counts['Progressive Passes'].append(len(df[(df['name'] == name) & (df['prog_pass'] >= 9.144) & (df['x']>=35) & (df['outcomeType']=='Successful') & (df['type_value_Corner taken']!=6) & (df['type_value_Free kick taken']!=5)]))
            home_progressor_counts['Progressive Carries'].append(len(df[(df['name'] == name) & (df['prog_carry'] >= 9.144) & (df['endX']>=35)]))
            
        home_progressor_df = pd.DataFrame(home_progressor_counts)
        home_progressor_df['total'] = home_progressor_df['Progressive Passes']+home_progressor_df['Progressive Carries']
        home_progressor_df = home_progressor_df.sort_values(by='total', ascending=False)
        home_progressor_df.reset_index(drop=True, inplace=True)
        home_progressor_df = home_progressor_df.head(5)
        home_progressor_df['shortName'] = home_progressor_df['name'].apply(get_short_name)
        
        # Initialize an empty dictionary to store away players different type of pass counts
        away_progressor_counts = {'name': away_unique_players, 'Progressive Passes': [], 'Progressive Carries': []}
        for name in away_unique_players:
            away_progressor_counts['Progressive Passes'].append(len(df[(df['name'] == name) & (df['prog_pass'] >= 9.144) & (df['x']>=35) & (df['outcomeType']=='Successful') & (df['type_value_Corner taken']!=6) & (df['type_value_Free kick taken']!=5)]))
            away_progressor_counts['Progressive Carries'].append(len(df[(df['name'] == name) & (df['prog_carry'] >= 9.144) & (df['endX']>=35)]))
            
        away_progressor_df = pd.DataFrame(away_progressor_counts)
        away_progressor_df['total'] = away_progressor_df['Progressive Passes']+away_progressor_df['Progressive Carries']
        away_progressor_df = away_progressor_df.sort_values(by='total', ascending=False)
        away_progressor_df.reset_index(drop=True, inplace=True)
        away_progressor_df = away_progressor_df.head(5)
        away_progressor_df['shortName'] = away_progressor_df['name'].apply(get_short_name)
        
        
        # Top Threate Creators
        # Initialize an empty dictionary to store home players different type of Carries counts
        home_xT_counts = {'name': home_unique_players, 'xT from Pass': [], 'xT from Carry': []}
        for name in home_unique_players:
            home_xT_counts['xT from Pass'].append((df[(df['name'] == name) & (df['type'] == 'Pass') & (df['xT']>=0) & (df['outcomeType']=='Successful') & (df['type_value_Corner taken']!=6) & (df['type_value_Free kick taken']!=5)])['xT'].sum().round(2))
            home_xT_counts['xT from Carry'].append((df[(df['name'] == name) & (df['type'] == 'Carry') & (df['xT']>=0)])['xT'].sum().round(2))
        home_xT_df = pd.DataFrame(home_xT_counts)
        home_xT_df['total'] = home_xT_df['xT from Pass']+home_xT_df['xT from Carry']
        home_xT_df = home_xT_df.sort_values(by='total', ascending=False)
        home_xT_df.reset_index(drop=True, inplace=True)
        home_xT_df = home_xT_df.head(5)
        home_xT_df['shortName'] = home_xT_df['name'].apply(get_short_name)
        
        # Initialize an empty dictionary to store home players different type of Carries counts
        away_xT_counts = {'name': away_unique_players, 'xT from Pass': [], 'xT from Carry': []}
        for name in away_unique_players:
            away_xT_counts['xT from Pass'].append((df[(df['name'] == name) & (df['type'] == 'Pass') & (df['xT']>=0) & (df['outcomeType']=='Successful') & (df['type_value_Corner taken']!=6) & (df['type_value_Free kick taken']!=5)])['xT'].sum().round(2))
            away_xT_counts['xT from Carry'].append((df[(df['name'] == name) & (df['type'] == 'Carry') & (df['xT']>=0)])['xT'].sum().round(2))
        away_xT_df = pd.DataFrame(away_xT_counts)
        away_xT_df['total'] = away_xT_df['xT from Pass']+away_xT_df['xT from Carry']
        away_xT_df = away_xT_df.sort_values(by='total', ascending=False)
        away_xT_df.reset_index(drop=True, inplace=True)
        away_xT_df = away_xT_df.head(5)
        away_xT_df['shortName'] = away_xT_df['name'].apply(get_short_name)
        
        
        # Shot Sequence Involvement
        df_no_carry = df[df['type']!='Carry']
        # Initialize an empty dictionary to store home players different type of shot sequence counts
        home_shot_seq_counts = {'name': home_unique_players, 'Shots': [], 'Shot Assist': [], 'Buildup to shot': []}
        # Putting counts in those lists
        for name in home_unique_players:
            home_shot_seq_counts['Shots'].append(len(df[(df['name'] == name) & ((df['type']=='MissedShots') | (df['type']=='SavedShot') | (df['type']=='ShotOnPost') | (df['type']=='Goal'))]))
            home_shot_seq_counts['Shot Assist'].append(len(df[(df['name'] == name) & (df['type'] == 'Pass') & (df['type_value_Assist']==210)]))
            home_shot_seq_counts['Buildup to shot'].append(len(df_no_carry[(df_no_carry['name'] == name) & (df_no_carry['type'] == 'Pass') & (df['type_value_Assist'].shift(-1)==210)]))
        # converting that list into a dataframe
        home_sh_sq_df = pd.DataFrame(home_shot_seq_counts)
        home_sh_sq_df['total'] = home_sh_sq_df['Shots']+home_sh_sq_df['Shot Assist']+home_sh_sq_df['Buildup to shot']
        home_sh_sq_df = home_sh_sq_df.sort_values(by='total', ascending=False)
        home_sh_sq_df.reset_index(drop=True, inplace=True)
        home_sh_sq_df = home_sh_sq_df.head(5)
        home_sh_sq_df['shortName'] = home_sh_sq_df['name'].apply(get_short_name)
        
        # Initialize an empty dictionary to store away players different type of shot sequence counts
        away_shot_seq_counts = {'name': away_unique_players, 'Shots': [], 'Shot Assist': [], 'Buildup to shot': []}
        for name in away_unique_players:
            away_shot_seq_counts['Shots'].append(len(df[(df['name'] == name) & ((df['type']=='MissedShots') | (df['type']=='SavedShot') | (df['type']=='ShotOnPost') | (df['type']=='Goal'))]))
            away_shot_seq_counts['Shot Assist'].append(len(df[(df['name'] == name) & (df['type'] == 'Pass') & (df['type_value_Assist']==210)]))
            away_shot_seq_counts['Buildup to shot'].append(len(df_no_carry[(df_no_carry['name'] == name) & (df_no_carry['type'] == 'Pass') & (df['type_value_Assist'].shift(-1)==210)]))
        away_sh_sq_df = pd.DataFrame(away_shot_seq_counts)
        away_sh_sq_df['total'] = away_sh_sq_df['Shots']+away_sh_sq_df['Shot Assist']+away_sh_sq_df['Buildup to shot']
        away_sh_sq_df = away_sh_sq_df.sort_values(by='total', ascending=False)
        away_sh_sq_df.reset_index(drop=True, inplace=True)
        away_sh_sq_df = away_sh_sq_df.head(5)
        away_sh_sq_df['shortName'] = away_sh_sq_df['name'].apply(get_short_name)
        
        
        # Top Defenders
        # Initialize an empty dictionary to store home players different type of defensive actions counts
        home_defensive_actions_counts = {'name': home_unique_players, 'Tackles': [], 'Interceptions': [], 'Clearance': []}
        for name in home_unique_players:
            home_defensive_actions_counts['Tackles'].append(len(df[(df['name'] == name) & (df['type'] == 'Tackle') & (df['outcomeType']=='Successful')]))
            home_defensive_actions_counts['Interceptions'].append(len(df[(df['name'] == name) & (df['type'] == 'Interception')]))
            home_defensive_actions_counts['Clearance'].append(len(df[(df['name'] == name) & (df['type'] == 'Clearance')]))
        home_defender_df = pd.DataFrame(home_defensive_actions_counts)
        home_defender_df['total'] = home_defender_df['Tackles']+home_defender_df['Interceptions']+home_defender_df['Clearance']
        home_defender_df = home_defender_df.sort_values(by='total', ascending=False)
        home_defender_df.reset_index(drop=True, inplace=True)
        home_defender_df = home_defender_df.head(5)
        home_defender_df['shortName'] = home_defender_df['name'].apply(get_short_name)
        
        # Initialize an empty dictionary to store away players different type of defensive actions counts
        away_defensive_actions_counts = {'name': away_unique_players, 'Tackles': [], 'Interceptions': [], 'Clearance': []}
        for name in away_unique_players:
            away_defensive_actions_counts['Tackles'].append(len(df[(df['name'] == name) & (df['type'] == 'Tackle') & (df['outcomeType']=='Successful')]))
            away_defensive_actions_counts['Interceptions'].append(len(df[(df['name'] == name) & (df['type'] == 'Interception')]))
            away_defensive_actions_counts['Clearance'].append(len(df[(df['name'] == name) & (df['type'] == 'Clearance')]))
        away_defender_df = pd.DataFrame(away_defensive_actions_counts)
        away_defender_df['total'] = away_defender_df['Tackles']+away_defender_df['Interceptions']+away_defender_df['Clearance']
        away_defender_df = away_defender_df.sort_values(by='total', ascending=False)
        away_defender_df.reset_index(drop=True, inplace=True)
        away_defender_df = away_defender_df.head(5)
        away_defender_df['shortName'] = away_defender_df['name'].apply(get_short_name)
        
        # Get unique players
        unique_players = df['name'].unique()
        
        
        # Top Ball Progressor
        # Initialize an empty dictionary to store home players different type of pass counts
        progressor_counts = {'name': unique_players, 'Progressive Passes': [], 'Progressive Carries': []}
        for name in unique_players:
            progressor_counts['Progressive Passes'].append(len(df[(df['name'] == name) & (df['prog_pass'] >= 9.144) & (df['x']>=35) & (df['outcomeType']=='Successful') & (df['type_value_Corner taken']!=6) & (df['type_value_Free kick taken']!=5)]))
            progressor_counts['Progressive Carries'].append(len(df[(df['name'] == name) & (df['prog_carry'] >= 9.144) & (df['endX']>=35)]))
        
        progressor_df = pd.DataFrame(progressor_counts)
        progressor_df['total'] = progressor_df['Progressive Passes']+progressor_df['Progressive Carries']
        progressor_df = progressor_df.sort_values(by='total', ascending=False)
        progressor_df.reset_index(drop=True, inplace=True)
        progressor_df = progressor_df.head(10)
        progressor_df['shortName'] = progressor_df['name'].apply(get_short_name)
        
        
        
        
        # Top Threate Creators
        # Initialize an empty dictionary to store home players different type of Carries counts
        xT_counts = {'name': unique_players, 'xT from Pass': [], 'xT from Carry': []}
        for name in unique_players:
            xT_counts['xT from Pass'].append((df[(df['name'] == name) & (df['type'] == 'Pass') & (df['xT']>=0) & (df['outcomeType']=='Successful') & (df['type_value_Corner taken']!=6) & (df['type_value_Free kick taken']!=5)])['xT'].sum().round(2))
            xT_counts['xT from Carry'].append((df[(df['name'] == name) & (df['type'] == 'Carry') & (df['xT']>=0)])['xT'].sum().round(2))
        xT_df = pd.DataFrame(xT_counts)
        xT_df['total'] = xT_df['xT from Pass']+xT_df['xT from Carry']
        xT_df = xT_df.sort_values(by='total', ascending=False)
        xT_df.reset_index(drop=True, inplace=True)
        xT_df = xT_df.head(10)
        xT_df['shortName'] = xT_df['name'].apply(get_short_name)
        
        
        
        
        # Shot Sequence Involvement
        df_no_carry = df[df['type']!='Carry']
        # Initialize an empty dictionary to store home players different type of shot sequence counts
        shot_seq_counts = {'name': unique_players, 'Shots': [], 'Shot Assist': [], 'Buildup to shot': []}
        # Putting counts in those lists
        for name in unique_players:
            shot_seq_counts['Shots'].append(len(df[(df['name'] == name) & ((df['type']=='MissedShots') | (df['type']=='SavedShot') | (df['type']=='ShotOnPost') | (df['type']=='Goal'))]))
            shot_seq_counts['Shot Assist'].append(len(df[(df['name'] == name) & (df['type'] == 'Pass') & (df['type_value_Assist']==210)]))
            shot_seq_counts['Buildup to shot'].append(len(df_no_carry[(df_no_carry['name'] == name) & (df_no_carry['type'] == 'Pass') & (df['type_value_Assist'].shift(-1)==210)]))
        # converting that list into a dataframe
        sh_sq_df = pd.DataFrame(shot_seq_counts)
        sh_sq_df['total'] = sh_sq_df['Shots']+sh_sq_df['Shot Assist']+sh_sq_df['Buildup to shot']
        sh_sq_df = sh_sq_df.sort_values(by='total', ascending=False)
        sh_sq_df.reset_index(drop=True, inplace=True)
        sh_sq_df = sh_sq_df.head(10)
        sh_sq_df['shortName'] = sh_sq_df['name'].apply(get_short_name)
        
        
        
        
        # Top Defenders
        # Initialize an empty dictionary to store home players different type of defensive actions counts
        defensive_actions_counts = {'name': unique_players, 'Tackles': [], 'Interceptions': [], 'Clearance': []}
        for name in unique_players:
            defensive_actions_counts['Tackles'].append(len(df[(df['name'] == name) & (df['type'] == 'Tackle') & (df['outcomeType']=='Successful')]))
            defensive_actions_counts['Interceptions'].append(len(df[(df['name'] == name) & (df['type'] == 'Interception')]))
            defensive_actions_counts['Clearance'].append(len(df[(df['name'] == name) & (df['type'] == 'Clearance')]))
        defender_df = pd.DataFrame(defensive_actions_counts)
        defender_df['total'] = defender_df['Tackles']+defender_df['Interceptions']+defender_df['Clearance']
        defender_df = defender_df.sort_values(by='total', ascending=False)
        defender_df.reset_index(drop=True, inplace=True)
        defender_df = defender_df.head(10)
        defender_df['shortName'] = defender_df['name'].apply(get_short_name)
        
        
        
        # Top Passer's PassMap
        
        def home_player_passmap(ax):
            pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
            pitch.draw(ax=ax)
            ax.set_xlim(-0.5, 105.5)
            ax.set_ylim(-0.5, 68.5)
        
            # taking the top home passer and plotting his passmap
            home_player_name = home_progressor_df['name'].iloc[0]
        
            acc_pass = df[(df['name']==home_player_name) & (df['type']=='Pass') & (df['outcomeType']=='Successful')]
            pro_pass = acc_pass[(acc_pass['prog_pass']>=9.11) & (acc_pass['x']>=35) & (acc_pass['type_value_Corner taken']!=6) & (acc_pass['type_value_Free kick taken']!=5)]
            pro_carry = df[(df['name']==home_player_name) & (df['prog_carry']>=9.11) & (df['endX']>=35)]
            key_pass = acc_pass[acc_pass['type_value_Assist']==210]
            g_assist = acc_pass[acc_pass['assist']==1]
        
            pitch.lines(acc_pass.x, acc_pass.y, acc_pass.endX, acc_pass.endY, color=line_color, lw=2, alpha=0.15, comet=True, zorder=2, ax=ax)
            pitch.lines(pro_pass.x, pro_pass.y, pro_pass.endX, pro_pass.endY, color=col1, lw=3, alpha=1,    comet=True, zorder=3, ax=ax)
            pitch.lines(key_pass.x, key_pass.y, key_pass.endX, key_pass.endY, color=violet,     lw=4, alpha=1,    comet=True, zorder=4, ax=ax)
            pitch.lines(g_assist.x, g_assist.y, g_assist.endX, g_assist.endY, color='green',      lw=4, alpha=1,    comet=True, zorder=5, ax=ax)
        
            ax.scatter(acc_pass.endX, acc_pass.endY, s=30, color=bg_color,    edgecolor='gray', alpha=1, zorder=2)
            ax.scatter(pro_pass.endX, pro_pass.endY, s=40, color=bg_color,  edgecolor= col1,  alpha=1, zorder=3)
            ax.scatter(key_pass.endX, key_pass.endY, s=50, color=bg_color,  edgecolor=violet, alpha=1, zorder=4)
            ax.scatter(g_assist.endX, g_assist.endY, s=50, color=bg_color,  edgecolor= 'green', alpha=1, zorder=5)
        
            for index, row in pro_carry.iterrows():
                arrow = patches.FancyArrowPatch((row['x'], row['y']), (row['endX'], row['endY']), arrowstyle='->', color=col1, zorder=4, mutation_scale=20, 
                                                alpha=0.9, linewidth=2, linestyle='--')
                ax.add_patch(arrow) 
        
        
            home_name_show = home_progressor_df['shortName'].iloc[0]
            ax.set_title(f"{home_name_show} PassMap", color=col1, fontsize=25, fontweight='bold', y=1.03)
            ax.text(0,-3, f'Prog. Pass: {len(pro_pass)}          Prog. Carry: {len(pro_carry)}', fontsize=15, color=col1, ha='left', va='center')
            ax_text(105,-3, s=f'Key Pass: {len(key_pass)}          <Assist: {len(g_assist)}>', fontsize=15, color=violet, ha='right', va='center',
                    highlight_textprops=[{'color':'green'}], ax=ax)
        
        def away_player_passmap(ax):
            pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
            pitch.draw(ax=ax)
            ax.set_xlim(-0.5, 105.5)
            ax.set_ylim(-0.5, 68.5)
            ax.invert_xaxis()
            ax.invert_yaxis()
        
            # taking the top away passer and plotting his passmap
            away_player_name = away_progressor_df['name'].iloc[0]
            
            acc_pass = df[(df['name']==away_player_name) & (df['type']=='Pass') & (df['outcomeType']=='Successful')]
            pro_pass = acc_pass[(acc_pass['prog_pass']>=9.11) & (acc_pass['x']>=35) & (acc_pass['type_value_Corner taken']!=6) & (acc_pass['type_value_Free kick taken']!=5)]
            pro_carry = df[(df['name']==away_player_name) & (df['prog_carry']>=9.11) & (df['endX']>=35)]
            key_pass = acc_pass[acc_pass['type_value_Assist']==210]
            g_assist = acc_pass[acc_pass['assist']==1]
        
            pitch.lines(acc_pass.x, acc_pass.y, acc_pass.endX, acc_pass.endY, color=line_color, lw=2, alpha=0.15, comet=True, zorder=2, ax=ax)
            pitch.lines(pro_pass.x, pro_pass.y, pro_pass.endX, pro_pass.endY, color=col2      , lw=3, alpha=1,    comet=True, zorder=3, ax=ax)
            pitch.lines(key_pass.x, key_pass.y, key_pass.endX, key_pass.endY, color=violet,     lw=4, alpha=1,    comet=True, zorder=4, ax=ax)
            pitch.lines(g_assist.x, g_assist.y, g_assist.endX, g_assist.endY, color='green',      lw=4, alpha=1,    comet=True, zorder=5, ax=ax)
        
            ax.scatter(acc_pass.endX, acc_pass.endY, s=30, color=bg_color,    edgecolor='gray', alpha=1, zorder=2)
            ax.scatter(pro_pass.endX, pro_pass.endY, s=40, color=bg_color,  edgecolor= col2,  alpha=1, zorder=3)
            ax.scatter(key_pass.endX, key_pass.endY, s=50, color=bg_color,  edgecolor=violet, alpha=1, zorder=4)
            ax.scatter(g_assist.endX, g_assist.endY, s=50, color=bg_color,  edgecolor= 'green', alpha=1, zorder=5)
        
            for index, row in pro_carry.iterrows():
                arrow = patches.FancyArrowPatch((row['x'], row['y']), (row['endX'], row['endY']), arrowstyle='->', color=col2, zorder=4, mutation_scale=20, 
                                                alpha=0.9, linewidth=2, linestyle='--')
                ax.add_patch(arrow) 
        
        
            away_name_show = away_progressor_df['shortName'].iloc[0]
            ax.set_title(f"{away_name_show} PassMap", color=col2, fontsize=25, fontweight='bold', y=1.03)
            ax.text(0,71, f'Prog. Pass: {len(pro_pass)}          Prog. Carry: {len(pro_carry)}', fontsize=15, color=col2, ha='right', va='center')
            ax_text(105,71, s=f'Key Pass: {len(key_pass)}          <Assist: {len(g_assist)}>', fontsize=15, color=violet, ha='left', va='center',
                    highlight_textprops=[{'color':'green'}], ax=ax)
            
            
        # Forward Pass Receiving
        
        def home_passes_recieved(ax):
            pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
            pitch.draw(ax=ax)
            ax.set_xlim(-0.5, 105.5)
            ax.set_ylim(-0.5, 68.5)
        
            # plotting the home center forward pass receiving
            name = home_Forward
            name_show = get_short_name(name)
            filtered_rows = df[(df['type'] == 'Pass') & (df['outcomeType'] == 'Successful') & (df['name'].shift(-1) == name)]
            keypass_recieved_df = filtered_rows[filtered_rows['type_value_Assist']==210]
            assist_recieved_df = filtered_rows[filtered_rows['assist']==1]
            pr = len(filtered_rows)
            kpr = len(keypass_recieved_df)
            ar = len(assist_recieved_df)
        
            lc1 = pitch.lines(filtered_rows.x, filtered_rows.y, filtered_rows.endX, filtered_rows.endY, lw=3, transparent=True, comet=True,color=col1, ax=ax, alpha=0.5)
            lc2 = pitch.lines(keypass_recieved_df.x, keypass_recieved_df.y, keypass_recieved_df.endX, keypass_recieved_df.endY, lw=4, transparent=True, comet=True,color=violet, ax=ax, alpha=0.75)
            lc3 = pitch.lines(assist_recieved_df.x, assist_recieved_df.y, assist_recieved_df.endX, assist_recieved_df.endY, lw=4, transparent=True, comet=True,color='green', ax=ax, alpha=0.75)
            sc1 = pitch.scatter(filtered_rows.endX, filtered_rows.endY, s=30, edgecolor=col1, linewidth=1, color=bg_color, zorder=2, ax=ax)
            sc2 = pitch.scatter(keypass_recieved_df.endX, keypass_recieved_df.endY, s=40, edgecolor=violet, linewidth=1.5, color=bg_color, zorder=2, ax=ax)
            sc3 = pitch.scatter(assist_recieved_df.endX, assist_recieved_df.endY, s=50, edgecolors='green', linewidths=1, marker='football', c=bg_color, zorder=2, ax=ax)
        
            avg_endY = filtered_rows['endY'].median()
            avg_endX = filtered_rows['endX'].median()
            ax.axvline(x=avg_endX, ymin=0, ymax=68, color='gray', linestyle='--', alpha=0.6, linewidth=2)
            ax.axhline(y=avg_endY, xmin=0, xmax=105, color='gray', linestyle='--', alpha=0.6, linewidth=2)
            ax.set_title(f"{name_show} Passes Recieved", color=col1, fontsize=25, fontweight='bold', y=1.03)
            highlight_text=[{'color':violet}, {'color':'green'}]
            ax_text(52.5,-3, f'Passes Recieved:{pr+kpr} | <Keypasses Recieved:{kpr}> | <Assist Received: {ar}>', color=line_color, fontsize=15, ha='center', 
                    va='center', highlight_textprops=highlight_text, ax=ax)
        
            return
        
        def away_passes_recieved(ax):
            pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
            pitch.draw(ax=ax)
            ax.set_xlim(-0.5, 105.5)
            ax.set_ylim(-0.5, 68.5)
            ax.invert_xaxis()
            ax.invert_yaxis()
        
            # plotting the away center forward pass receiving
            name = away_Forward
            name_show = get_short_name(name)
            filtered_rows = df[(df['type'] == 'Pass') & (df['outcomeType'] == 'Successful') & (df['name'].shift(-1) == name)]
            keypass_recieved_df = filtered_rows[filtered_rows['type_value_Assist']==210]
            assist_recieved_df = filtered_rows[filtered_rows['assist']==1]
            pr = len(filtered_rows)
            kpr = len(keypass_recieved_df)
            ar = len(assist_recieved_df)
        
            lc1 = pitch.lines(filtered_rows.x, filtered_rows.y, filtered_rows.endX, filtered_rows.endY, lw=3, transparent=True, comet=True,color=col2, ax=ax, alpha=0.5)
            lc2 = pitch.lines(keypass_recieved_df.x, keypass_recieved_df.y, keypass_recieved_df.endX, keypass_recieved_df.endY, lw=4, transparent=True, comet=True,color=violet, ax=ax, alpha=0.75)
            lc3 = pitch.lines(assist_recieved_df.x, assist_recieved_df.y, assist_recieved_df.endX, assist_recieved_df.endY, lw=4, transparent=True, comet=True,color='green', ax=ax, alpha=0.75)
            sc1 = pitch.scatter(filtered_rows.endX, filtered_rows.endY, s=30, edgecolor=col2, linewidth=1, color=bg_color, zorder=2, ax=ax)
            sc2 = pitch.scatter(keypass_recieved_df.endX, keypass_recieved_df.endY, s=40, edgecolor=violet, linewidth=1.5, color=bg_color, zorder=2, ax=ax)
            sc3 = pitch.scatter(assist_recieved_df.endX, assist_recieved_df.endY, s=50, edgecolors='green', linewidths=1, marker='football', c=bg_color, zorder=2, ax=ax)
        
            avg_endX = filtered_rows['endX'].median()
            avg_endY = filtered_rows['endY'].median()
            ax.axvline(x=avg_endX, ymin=0, ymax=68, color='gray', linestyle='--', alpha=0.6, linewidth=2)
            ax.axhline(y=avg_endY, xmin=0, xmax=105, color='gray', linestyle='--', alpha=0.6, linewidth=2)
            ax.set_title(f"{name_show} Passes Recieved", color=col2, fontsize=25, fontweight='bold', y=1.03)
            highlight_text=[{'color':violet}, {'color':'green'}]
            ax_text(52.5,71, f'Passes Recieved:{pr+kpr} | <Keypasses Recieved:{kpr}> | <Assist Received: {ar}>', color=line_color, fontsize=15, ha='center', 
                    va='center', highlight_textprops=highlight_text, ax=ax)
        
            return
        
        
        # Top Defenders
        
        def home_player_def_acts(ax):
            pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, line_zorder=2, linewidth=2)
            pitch.draw(ax=ax)
            ax.set_xlim(-0.5, 105.5)
            ax.set_ylim(-12,68.5)
        
            # taking the top home defender and plotting his defensive actions
            home_player_name = home_defender_df['name'].iloc[0]
            home_playerdf = df[(df['name']==home_player_name)]
        
            hp_tk = home_playerdf[home_playerdf['type']=='Tackle']
            hp_intc = home_playerdf[(home_playerdf['type']=='Interception') | (home_playerdf['type']=='BlockedPass')]
            hp_br = home_playerdf[home_playerdf['type']=='BallRecovery']
            hp_cl = home_playerdf[home_playerdf['type']=='Clearance']
            hp_fl = home_playerdf[(home_playerdf['type']=='Foul') & (home_playerdf['outcomeType']=='Unsuccessful')]
            hp_ar = home_playerdf[(home_playerdf['type']=='Aerial')]
        
            sc1 = pitch.scatter(hp_tk.x, hp_tk.y, s=250, c=col1, lw=2.5, edgecolor=col1, marker='+', hatch='/////', ax=ax)
            sc2 = pitch.scatter(hp_intc.x, hp_intc.y, s=250, c='None', lw=2.5, edgecolor=col1, marker='s', hatch='/////', ax=ax)
            sc3 = pitch.scatter(hp_br.x, hp_br.y, s=250, c='None', lw=2.5, edgecolor=col1, marker='o', hatch='/////', ax=ax)
            sc4 = pitch.scatter(hp_cl.x, hp_cl.y, s=250, c='None', lw=2.5, edgecolor=col1, marker='d', hatch='/////', ax=ax)
            sc5 = pitch.scatter(hp_fl.x, hp_fl.y, s=250, c=col1, lw=2.5, edgecolor=col1, marker='x', hatch='/////', ax=ax)
            sc6 = pitch.scatter(hp_ar.x, hp_ar.y, s=250, c='None', lw=2.5, edgecolor=col1, marker='^', hatch='/////', ax=ax)
        
            sc7 =  pitch.scatter(2, -4, s=150, c=col1, lw=2.5, edgecolor=col1, marker='+', hatch='/////', ax=ax)
            sc8 =  pitch.scatter(2, -10, s=150, c='None', lw=2.5, edgecolor=col1, marker='s', hatch='/////', ax=ax)
            sc9 =  pitch.scatter(41, -4, s=150, c='None', lw=2.5, edgecolor=col1, marker='o', hatch='/////', ax=ax)
            sc10 = pitch.scatter(41, -10, s=150, c='None', lw=2.5, edgecolor=col1, marker='d', hatch='/////', ax=ax)
            sc11 = pitch.scatter(103, -4, s=150, c=col1, lw=2.5, edgecolor=col1, marker='x', hatch='/////', ax=ax)
            sc12 = pitch.scatter(103, -10, s=150, c='None', lw=2.5, edgecolor=col1, marker='^', hatch='/////', ax=ax)
        
            ax.text(5, -3, f"Tackle: {len(hp_tk)}\n\nInterception: {len(hp_intc)}", color=col1, ha='left', va='top', fontsize=13)
            ax.text(43, -3, f"BallRecovery: {len(hp_br)}\n\nClearance: {len(hp_cl)}", color=col1, ha='left', va='top', fontsize=13)
            ax.text(100, -3, f"{len(hp_fl)} Fouls\n\n{len(hp_ar)} Aerials", color=col1, ha='right', va='top', fontsize=13)
        
            home_name_show = home_defender_df['shortName'].iloc[0]
            ax.set_title(f"{home_name_show} Defensive Actions", color=col1, fontsize=25, fontweight='bold')
        
        def away_player_def_acts(ax):
            pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, line_zorder=2, linewidth=2)
            pitch.draw(ax=ax)
            ax.set_xlim(-0.5, 105.5)
            ax.set_ylim(-0.5,80)
            ax.invert_xaxis()
            ax.invert_yaxis()
        
            # taking the top home defender and plotting his defensive actions
            away_player_name = away_defender_df['name'].iloc[0]
            away_playerdf = df[(df['name']==away_player_name)]
        
            ap_tk = away_playerdf[away_playerdf['type']=='Tackle']
            ap_intc = away_playerdf[(away_playerdf['type']=='Interception') | (away_playerdf['type']=='BlockedPass')]
            ap_br = away_playerdf[away_playerdf['type']=='BallRecovery']
            ap_cl = away_playerdf[away_playerdf['type']=='Clearance']
            ap_fl = away_playerdf[(away_playerdf['type']=='Foul') & (away_playerdf['outcomeType']=='Unsuccessful')]
            ap_ar = away_playerdf[(away_playerdf['type']=='Aerial')]
        
            sc1 = pitch.scatter(ap_tk.x, ap_tk.y, s=250, c=col2, lw=2.5, edgecolor=col2, marker='+', hatch='/////', ax=ax)
            sc2 = pitch.scatter(ap_intc.x, ap_intc.y, s=250, c='None', lw=2.5, edgecolor=col2, marker='s', hatch='/////', ax=ax)
            sc3 = pitch.scatter(ap_br.x, ap_br.y, s=250, c='None', lw=2.5, edgecolor=col2, marker='o', hatch='/////', ax=ax)
            sc4 = pitch.scatter(ap_cl.x, ap_cl.y, s=250, c='None', lw=2.5, edgecolor=col2, marker='d', hatch='/////', ax=ax)
            sc5 = pitch.scatter(ap_fl.x, ap_fl.y, s=250, c=col2, lw=2.5, edgecolor=col2, marker='x', hatch='/////', ax=ax)
            sc6 = pitch.scatter(ap_ar.x, ap_ar.y, s=250, c='None', lw=2.5, edgecolor=col2, marker='^', hatch='/////', ax=ax)
        
            sc7 =  pitch.scatter(2, 72, s=150, c=col2, lw=2.5, edgecolor=col2, marker='+', hatch='/////', ax=ax)
            sc8 =  pitch.scatter(2, 78, s=150, c='None', lw=2.5, edgecolor=col2, marker='s', hatch='/////', ax=ax)
            sc9 =  pitch.scatter(41, 72, s=150, c='None', lw=2.5, edgecolor=col2, marker='o', hatch='/////', ax=ax)
            sc10 = pitch.scatter(41, 78, s=150, c='None', lw=2.5, edgecolor=col2, marker='d', hatch='/////', ax=ax)
            sc11 = pitch.scatter(103, 72, s=150, c=col2, lw=2.5, edgecolor=col2, marker='x', hatch='/////', ax=ax)
            sc12 = pitch.scatter(103, 78, s=150, c='None', lw=2.5, edgecolor=col2, marker='^', hatch='/////', ax=ax)
        
            ax.text(5, 71, f"Tackle: {len(ap_tk)}\n\nInterception: {len(ap_intc)}", color=col2, ha='right', va='top', fontsize=13)
            ax.text(43, 71, f"BallRecovery: {len(ap_br)}\n\nClearance: {len(ap_cl)}", color=col2, ha='right', va='top', fontsize=13)
            ax.text(100, 71, f"{len(ap_fl)} Fouls\n\n{len(ap_ar)} Aerials", color=col2, ha='left', va='top', fontsize=13)
        
            away_name_show = away_defender_df['shortName'].iloc[0]
            ax.set_title(f"{away_name_show} Defensive Actions", color=col2, fontsize=25, fontweight='bold')
            
            
        # GoalKeeper PassMap
        
        def home_gk(ax):
            df_gk = df[(df['name']==homeGK) & (df['shortName'].notna())]
            gk_pass = df_gk[df_gk['type']=='Pass']
            op_pass = df_gk[(df_gk['type']=='Pass') & (df_gk['type_value_Goal Kick']!=124) & (df_gk['type_value_Free kick taken']!=5)]
            sp_pass = df_gk[(df_gk['type']=='Pass') & ((df_gk['type_value_Goal Kick']==124) | (df_gk['type_value_Free kick taken']==5))]
            pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
            pitch.draw(ax=ax)
            ax.set_xlim(-0.5, 105.5)
            ax.set_ylim(-0.5, 68.5)
            gk_name = df_gk['shortName'].unique()[0]
            op_succ = sp_succ = 0
            for index, row in op_pass.iterrows():
                if row['outcomeType']=='Successful':
                    pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=col1, lw=4, comet=True, alpha=0.5, zorder=2, ax=ax)
                    ax.scatter(row['endX'], row['endY'], s=40, color=col1, edgecolor=line_color, zorder=3)
                    op_succ += 1
                if row['outcomeType']=='Unsuccessful':
                    pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=col1, lw=4, comet=True, alpha=0.5, zorder=2, ax=ax)
                    ax.scatter(row['endX'], row['endY'], s=40, color=bg_color, edgecolor=col1, zorder=3)
            for index, row in sp_pass.iterrows():
                if row['outcomeType']=='Successful':
                    pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=violet, lw=4, comet=True, alpha=0.5, zorder=2, ax=ax)
                    ax.scatter(row['endX'], row['endY'], s=40, color=violet, edgecolor=line_color, zorder=3)
                    sp_succ += 1
                if row['outcomeType']=='Unsuccessful':
                    pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=violet, lw=4, comet=True, alpha=0.35, zorder=2, ax=ax)
                    ax.scatter(row['endX'], row['endY'], s=40, color=bg_color, edgecolor=violet, zorder=3)
        
            op_pass['length'] = np.sqrt((op_pass['x']-op_pass['endX'])**2 + (op_pass['y']-op_pass['endY'])**2)
            sp_pass['length'] = np.sqrt((sp_pass['x']-sp_pass['endX'])**2 + (sp_pass['y']-sp_pass['endY'])**2)
            avg_len_op = round(op_pass['length'].median(), 2)
            avg_len_sp = round(sp_pass['length'].median(), 2)
            
            ax.set_title(f'{gk_name} PassMap', color=col1, fontsize=25, fontweight='bold', y=1.07)
            ax.text(52.5, -3, f'Avg. OpenPlay Pass Length: {avg_len_op}m     |     Avg. SetPiece Pass Length: {avg_len_sp}m', color=line_color, fontsize=14, ha='center', va='center')
            ax_text(52.5, 70, s=f'<Open-play Pass (Acc.): {len(op_pass)} ({op_succ})>     |     <GoalKick/Freekick (Acc.): {len(sp_pass)} ({sp_succ})>', 
                    fontsize=15, highlight_textprops=[{'color':col1}, {'color':violet}], ha='center', va='center', ax=ax)
        
            return
        
        def away_gk(ax):
            df_gk = df[(df['name']==awayGK) & (df['shortName'].notna())]
            gk_pass = df_gk[df_gk['type']=='Pass']
            op_pass = df_gk[(df_gk['type']=='Pass') & (df_gk['type_value_Goal Kick']!=124) & (df_gk['type_value_Free kick taken']!=5)]
            sp_pass = df_gk[(df_gk['type']=='Pass') & ((df_gk['type_value_Goal Kick']==124) | (df_gk['type_value_Free kick taken']==5))]
            pitch = Pitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2)
            pitch.draw(ax=ax)
            ax.set_xlim(-0.5, 105.5)
            ax.set_ylim(-0.5, 68.5)
            ax.invert_xaxis()
            ax.invert_yaxis()
            gk_name = df_gk['shortName'].unique()[0]
            op_succ = sp_succ = 0
            for index, row in op_pass.iterrows():
                if row['outcomeType']=='Successful':
                    pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=col2, lw=4, comet=True, alpha=0.5, zorder=2, ax=ax)
                    ax.scatter(row['endX'], row['endY'], s=40, color=col2, edgecolor=line_color, zorder=3)
                    op_succ += 1
                if row['outcomeType']=='Unsuccessful':
                    pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=col2, lw=4, comet=True, alpha=0.5, zorder=2, ax=ax)
                    ax.scatter(row['endX'], row['endY'], s=40, color=bg_color, edgecolor=col2, zorder=3)
            for index, row in sp_pass.iterrows():
                if row['outcomeType']=='Successful':
                    pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=violet, lw=4, comet=True, alpha=0.5, zorder=2, ax=ax)
                    ax.scatter(row['endX'], row['endY'], s=40, color=violet, edgecolor=line_color, zorder=3)
                    sp_succ += 1
                if row['outcomeType']=='Unsuccessful':
                    pitch.lines(row['x'], row['y'], row['endX'], row['endY'], color=violet, lw=4, comet=True, alpha=0.35, zorder=2, ax=ax)
                    ax.scatter(row['endX'], row['endY'], s=40, color=bg_color, edgecolor=violet, zorder=3)
        
            op_pass['length'] = np.sqrt((op_pass['x']-op_pass['endX'])**2 + (op_pass['y']-op_pass['endY'])**2)
            sp_pass['length'] = np.sqrt((sp_pass['x']-sp_pass['endX'])**2 + (sp_pass['y']-sp_pass['endY'])**2)
            avg_len_op = round(op_pass['length'].median(), 2)
            avg_len_sp = round(sp_pass['length'].median(), 2)
        
            ax.set_title(f'{gk_name} PassMap', color=col2, fontsize=25, fontweight='bold', y=1.07)
            ax.text(52.5, 71, f'Avg. OpenPlay Pass Length: {avg_len_op}m     |     Avg. SetPiece Pass Length: {avg_len_sp}m', color=line_color, fontsize=14, ha='center', va='center')
            ax_text(52.5, -2, s=f'<Open-play Pass (Acc.): {len(op_pass)} ({op_succ})>     |     <GoalKick/Freekick (Acc.): {len(sp_pass)} ({sp_succ})>', 
                    fontsize=15, highlight_textprops=[{'color':col2}, {'color':violet}], ha='center', va='center', ax=ax)
        
            return
        
        
        from matplotlib.ticker import MaxNLocator
        def sh_sq_bar(ax):
          top10_sh_sq = sh_sq_df.nsmallest(10, 'total')['shortName'].tolist()
        
          shsq_sh = sh_sq_df.nsmallest(10, 'total')['Shots'].tolist()
          shsq_sa = sh_sq_df.nsmallest(10, 'total')['Shot Assist'].tolist()
          shsq_bs = sh_sq_df.nsmallest(10, 'total')['Buildup to shot'].tolist()
        
          left1 = [w + x for w, x in zip(shsq_sh, shsq_sa)]
        
          ax.barh(top10_sh_sq, shsq_sh, label='Shot', color=col1, left=0)
          ax.barh(top10_sh_sq, shsq_sa, label='Shot Assist', color=violet, left=shsq_sh)
          ax.barh(top10_sh_sq, shsq_bs, label='Buildup to Shot', color=col2, left=left1)
        
          # Add counts in the middle of the bars (if count > 0)
          for i, player in enumerate(top10_sh_sq):
              for j, count in enumerate([shsq_sh[i], shsq_sa[i], shsq_bs[i]]):
                  if count > 0:
                      x_position = sum([shsq_sh[i], shsq_sa[i]][:j]) + count / 2
                      ax.text(x_position, i, str(count), ha='center', va='center', color=bg_color, fontsize=18, fontweight='bold')
        
          max_x = sh_sq_df['total'].iloc()[0]
          x_coord = [2 * i for i in range(1, int(max_x/2))]
          for x in x_coord:
              ax.axvline(x=x, color='gray', linestyle='--', zorder=2, alpha=0.5)
        
          ax.set_facecolor(bg_color)
          ax.tick_params(axis='x', colors=line_color, labelsize=15)
          ax.tick_params(axis='y', colors=line_color, labelsize=15)
          ax.xaxis.label.set_color(line_color)
          ax.yaxis.label.set_color(line_color)
          for spine in ax.spines.values():
            spine.set_edgecolor(bg_color)
        
          ax.set_title(f"Shot Sequence Involvement", color=line_color, fontsize=25, fontweight='bold')
          ax.legend(fontsize=13)
        
        def passer_bar(ax):
          top10_passers = progressor_df.nsmallest(10, 'total')['shortName'].tolist()
        
          passers_pp = progressor_df.nsmallest(10, 'total')['Progressive Passes'].tolist()
          passers_tp = progressor_df.nsmallest(10, 'total')['Progressive Carries'].tolist()
        
          left1 = [w + x for w, x in zip(passers_pp, passers_tp)]
        
          ax.barh(top10_passers, passers_pp, label='Prog. Pass', color=col1, left=0)
          ax.barh(top10_passers, passers_tp, label='Prog. Carries', color=col2, left=passers_pp)
        
          # Add counts in the middle of the bars (if count > 0)
          for i, player in enumerate(top10_passers):
              for j, count in enumerate([passers_pp[i], passers_tp[i]]):
                  if count > 0:
                      x_position = sum([passers_pp[i], passers_tp[i]][:j]) + count / 2
                      ax.text(x_position, i, str(count), ha='center', va='center', color=bg_color, fontsize=18, fontweight='bold')
        
          max_x = progressor_df['total'].iloc()[0]
          x_coord = [2 * i for i in range(1, int(max_x/2))]
          for x in x_coord:
              ax.axvline(x=x, color='gray', linestyle='--', zorder=2, alpha=0.5)
        
          ax.set_facecolor(bg_color)
          ax.tick_params(axis='x', colors=line_color, labelsize=15)
          ax.tick_params(axis='y', colors=line_color, labelsize=15)
          ax.xaxis.label.set_color(line_color)
          ax.yaxis.label.set_color(line_color)
          for spine in ax.spines.values():
            spine.set_edgecolor(bg_color)
        
          ax.set_title(f"Top10 Ball Progressors", color=line_color, fontsize=25, fontweight='bold')
          ax.legend(fontsize=13)
        
        
        def defender_bar(ax):
          top10_defenders = defender_df.nsmallest(10, 'total')['shortName'].tolist()
        
          defender_tk = defender_df.nsmallest(10, 'total')['Tackles'].tolist()
          defender_in = defender_df.nsmallest(10, 'total')['Interceptions'].tolist()
          defender_ar = defender_df.nsmallest(10, 'total')['Clearance'].tolist()
        
          left1 = [w + x for w, x in zip(defender_tk, defender_in)]
        
          ax.barh(top10_defenders, defender_tk, label='Tackle', color=col1, left=0)
          ax.barh(top10_defenders, defender_in, label='Interception', color=violet, left=defender_tk)
          ax.barh(top10_defenders, defender_ar, label='Clearance', color=col2, left=left1)
        
          # Add counts in the middle of the bars (if count > 0)
          for i, player in enumerate(top10_defenders):
              for j, count in enumerate([defender_tk[i], defender_in[i], defender_ar[i]]):
                  if count > 0:
                      x_position = sum([defender_tk[i], defender_in[i]][:j]) + count / 2
                      ax.text(x_position, i, str(count), ha='center', va='center', color=bg_color, fontsize=18, fontweight='bold')
        
          max_x = defender_df['total'].iloc()[0]
          x_coord = [2 * i for i in range(1, int(max_x/2))]
          for x in x_coord:
              ax.axvline(x=x, color='gray', linestyle='--', zorder=2, alpha=0.5)
        
          ax.set_facecolor(bg_color)
          ax.tick_params(axis='x', colors=line_color, labelsize=15)
          ax.tick_params(axis='y', colors=line_color, labelsize=15)
          ax.xaxis.label.set_color(line_color)
          ax.yaxis.label.set_color(line_color)
          for spine in ax.spines.values():
            spine.set_edgecolor(bg_color)
        
        
          ax.set_title(f"Top10 Defenders", color=line_color, fontsize=25, fontweight='bold')
          ax.legend(fontsize=13)
        
        
        def threat_creators(ax):
          top10_xT = xT_df.nsmallest(10, 'total')['shortName'].tolist()
        
          xT_pass = xT_df.nsmallest(10, 'total')['xT from Pass'].tolist()
          xT_carry = xT_df.nsmallest(10, 'total')['xT from Carry'].tolist()
        
          left1 = [w + x for w, x in zip(xT_pass, xT_carry)]
        
          ax.barh(top10_xT, xT_pass, label='xT from pass', color=col1, left=0)
          ax.barh(top10_xT, xT_carry, label='xT from carry', color=violet, left=xT_pass)
        
          # Add counts in the middle of the bars (if count > 0)
          for i, player in enumerate(top10_xT):
              for j, count in enumerate([xT_pass[i], xT_carry[i]]):
                  if count > 0:
                      x_position = sum([xT_pass[i], xT_carry[i]][:j]) + count / 2
                      ax.text(x_position, i, str(count), ha='center', va='center', color=line_color, fontsize=15, rotation=45)
        
          # max_x = xT_df['total'].iloc()[0]
          # x_coord = [2 * i for i in range(1, int(max_x/2))]
          # for x in x_coord:
          #     ax.axvline(x=x, color='gray', linestyle='--', zorder=2, alpha=0.5)
        
          ax.set_facecolor(bg_color)
          ax.tick_params(axis='x', colors=line_color, labelsize=15)
          ax.tick_params(axis='y', colors=line_color, labelsize=15)
          ax.xaxis.label.set_color(line_color)
          ax.yaxis.label.set_color(line_color)
          for spine in ax.spines.values():
            spine.set_edgecolor(bg_color)
        
        
          ax.set_title(f"Top10 Threatening Players", color=line_color, fontsize=25, fontweight='bold')
          ax.legend(fontsize=13)
          
          
          
        fig1, axs1 = plt.subplots(4,3, figsize=(35,35), facecolor=bg_color)
        
        pass_network(axs1[0,0], hteamName, col1)
        plot_shotmap(axs1[0,1])
        pass_network(axs1[0,2], ateamName, col2)
        
        defensive_heatmap(axs1[1,0], hteamName, col1)
        plot_goalPost(axs1[1,1])
        defensive_heatmap(axs1[1,2], ateamName, col2)
        
        draw_progressive_pass_map(axs1[2,0], hteamName, col1)
        plot_Momentum(axs1[2,1])
        draw_progressive_pass_map(axs1[2,2], ateamName, col2)
        
        draw_progressive_carry_map(axs1[3,0], hteamName, col1)
        plotting_match_stats(axs1[3,1])
        draw_progressive_carry_map(axs1[3,2], ateamName, col2)
        
        highlight_text = [{'color':col1}, {'color':col2}]
        fig_text(0.5, 0.98, f"<{hteamName} {hgoal_count}> - <{agoal_count} {ateamName}>", color=line_color, fontsize=70, fontweight='bold',
                    highlight_textprops=highlight_text, ha='center', va='center', ax=fig1)
        
        # fig.text(0.5, 0.95, f"Primera Division Femenina 2024-25  |  Post Match Report-1", color=line_color, fontsize=30, ha='center', va='center')
        fig1.text(0.5, 0.95, f"{league_name}  |  Post Match Report-1", color=line_color, fontsize=30, ha='center', va='center')
        # fig.text(0.5, 0.95, f"CONMEBOL WC Qualifiers  |  Post Match Report-1", color=line_color, fontsize=30, ha='center', va='center')
        fig1.text(0.5, 0.93, f"Data from: Opta  |  made by: @adnaaan433", color=line_color, fontsize=22.5, ha='center', va='center')
        
        fig1.text(0.125,0.1, 'Attacking Direction ------->', color=col1, fontsize=25, ha='left', va='center')
        fig1.text(0.9,0.1, '<------- Attacking Direction', color=col2, fontsize=25, ha='right', va='center')
        
        
        fig2, axs2 = plt.subplots(4,3, figsize=(35,35), facecolor=bg_color)
        
        Final_third_entry(axs2[0,0], hteamName, col1)
        box_entry(axs2[0,1])
        Final_third_entry(axs2[0,2], ateamName, col2)
        
        zone14hs(axs2[1,0], hteamName, col1)
        Crosses(axs2[1,1])
        zone14hs(axs2[1,2], ateamName, col2)
        
        Pass_end_zone(axs2[2,0], hteamName, pearl_earring_cmaph)
        HighTO(axs2[2,1])
        Pass_end_zone(axs2[2,2], ateamName, pearl_earring_cmapa)
        
        Chance_creating_zone(axs2[3,0], hteamName, pearl_earring_cmaph, col1)
        plot_congestion(axs2[3,1])
        Chance_creating_zone(axs2[3,2], ateamName, pearl_earring_cmapa, col2)
        
        
        highlight_text = [{'color':col1}, {'color':col2}]
        fig_text(0.5, 0.98, f"<{hteamName} {hgoal_count}> - <{agoal_count} {ateamName}>", color=line_color, fontsize=70, fontweight='bold',
                    highlight_textprops=highlight_text, ha='center', va='center', ax=fig2)
        
        
        fig2.text(0.5, 0.95, f"{league_name}  |  Post Match Report-2", color=line_color, fontsize=30, ha='center', va='center')
        fig2.text(0.5, 0.93, f"Data from: Opta  |  made by: @adnaaan433", color=line_color, fontsize=22.5, ha='center', va='center')
        
        
        fig2.text(0.125,0.1, 'Attacking Direction ------->', color=col1, fontsize=25, ha='left', va='center')
        fig2.text(0.9,0.1, '<------- Attacking Direction', color=col2, fontsize=25, ha='right', va='center')
        
        
        column1, column2 = st.columns(2)
        with column1:
            st.write('Post Match Report - 1')
            st.pyplot(fig1)
        with column2:
            st.write('Post Match Report - 2')
            st.pyplot(fig2)
            
            
            
        fig, axs = plt.subplots(4,3, figsize=(35,35), facecolor=bg_color)
        
        home_player_passmap(axs[0,0])
        passer_bar(axs[0,1])
        away_player_passmap(axs[0,2])
        home_passes_recieved(axs[1,0])
        sh_sq_bar(axs[1,1])
        away_passes_recieved(axs[1,2])
        home_player_def_acts(axs[2,0])
        defender_bar(axs[2,1])
        away_player_def_acts(axs[2,2])
        home_gk(axs[3,0])
        threat_creators(axs[3,1])
        away_gk(axs[3,2])
        
        highlight_text = [{'color':col1}, {'color':col2}]
        fig_text(0.5, 0.98, f"<{hteamName} {hgoal_count}> - <{agoal_count} {ateamName}>", color=line_color, fontsize=70, fontweight='bold',
                    highlight_textprops=highlight_text, ha='center', va='center', ax=fig)
        
        
        # fig.text(0.5, 0.95, f"Primera Division Femenina 2024-25 |  Top Players of the Match", color=line_color, fontsize=30, ha='center', va='center')
        fig.text(0.5, 0.95, f"{league_name} |  Top Players of the Match", color=line_color, fontsize=30, ha='center', va='center')
        fig.text(0.5, 0.93, f"Data from: Opta  |  made by: @adnaaan433", color=line_color, fontsize=22.5, ha='center', va='center')
        fig.text(0.125,0.097, 'Attacking Direction ------->', color=col1, fontsize=25, ha='left', va='center')
        fig.text(0.9,0.097, '<------- Attacking Direction', color=col2, fontsize=25, ha='right', va='center')
        
        st.write('Top Players of the Match') 
        st.pyplot(fig)


        def offensive_actions(ax, pname):
            # Viz Dfs:
            playerdf = df[df['name']==pname]
            passdf = playerdf[playerdf['type']=='Pass']
            succ_passdf = passdf[passdf['outcomeType']=='Successful']
            prg_pass = playerdf[(playerdf['prog_pass']>9.144) & (playerdf['outcomeType']=='Successful') & (playerdf['x']>35) &
                                (playerdf['type_value_Corner taken']!=6) & (playerdf['type_value_Free kick taken']!=5)]
            prg_carry = playerdf[(playerdf['prog_carry']>9.144) & (playerdf['endX']>35)]
            cc = playerdf[(playerdf['type_value_Assist']==210)]
            ga = playerdf[(playerdf['assist']==1)]
            goal = playerdf[(playerdf['type']=='Goal') & (playerdf['isOwnGoal'].isna())]
            owngoal = playerdf[(playerdf['type']=='Goal') & (playerdf['isOwnGoal'].notna())]
            ontr = playerdf[(playerdf['type']=='SavedShot')]
            oftr = playerdf[playerdf['type'].isin(['MissedShots', 'ShotOnPost'])]
            takeOns = playerdf[(playerdf['type']=='TakeOn') & (playerdf['outcomeType']=='Successful')]
            takeOnu = playerdf[(playerdf['type']=='TakeOn') & (playerdf['outcomeType']=='Unsuccessful')]

            # Pitch Plot
            pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, line_zorder=2, linewidth=2, pad_bottom=27)
            pitch.draw(ax=ax)

            # line, arrow, scatter Plots
            pitch.lines(succ_passdf.x, succ_passdf.y, succ_passdf.endX, succ_passdf.endY, color='gray', comet=True, lw=2, alpha=0.65, zorder=1, ax=ax)
            pitch.scatter(succ_passdf.endX, succ_passdf.endY, color=bg_color, ec='gray', s=20, zorder=2, ax=ax)
            pitch.lines(prg_pass.x, prg_pass.y, prg_pass.endX, prg_pass.endY, color=col2, comet=True, lw=3, zorder=2, ax=ax)
            pitch.scatter(prg_pass.endX, prg_pass.endY, color=bg_color, ec=col2, s=40, zorder=3, ax=ax)
            pitch.lines(cc.x, cc.y, cc.endX, cc.endY, color=violet, comet=True, lw=3.5, zorder=3, ax=ax)
            pitch.scatter(cc.endX, cc.endY, color=bg_color, ec=violet, s=50, lw=1.5, zorder=4, ax=ax)
            pitch.lines(ga.x, ga.y, ga.endX, ga.endY, color='green', comet=True, lw=4, zorder=4, ax=ax)
            pitch.scatter(ga.endX, ga.endY, color=bg_color, ec='green', s=60, lw=2, zorder=5, ax=ax)

            for index, row in prg_carry.iterrows():
                arrow = patches.FancyArrowPatch((row['y'], row['x']), (row['endY'], row['endX']), arrowstyle='->', color=col2, zorder=2, mutation_scale=20, 
                                                linewidth=2, linestyle='--')
                ax.add_patch(arrow)

            pitch.scatter(goal.x, goal.y, c=bg_color, edgecolors='green', linewidths=1.2, s=300, marker='football', zorder=10, ax=ax)
            pitch.scatter(owngoal.x, owngoal.y, c=bg_color, edgecolors='orange', linewidths=1.2, s=300, marker='football', zorder=10, ax=ax)
            pitch.scatter(ontr.x, ontr.y, c=col1, edgecolors=line_color, linewidths=1.2, s=200, alpha=0.75, zorder=9, ax=ax)
            pitch.scatter(oftr.x, oftr.y, c=bg_color, edgecolors=col1, linewidths=1.2, s=200, alpha=0.75, zorder=8, ax=ax)

            pitch.scatter(takeOns.x, takeOns.y, c='orange', edgecolors=line_color, marker='h', s=200, alpha=0.75, zorder=7, ax=ax)
            pitch.scatter(takeOnu.x, takeOnu.y, c=bg_color, edgecolors='orange', marker='h', lw=1.2, hatch='//////', s=200, alpha=0.85, zorder=7, ax=ax)

            # Stats:
            pitch.scatter(-5, 68, c=bg_color, edgecolors='green', linewidths=1.2, s=300, marker='football', zorder=10, ax=ax)
            pitch.scatter(-10, 68, c=col1, edgecolors=line_color, linewidths=1.2, s=300, alpha=0.75, zorder=9, ax=ax)
            pitch.scatter(-15, 68, c=bg_color, edgecolors=col1, linewidths=1.2, s=300, alpha=0.75, zorder=8, ax=ax)
            pitch.scatter(-20, 68, c='orange', edgecolors=line_color, marker='h', s=300, alpha=0.75, zorder=7, ax=ax)
            pitch.scatter(-25, 68, c=bg_color, edgecolors='orange', marker='h', lw=1.2, hatch='//////', s=300, alpha=0.85, zorder=7, ax=ax)
            if len(owngoal)>0:
                ax_text(64, -4.5, f'Goals: {len(goal)} | <OwnGoal: {len(owngoal)}>', fontsize=12, highlight_textprops=[{'color':'orange'}], ax=ax)
            else:
                ax.text(64, -5.5, f'Goals: {len(goal)}', fontsize=12)
            ax.text(64, -10.5, f'Shots on Target: {len(ontr)}', fontsize=12)
            ax.text(64, -15.5, f'Shots off Target: {len(oftr)}', fontsize=12)
            ax.text(64, -20.5, f'TakeOn (Succ.): {len(takeOns)}', fontsize=12)
            ax.text(64, -25.5, f'TakeOn (Unsucc.): {len(takeOnu)}', fontsize=12)

            pitch.lines(-5, 34, -5, 24, color='gray', comet=True, lw=2, alpha=0.65, zorder=1, ax=ax)
            pitch.scatter(-5, 24, color=bg_color, ec='gray', s=20, zorder=2, ax=ax)
            pitch.lines(-10, 34, -10, 24, color=col2, comet=True, lw=3, zorder=2, ax=ax)
            pitch.scatter(-10, 24, color=bg_color, ec=col2, s=40, zorder=3, ax=ax)
            arrow = patches.FancyArrowPatch((34, -15), (23, -15), arrowstyle='->', color=col2, zorder=2, mutation_scale=20, 
                                                linewidth=2, linestyle='--')
            ax.add_patch(arrow)
            pitch.lines(-20, 34, -20, 24, color=violet, comet=True, lw=3.5, zorder=3, ax=ax)
            pitch.scatter(-20, 24, color=bg_color, ec=violet, s=50, lw=1.5, zorder=4, ax=ax)
            pitch.lines(-25, 34, -25, 24, color='green', comet=True, lw=4, zorder=4, ax=ax)
            pitch.scatter(-25, 24, color=bg_color, ec='green', s=60, lw=2, zorder=5, ax=ax)

            ax.text(21, -5.5, f'Successful Pass: {len(succ_passdf)}', fontsize=12)
            ax.text(21, -10.5, f'Porgressive Pass: {len(prg_pass)}', fontsize=12)
            ax.text(21, -15.5, f'Porgressive Carry: {len(prg_carry)}', fontsize=12)
            ax.text(21, -20.5, f'Key Passes: {len(cc)}', fontsize=12)
            ax.text(21, -25.5, f'Assists: {len(ga)}', fontsize=12)

            ax.text(34, 110, 'Offensive Actions', fontsize=20, fontweight='bold', ha='center', va='center')
            return
        

        def pass_receiving_and_touchmap(ax, pname):
            # Viz Dfs:
            playerdf = df[df['name']==pname]
            touch_df = playerdf[(playerdf['x']>0) & (playerdf['y']>0)]
            pass_rec = df[(df['type']=='Pass') & (df['outcomeType']=='Successful') & (df['name'].shift(-1)==pname)]
            # touch_df = pd.concat([acts_df, pass_rec], ignore_index=True)
            actual_touch = playerdf[playerdf['isTouch']==1]

            fthd_tch = actual_touch[actual_touch['x']>=70]
            penbox_tch = actual_touch[(actual_touch['x']>=88.5) & (actual_touch['y']>=13.6) & (actual_touch['y']<=54.4)]

            fthd_rec = pass_rec[pass_rec['endX']>=70]
            penbox_rec = pass_rec[(pass_rec['endX']>=88.5) & (pass_rec['endY']>=13.6) & (pass_rec['endY']<=54.4)]

            pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2, pad_bottom=27)
            pitch.draw(ax=ax)

            ax.scatter(touch_df.y, touch_df.x, marker='o', s=30, c='None', edgecolor=col2, lw=2)
            if len(touch_df)>3:
                # Calculate mean point
                mean_point = np.mean(touch_df[['y', 'x']].values, axis=0)
                
                # Calculate distances from the mean point
                distances = np.linalg.norm(touch_df[['y', 'x']].values - mean_point[None, :], axis=1)
                
                # Compute the interquartile range (IQR)
                q1, q3 = np.percentile(distances, [20, 80])  # Middle 75%: 12.5th to 87.5th percentile
                iqr_mask = (distances >= q1) & (distances <= q3)
                
                # Filter points within the IQR
                points_within_iqr = touch_df[['y', 'x']].values[iqr_mask]
                
                # Check if we have enough points for a convex hull
                if len(points_within_iqr) >= 3:
                    hull = ConvexHull(points_within_iqr)
                    for simplex in hull.simplices:
                        ax.plot(points_within_iqr[simplex, 0], points_within_iqr[simplex, 1], color=col2, linestyle='--')
                    ax.fill(points_within_iqr[hull.vertices, 0], points_within_iqr[hull.vertices, 1], 
                            facecolor='none', edgecolor=col2, alpha=0.3, hatch='/////', zorder=1)
                else:
                    pass
            else:
                pass

            ax.scatter(pass_rec.endY, pass_rec.endX, marker='o', s=30, c='None', edgecolor=col1, lw=2)
            if len(touch_df)>4:
                # Calculate mean point
                mean_point = np.mean(pass_rec[['endY', 'endX']].values, axis=0)
                
                # Calculate distances from the mean point
                distances = np.linalg.norm(pass_rec[['endY', 'endX']].values - mean_point[None, :], axis=1)
                
                # Compute the interquartile range (IQR)
                q1, q3 = np.percentile(distances, [25, 75])  # Middle 75%: 12.5th to 87.5th percentile
                iqr_mask = (distances >= q1) & (distances <= q3)
                
                # Filter points within the IQR
                points_within_iqr = pass_rec[['endY', 'endX']].values[iqr_mask]
                
                # Check if we have enough points for a convex hull
                if len(points_within_iqr) >= 3:
                    hull = ConvexHull(points_within_iqr)
                    for simplex in hull.simplices:
                        ax.plot(points_within_iqr[simplex, 0], points_within_iqr[simplex, 1], color=col1, linestyle='--')
                    ax.fill(points_within_iqr[hull.vertices, 0], points_within_iqr[hull.vertices, 1], 
                            facecolor='none', edgecolor=col1, alpha=0.3, hatch='/////', zorder=1)
                else:
                    pass
            else:
                pass

            ax_text(34, 110, '<Touches> & <Pass Receiving> Points', fontsize=20, fontweight='bold', ha='center', va='center', 
                    highlight_textprops=[{'color':col2}, {'color':col1}])
            ax.text(34, -5, f'Total Touches: {len(actual_touch)} | at Final Third: {len(fthd_tch)} | at Penalty Box: {len(penbox_tch)}', color=col2, fontsize=13, ha='center', va='center')
            ax.text(34, -9, f'Total Pass Received: {len(pass_rec)} | at Final Third: {len(fthd_rec)} | at Penalty Box: {len(penbox_rec)}', color=col1, fontsize=13, ha='center', va='center')
            ax.text(34, -17, '*blue area = middle 75% touches area', color=col2, fontsize=13, fontstyle='italic', ha='center', va='center')
            ax.text(34, -21, '*red area = middle 75% pass receiving area', color=col1, fontsize=13, fontstyle='italic', ha='center', va='center')
            return
        

        def defensive_actions(ax, pname):
            # Viz Dfs:
            playerdf = df[df['name']==pname]
            tackles = playerdf[(playerdf['type']=='Tackle') & (playerdf['outcomeType']=='Successful')]
            tackleu = playerdf[(playerdf['type']=='Tackle') & (playerdf['outcomeType']=='Unsuccessful')]
            ballrec = playerdf[playerdf['type']=='BallRecovery']
            intercp = playerdf[playerdf['type']=='Interception']
            clearnc = playerdf[playerdf['type']=='Clearance']
            passbkl = playerdf[playerdf['type']=='BlockedPass']
            shotbkl = playerdf[playerdf['type']=='Save']
            chalnge = playerdf[playerdf['type']=='Challenge']
            aerialw = playerdf[(playerdf['type']=='Aerial') & (playerdf['outcomeType']=='Successful')]
            aerialu = playerdf[(playerdf['type']=='Aerial') & (playerdf['outcomeType']=='Unsuccessful')]

            # Pitch Plot
            pitch = VerticalPitch(pitch_type='uefa', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=2, pad_bottom=27)
            pitch.draw(ax=ax)

            # Scatter Plots
            sns.scatterplot(x=tackles.y, y=tackles.x, marker='X', s=300, color=col2, edgecolor=line_color, linewidth=1.5, alpha=0.8, ax=ax)
            sns.scatterplot(x=tackleu.y, y=tackleu.x, marker='X', s=300, color=col1, edgecolor=line_color, linewidth=1.5, alpha=0.8, ax=ax)
            pitch.scatter(ballrec.x, ballrec.y, marker='o', lw=1.5, s=300, c=col2, edgecolors=line_color, ax=ax, alpha=0.8)
            pitch.scatter(intercp.x, intercp.y, marker='*', lw=1.25, s=600, c=col2, edgecolors=line_color, ax=ax, alpha=0.8)
            pitch.scatter(clearnc.x, clearnc.y, marker='h', lw=1.5, s=400, c=col2, edgecolors=line_color, ax=ax, alpha=0.8)
            pitch.scatter(passbkl.x, passbkl.y, marker='s', lw=1.5, s=300, c=col2, edgecolors=line_color, ax=ax, alpha=0.8)
            pitch.scatter(shotbkl.x, shotbkl.y, marker='s', lw=1.5, s=300, c=col1, edgecolors=line_color, ax=ax, alpha=0.8)
            pitch.scatter(chalnge.x, chalnge.y, marker='+', lw=5, s=300, c=col1, edgecolors=line_color, ax=ax, alpha=0.8)
            pitch.scatter(aerialw.x, aerialw.y, marker='^', lw=1.5, s=300, c=col2, edgecolors=line_color, ax=ax, alpha=0.8)
            pitch.scatter(aerialu.x, aerialu.y, marker='^', lw=1.5, s=300, c=col1, edgecolors=line_color, ax=ax, alpha=0.8)

            # Stats
            sns.scatterplot(x=[65], y=[-5], marker='X', s=300, color=col2, edgecolor=line_color, linewidth=1.5, alpha=0.8, ax=ax)
            sns.scatterplot(x=[65], y=[-10], marker='X', s=300, color=col1, edgecolor=line_color, linewidth=1.5, alpha=0.8, ax=ax)
            pitch.scatter(-15, 65, marker='o', lw=1.5, s=300, c=col2, edgecolors=line_color, ax=ax, alpha=0.8)
            pitch.scatter(-20, 65, marker='*', lw=1.25, s=600, c=col2, edgecolors=line_color, ax=ax, alpha=0.8)
            pitch.scatter(-25, 65, marker='h', lw=1.5, s=400, c=col2, edgecolors=line_color, ax=ax, alpha=0.8)
            
            pitch.scatter(-5, 26, marker='s', lw=1.5, s=300, c=col2, edgecolors=line_color, ax=ax, alpha=0.8)
            pitch.scatter(-10, 26, marker='s', lw=1.5, s=300, c=col1, edgecolors=line_color, ax=ax, alpha=0.8)
            pitch.scatter(-15, 26, marker='+', lw=5, s=300, c=col1, edgecolors=line_color, ax=ax, alpha=0.8)
            pitch.scatter(-20, 26, marker='^', lw=1.5, s=300, c=col2, edgecolors=line_color, ax=ax, alpha=0.8)
            pitch.scatter(-25, 26, marker='^', lw=1.5, s=300, c=col1, edgecolors=line_color, ax=ax, alpha=0.8)

            ax.text(60, -5.5, f'Tackle (Succ.): {len(tackles)}', fontsize=12)
            ax.text(60, -10.5, f'Tackle (Unsucc.): {len(tackleu)}', fontsize=12)
            ax.text(60, -15.5, f'Ball Recoveries: {len(ballrec)}', fontsize=12)
            ax.text(60, -20.5, f'Interceptions: {len(intercp)}', fontsize=12)
            ax.text(60, -25.5, f'Clearance: {len(clearnc)}', fontsize=12)

            ax.text(21, -5.5, f'Passes Blocked: {len(passbkl)}', fontsize=12)
            ax.text(21, -10.5, f'Shots Blocked: {len(shotbkl)}', fontsize=12)
            ax.text(21, -15.5, f'Dribble Past: {len(chalnge)}', fontsize=12)
            ax.text(21, -20.5, f'Aerials Won: {len(aerialw)}', fontsize=12)
            ax.text(21, -25.5, f'Aerials Lost: {len(aerialu)}', fontsize=12)

            ax.text(34, 110, 'Defensive Actions', fontsize=20, fontweight='bold', ha='center', va='center')
            return
        

        def generate_player_dahsboard(pname):
            fig, axs = plt.subplots(1, 3, figsize=(27, 17), facecolor='#f5f5f5')
            
            # Generate individual plots
            offensive_actions(axs[0], pname)
            defensive_actions(axs[1], pname)
            pass_receiving_and_touchmap(axs[2], pname)
            fig.subplots_adjust(wspace=0.025)
            
            # Add text and images to the figure
            fig.text(0.14, 1.02, f'{pname}', fontsize=50, fontweight='bold', ha='left', va='center')
            fig.text(0.14, 0.97, f'in {hteamName} {hgoal_count} - {agoal_count} {ateamName}', 
                    fontsize=30, ha='left', va='center')
            
            st.pyplot(fig)
            

        player_list = df['name'].unique().tolist()
        pname = st.selectbox('Select Player', player_list, index=0)
        generate_player_dahsboard(pname)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    