import pandas as pd
import numpy as np
from sys import stdout
import os, glob
from scipy.signal import detrend
from scipy import arctan2
import math


def get_file_paths(path_to_csvs, day):

    # this is hard-coded as it is inherent in the ttu data
    groups = [1, 2, 3, 4]        
    
    # initialize a dictionary to keep track of filenames per "group"
    filepaths_dict = {group_int:list() for group_int in groups}

    # loop through all files that are for the desired day and save them as belonging to a certain group in the dictionary
    for file in sorted(glob.glob(os.path.join(path_to_csvs, "*201702{0}*.csv".format(day)))):
        # this is just a bunch of string manipulation to get the group number which avoids all the if statements
        group_int = int(os.path.split(file)[-1].split("TR_D")[-1].split(".csv")[0])
        filepaths_dict[group_int].append(file)

    return filepaths_dict

def get_file_headers(path_to_csvs):
                        
    # this is hard-coded as it is inherent in the ttu data
    groups = [1, 2, 3, 4]        
    
    # allocate space to keep heights available in the file and header names in the file, also organized per group
    height_feet_dict   = {}
    height_meters_dict = {}
    header_names_dict  = {}
    unique_heights     = {}

    i = -1    
    for group in groups:

        header_path   = os.path.join(path_to_csvs,"FT2_E04_C06 Channel List.xlsx")
        header_df     = pd.read_excel(header_path, index_col=[0], sheet_name="Group {0} Raw".format(group))
        ############## Issac's Fix #####################
        s = header_df.iloc[:,0]
        header_names  = s.values 
      
        ################################################
        #header_names  = header_df.VarName.values
        height_feet   = np.array([float(item.split('T')[0]) for item in header_names])

        height_feet_dict[group]   = height_feet.copy()
        height_meters_dict[group] = np.round(height_feet/3.2808,1)
        header_names_dict[group]  = header_names.copy()
        unique_heights[group]     = np.unique(height_meters_dict[group])

    # a vector (1-d numpy array) of all heights available in the four groups combined
    unique_heights_all = np.array(unique_heights[1].tolist() + unique_heights[2].tolist() + unique_heights[3].tolist() + unique_heights[4].tolist())        

    return height_feet_dict, height_meters_dict, header_names_dict, unique_heights, unique_heights_all

def get_boom_tower_angles(path_to_csvs):
    header_path = os.path.join(path_to_csvs,"2012-02-01 -- Anemometer Position Calculator.xlsx")
    sht_names = ["3","8","13", "33","55","155", "382","519","656"]
    boom_azm_deg = {}
    boom_vert_deg = {}
    ii = [0,1,2]

    RotMatrix = {}

    for name in sht_names:
        header_df = pd.read_excel(header_path, index_col=[0], sheet_name=name+"ft")

        boom_azm_deg[name] = header_df.iloc[12,11]
        boom_vert_deg[name] = header_df.iloc[21,11]

        U_rot = np.array(header_df.iloc[2,14:17])
        V_rot = np.array(header_df.iloc[3,14:17])
        W_rot = np.array(header_df.iloc[4,14:17])

        RotMatrix[name]= [U_rot, V_rot , W_rot]

    return RotMatrix, boom_azm_deg, boom_vert_deg

def csv_to_df(filepath, header_names, day=17):

    # Extract Time info from File Name
    item   = filepath.split("T")
    item_1 = item[4]
    item_2 = item_1.split("_")
    item_3 = item_2[0] 
    init_hour, init_minute = int(item_3[:2]), int(item_3[2:])

    # Create DF from filename
    df_new = pd.read_csv(filepath, header=None)
    df_new.columns = header_names

    # Create Time Index
    steps = len(df_new)
    init_timestamp = pd.Timestamp(year=2017, month=2, day=day, hour=init_hour, minute=init_minute).tz_localize("UTC")
    smart_index    = pd.date_range(start = init_timestamp, periods = steps,  freq = "20ms") 
    df_new         = df_new.set_index(smart_index)
    
    #cols_mph_to_mps = [s for s in df_new.columns if (("U" in s)or("V" in s)or("W" in s))]
    #df_new[cols_mph_to_mps] = df_new[cols_mph_to_mps]

    return df_new

def subset_df(df, string_to_keep="Sonic"):
    available_columns = df.columns
    varnames_to_keep  = [varname for varname in available_columns if string_to_keep in varname ]    
    subdf             = df[varnames_to_keep]
    return subdf

def df_to_df_dict(df):
    available_columns = df.columns    
    
    heights = np.unique([float(s.split('T')[0]) for s in available_columns])
    
    df_per_height = {}
    for height in heights:
        if height != 245.0: # Get rid of Faulty Data     
            columns_height = [s for s in available_columns if "{0:03d}".format(int(height)) in s]
            df_per_height[height] = df[columns_height]
            new_column_names = [s.split("Sonic ")[-1] for s in columns_height]
            new_column_names = [s if 'arm' not in s else s.split('-arm')[0] for s in new_column_names ]
            df_per_height[height].columns = new_column_names
        
    return df_per_height


def rotate_df_dict(df_dict,RotMatrix):
    
    df_dict_out = {}
    
    for key in df_dict.keys():
       
        key_strg = str(key).split(".")[0]
        dflink = df_dict[key]    
        
        u0 = dflink.U
        v0 = dflink.V
        w0 = dflink.W

        u1 = u0*np.cos(np.radians(RotMatrix[key_strg][0][0])) + v0*np.cos(np.radians(RotMatrix[key_strg][0][1]))+ w0*np.cos(np.radians(RotMatrix[key_strg][0][2]))
        v1 = u0*np.cos(np.radians(RotMatrix[key_strg][1][0])) + v0*np.cos(np.radians(RotMatrix[key_strg][1][1]))+ w0*np.cos(np.radians(RotMatrix[key_strg][1][2]))
        w1 = u0*np.cos(np.radians(RotMatrix[key_strg][2][0])) + v0*np.cos(np.radians(RotMatrix[key_strg][2][1]))+ w0*np.cos(np.radians(RotMatrix[key_strg][2][2]))
        
        sonic_yaw   = np.arctan2(v1.mean(),u1.mean())
        
        u2          = u1*np.cos(sonic_yaw)+v1*np.sin(sonic_yaw)
        v2          = -u1*np.sin(sonic_yaw)+v1*np.cos(sonic_yaw)
        w2         = w1.copy()
        
        sonic_pitch = np.arctan2(w2.mean(),u2.mean())
        
        u3          = u2*np.cos(sonic_pitch)+w2*np.sin(sonic_pitch)
        v3          = v2.copy()
        w3          = -u2*np.sin(sonic_pitch)+w2*np.cos(sonic_pitch)
        
        theta = np.degrees(np.arctan2(v3,u3))
        
        dfout = pd.DataFrame(index=dflink.index,columns=dflink.columns)
        dfout.U = u3.copy()
        dfout.V = v3.copy()
        dfout.W = w3.copy()
        dfout.Temperature = dflink.Temperature
        dfout.insert(4,"Theta",theta,True)
 
    
        df_dict_out[key] = dfout.copy()
    
    return df_dict_out

def compute_perturbations(df_dict, detrend_series=True):
    
    df_dict_out = {}
    
    for key in df_dict.keys():
       
        dflink = df_dict[key].copy()       
        df_dict_out[key] = dflink.copy()   
    
        if detrend_series:                
            for col in dflink.columns:
                series_no_na     = dflink[col].dropna().copy()
                detrended_series = pd.Series(index=series_no_na.index, data=detrend(series_no_na.values))
                dflink[col]      = detrended_series.copy()    
        
        uprime = dflink.U-dflink.U.mean()
        vprime = dflink.V-dflink.V.mean()
        wprime = dflink.W-dflink.W.mean()
        


#              uu = np.sqrt((df_sonic[u_h] - df_sonic[u_h].mean())**2)
    
#     ti_s = (uu/ df_sonic[u_h].mean()).mean()
        
        df_dict_out[key]["uprime"] = uprime
        df_dict_out[key]["vprime"] = vprime
        df_dict_out[key]["wprime"] = wprime

        #df_dict_out[key]["TI"] = ti
        
    return df_dict_out


def circmean(alpha,axis=None):
    alpha[alpha<0.0] = alpha[alpha<0.0] + 360.0
    alpha      = np.radians(alpha)
    mean_angle = np.arctan2(np.mean(np.sin(alpha),axis),np.mean(np.cos(alpha),axis))
    mean_angle = np.degrees(mean_angle)
    mean_angle = mean_angle + 360.0 if mean_angle<0 else mean_angle
    
    return mean_angle

def myround(x, base=5):
    return int(base * round(float(x)/base))

######################################################################
def get_sonic_met_dat(path_to_csvs, start_time ,end_time):
    
    print("------------------")
    print("Starting Master Function")
    # def get_that_met_dat(path_to_csvs,)
    groups = [1,2,3,4]
    year   = 2017 
    month  = 2
    day    = 17

    height_feet_dict, height_meters_dict, header_names_dict, unique_heights, unique_heights_all = get_file_headers(path_to_csvs)

    turbSim_TS  = {height:{} for height in unique_heights_all}

    # INIT TIME ##
    hours_i = int(start_time.split(':')[0])
    minute_i = int(start_time.split(':')[1])
    seconds_i = int(start_time.split(':')[2])

    if minute_i > 30:
        min_head_i = 30
    else:
        min_head_i = 0

    if hours_i > 9:
        if min_head_i ==0:
            init_timestamp = '2017-02-17 {0}:00:00' .format(hours_i)
        else:
            init_timestamp = '2017-02-17 {0}:{1}:00'.format(hours_i,min_head_i)

    if hours_i <= 9:
        if min_head_i ==0:
            init_timestamp = '2017-02-17 0{0}:00:00' .format(hours_i)
        else:
            init_timestamp = '2017-02-17 0{0}:{1}:00'.format(hours_i,min_head_i)        

    # Final Time ##
    hours_f = int(end_time.split(':')[0])
    minute_f = int(end_time.split(':')[1])
    seconds_f = int(end_time.split(':')[2])


    if minute_f > 30:
        min_head_f = 30
    else:
        min_head_f = 0

    if hours_f > 9:
        if min_head_f ==0:
            final_timestamp = '2017-02-17 {0}:00:00' .format(hours_f)
        else:
            final_timestamp = '2017-02-17 {0}:{1}:00'.format(hours_f,min_head_f)

    if hours_f <= 9:
        if min_head_f ==0:
            final_timestamp = '2017-02-17 0{0}:00:00' .format(hours_f)
        else:
            final_timestamp = '2017-02-17 0{0}:{1}:00'.format(hours_f,min_head_f)  

    indicies = int(2*(hours_f - hours_i)+(min_head_f - min_head_i)/30)

    filepaths_dict = get_file_paths(path_to_csvs, day=17)

    # Create File Identifier string  
    strg = init_timestamp.split(' ')[1]
    strg_1 = strg.split(':')[0]
    strg_2 = strg.split(':')[1]
    strg_3 = 'T'+strg_1 + strg_2

    needed_files = {}

    for group in groups:
        i = 0
        for name in filepaths_dict[group]:
            tF = strg_3 in name
            if tF == True:
                start_ind = i
            i = i + 1
        end_ind = start_ind+ indicies 

        if start_ind == end_ind:
            needed_files[group] = filepaths_dict[group][start_ind:start_ind+1]
        else:
            needed_files[group] = filepaths_dict[group][start_ind:end_ind]


    # Get Boom Tower Angles
    cosAngles, boom_azm_deg, boom_vert_deg = get_boom_tower_angles(path_to_csvs) 

    init_time = pd.to_datetime('2017-02-17 '  + start_time).tz_localize('UTC')
    final_time = pd.to_datetime('2017-02-17 '  + end_time).tz_localize('UTC')

    for group in groups:
        print('group {0} out of 4'.format(group))
        for file in needed_files[group]:
            df           = csv_to_df(file, header_names_dict[group])
            subdf        = subset_df(df)
            df_dict      = df_to_df_dict(subdf)
            df_dict_rot  = rotate_df_dict(df_dict,cosAngles)

            df_dict_pert = compute_perturbations(df_dict_rot)

            for height in df_dict_pert.keys():
                ############## BAD
                dflink        = df_dict_pert[height].copy()
                height        = np.round(height/3.2808,1)


                for idx in dflink.index:
                    if idx >= init_time and idx <= final_time:

                        turbSim_TS[height][idx,"U"] = dflink.loc[idx,"U"] 
                        turbSim_TS[height][idx,"V"] = dflink.loc[idx,"V"]
                        turbSim_TS[height][idx,"W"] = dflink.loc[idx,"W"] 
                        turbSim_TS[height][idx,"Theta"] = dflink.loc[idx,"Theta"]


    # Remove Faulty Data
    try:
        del turbSim_TS[74.7]
    except KeyError:
        print("Key 'testing' not found")

    #################
    print("Creating Final DataFrame")
    TS_series ={}
    for z in turbSim_TS.keys():
        TS_tmp = pd.Series(turbSim_TS[z]).sort_index() 

        TS_series[z] = TS_tmp.copy() 

    TurbSim_df = pd.DataFrame(TS_series)   
    TurbSim_df = TurbSim_df.unstack()

    print("DONE!")
    print("------------------")

    
    
    return TurbSim_df