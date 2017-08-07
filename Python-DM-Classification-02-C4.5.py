############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Data Mining
# Lesson: Decision Trees - C4.5

# Citation: 
# PEREIRA, V. (2018). Project: C4.5, File: Python-DM-Classification-02-C4.5.py, GitHub repository: <https://github.com/Valdecy/C4.5>

############################################################################

# Installing Required Libraries
import pandas as pd
import numpy  as np

# Function: is_number
def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def info_gain_ratio(target, feature = [], uniques = []):
    entropy = 0
    denominator_1 = feature.count()
    data = pd.concat([pd.DataFrame(target.values.reshape((target.shape[0], 1))), feature], axis = 1)
    for entp in range(0, len(np.unique(target))):
        numerator_1 = data.iloc[:,0][(data.iloc[:,0] == np.unique(target)[entp])].count()
        if numerator_1 > 0:
            entropy = entropy - (numerator_1/denominator_1)* np.log2((numerator_1/denominator_1))
    info_gain = float(entropy)
    info_gain_r = 0
    intrinsic_v = 0
    for word in range(0, len(uniques)):
        denominator_2 = feature[(feature == uniques[word])].count()
        if denominator_2[0] > 0:
            intrinsic_v = intrinsic_v - (denominator_2/denominator_1)* np.log2((denominator_2/denominator_1))
        for lbl in range(0, len(np.unique(target))):
            numerator_2 = data.iloc[:,0][(data.iloc[:,0] == np.unique(target)[lbl]) & (data.iloc[:,1]  == uniques[word])].count()
            if numerator_2 > 0:
                info_gain = info_gain + (denominator_2/denominator_1)*(numerator_2/denominator_2)* np.log2((numerator_2/denominator_2))
    if intrinsic_v[0] > 0:
        info_gain_r = info_gain/intrinsic_v
    return float(info_gain_r)

def split_me(target, feature, split):
    column = pd.DataFrame(feature.values.reshape((feature.shape[0], 1)))
    result = pd.DataFrame(feature.values.reshape((feature.shape[0], 1)))
    for fill in range(0, len(feature)):
        column.iloc[fill,0] = feature.iloc[fill]
        result.iloc[fill,0] = feature.iloc[fill]
    lower = "<=" + str(split)
    upper = "> " + str(split)
    for convert in range(0, len(feature)):
        if float(feature.iloc[convert]) <= float(split):
            result.iloc[convert,0] = lower
        else:
            result.iloc[convert,0] = upper
    binary_split = []
    binary_split = [lower, upper]
    return result, binary_split

# Function: dt_c45
def dt_c45(Xdata, ydata):
    
    ################     Part 1 - Preprocessing    #############################
    # Preprocessing - Creating Dataframe
    name = ydata.name
    ydata = pd.DataFrame(ydata.values.reshape((ydata.shape[0], 1)))
    dataset = pd.concat([ydata, Xdata], axis = 1)
    
    for j in range(0, dataset.shape[1]):
        if dataset.iloc[:,j].dtype == "bool":
            dataset.iloc[:,j] = dataset.iloc[:, j].astype(str)

    # Preprocessing - Unique Words List
    unique = []
    uniqueWords = []
    for j in range(0, dataset.shape[1]): 
        for i in range(0, dataset.shape[0]):
            token = dataset.iloc[i, j]
            if not token in unique:
                unique.append(token)
        uniqueWords.append(unique)
        unique = []  
    
    # Preprocessing - Label Matrix
    label = np.array(uniqueWords[0])
    label = label.reshape(1, len(uniqueWords[0]))
    
    ################    Part 2 - Initialization    #############################
    # C4.5 - Initializing Variables
    i = 0
    branch = [None]*1
    branch[0] = dataset
    gain_ratio = np.empty([1, branch[i].shape[1]])
    lower = "0"
    root_index = 0
    rule = [None]*1
    rule[0] = "IF "
    skip_update = False
    stop = 2
    upper = "1"
    
    ################     Part 3 - C4.5 Algorithm    #############################
    # C4.5 - Algorithm
    while (i < stop):
        gain_ratio.fill(0)
        for element in range(1, branch[i].shape[1]):
            if len(branch[i]) == 0:
                skip_update = True 
                break
            if len(np.unique(branch[i][0])) == 1 or len(branch[i]) == 1:
                 if "." not in rule[i]:
                     rule[i] = rule[i] + " THEN " + name + " = " + branch[i].iloc[0, 0] + "."
                     rule[i] = rule[i].replace(" AND  THEN ", " THEN ")
                 skip_update = True
                 break
            if is_number(dataset.iloc[0, element]):
                gain_ratio[0, element] = 0.0
                value = np.sort(branch[i].iloc[:, element].unique())
                if branch[i][(branch[i].iloc[:, element] == value[0])].count()[0] > 1:
                    start = 0
                    finish = len(branch[i].iloc[:, element].unique()) - 2
                else:
                    start = 1
                    finish = len(branch[i].iloc[:, element].unique()) - 2
                if len(branch[i]) == 2 or len(value) == 1 or len(value) == 2:
                    start = 0
                    finish = 1
                if len(value) == 3:
                    start = 0
                    finish = 2
                for bin_split in range(start, finish):
                    bin_sample = split_me(target = branch[i].iloc[:, 0], feature = branch[i].iloc[:, element], split = value[bin_split])
                    igr = info_gain_ratio(target = branch[i].iloc[:, 0], feature = bin_sample[0], uniques = bin_sample[1])
                    if igr > float(gain_ratio[0, element]):
                        gain_ratio[0, element] = igr
                        uniqueWords[element] = bin_sample[1]
                continue
            if is_number(dataset.iloc[0, element]) == False:
                gain_ratio[0, element] = 0.0
                igr = info_gain_ratio(target = branch[i].iloc[:, 0], feature =  pd.DataFrame(branch[i].iloc[:, element].values.reshape((branch[i].iloc[:, element].shape[0], 1))), uniques = uniqueWords[element])
                gain_ratio[0, element] = igr
            
        if skip_update == False:
            root_index = np.argmax(gain_ratio)
            rule[i] = rule[i] + list(branch[i])[root_index]
            
            for word in range(0, len(uniqueWords[root_index])):
                uw = uniqueWords[root_index][word].replace("<=", "")
                uw = uw.replace("> ", "")
                lower = "<=" + uw
                upper = "> " + uw
                if uniqueWords[root_index][word] == lower:
                    branch.append(branch[i][branch[i].iloc[:, root_index] <= float(uw)])
                elif uniqueWords[root_index][word] == upper:
                    branch.append(branch[i][branch[i].iloc[:, root_index]  > float(uw)])
                else:
                    branch.append(branch[i][branch[i].iloc[:, root_index] == uniqueWords[root_index][word]])
                
                rule.append(rule[i] + " = " + "{" + uniqueWords[root_index][word] + "}")
            
            for logic_connection in range(1, len(rule)):
                if len(np.unique(branch[i][0])) != 1 and rule[logic_connection].endswith(" AND ") == False  and rule[logic_connection].endswith("}") == True:
                    rule[logic_connection] = rule[logic_connection] + " AND "
        skip_update = False
        i = i + 1
        print("iteration: ", i)
        stop = len(rule)
    
    for i in range(len(rule) - 1, -1, -1):
        if rule[i].endswith(".") == False:
            del rule[i]
    
    rule.append("1) Total Number of Rules: " + str(len(rule)))
    rule.append("2) When No Rule Applies: " + name + " = " + dataset.agg(lambda x:x.value_counts().index[0])[0])
    
    return rule

    ############### End of Function ##############

######################## Part 4 - Usage ####################################

df = pd.read_csv('Python-DM-Classification-02-C4.5.csv', sep = ';')

X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

dt_c45(Xdata = X, ydata = y)

########################## End of Code #####################################
