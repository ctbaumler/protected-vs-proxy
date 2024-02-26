import numpy as np
import pandas as pd
import sklearn.preprocessing, sklearn.decomposition, sklearn.linear_model, sklearn.pipeline, sklearn.metrics, sklearn.model_selection, sklearn.impute, sklearn.ensemble
from sklearn_pandas import DataFrameMapper
import scipy.stats as stats
from scipy.stats import pearsonr
import csv, os
from PIL import Image

from itertools import product

import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')

STUDY_PATH = "study"
DATA_PATH = '../../data/prosper'



FLIP = -1
BACKWARDS = True
TUTORIAL_IDX = 420275
GEN_ALL = True


def create_or_clear_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        for file_or_dir in os.listdir(directory):
            file_or_dir_path = os.path.join(directory, file_or_dir)
            if os.path.isfile(file_or_dir_path):
                os.remove(file_or_dir_path)
            elif os.path.isdir(file_or_dir_path):
                for root, dirs, files in os.walk(file_or_dir_path, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(file_or_dir_path)


def group_loan_status(s):
    if s == 'Completed' or s == 'FinalPaymentInProgress':
        return 'Completed'
    if s == 'Current' or s == 'Cancelled':
        return 'Other'
    if s.startswith('Past Due') or s == 'Defaulted' or s == 'Chargedoff':
        return 'Late'
    assert False, s

def sample_corr(row, dist, labels, sensitive_feature):
    return np.random.choice(labels, 1, p=dist[row[sensitive_feature]])[0]


def load_and_clean(interp_cols):
    


    prosper = pd.read_csv(DATA_PATH+'/prosperLoanData.csv')
    prosper = prosper.drop_duplicates(subset=['ListingNumber'], keep='first').reset_index(drop=True)
    prosper = prosper.set_index('ListingNumber').sort_values(by=['ListingNumber'])




    prosper['GroupedLoanStatus'] = prosper['LoanStatus'].apply(group_loan_status)
    prosper['AfterJuly2009'] = prosper['ListingCreationDate'].apply(lambda s: int(s[:4]) > 2009 or (int(s[:4]) == 2009 and int(s[5:7]) > 7))

    # get only the rows where the loan is either Completed or Late, and after July 2009
    prosper_cd = prosper[((prosper['GroupedLoanStatus'] == 'Completed') | (prosper['GroupedLoanStatus'] == 'Late')) & prosper['AfterJuly2009']]


    prosper_cd = prosper_cd[interp_cols+['GroupedLoanStatus']].dropna()

    return prosper_cd


def generate_cols(condition, prosper, cor_labels, cor_dist, maj_percent, tutorial_idx, interp_cols, categorical_columns, sensitive_labels= ["m","f"], sensitive_feature = "Gender", cor_feature = "University"):

    completed = prosper[prosper.GroupedLoanStatus == "Completed"].copy()
    late = prosper[prosper.GroupedLoanStatus == "Late"].copy()

    maj = maj_percent
    mi =1-maj


    props_dict = {"Late":{'m':mi, 'f':maj}, "Completed":{"m":maj, "f":mi}}


    dist = [props_dict['Late'][x] for x in sensitive_labels]
    late[sensitive_feature] = np.random.choice(sensitive_labels, len(late), p=dist)

    
    completed_by_sen = []
    


    for i in range(len(sensitive_labels)):
        # based on the proportion that we're looking for and the number of late examples, pick a sample of completed examples for each gender
        l = sensitive_labels[i]
        completed_by_sen.append( completed.sample(int(len(late[late[sensitive_feature] == l])*(1/props_dict['Late'][l])*props_dict['Completed'][l])))
        completed = completed.drop(completed_by_sen[i].index)
        
        completed_by_sen[i][sensitive_feature] = [l]*len(completed_by_sen[i])
    
    tutorial_eg = completed[completed.index == tutorial_idx].copy()
    tutorial_eg["Gender"] = ["f" for _ in range(len(tutorial_eg))]



    prosper_extra = pd.concat(completed_by_sen+[late,tutorial_eg])
    prosper_extra = prosper_extra.sort_values(by=['ListingNumber'])

    


    prosper_extra[cor_feature] = prosper_extra.apply(lambda x: sample_corr(x, cor_dist, cor_labels, sensitive_feature), axis=1)
    prosper_extra.at[tutorial_idx, cor_feature] = 'Scripps College'
    prosper_extra.at[tutorial_idx, sensitive_feature] = 'f'


    prosper_biased = prosper_extra.copy()


    tutorial_eg = prosper_biased[prosper_biased.index == tutorial_idx]


    train, test = sklearn.model_selection.train_test_split(prosper_biased[prosper_biased.index != tutorial_idx], test_size=.4)

    train = train.sort_index()

    test = pd.concat([test, tutorial_eg])
    test = test.sort_index()

    

    binarizer = sklearn.preprocessing.LabelBinarizer()

    target = interp_cols+[sensitive_feature, cor_feature]

    label = 'GroupedLoanStatus'

    mapper = DataFrameMapper(
        [(label, binarizer)] + 
        [([col], sklearn.preprocessing.OneHotEncoder() ) for col in categorical_columns if col in target] +
        [([col], sklearn.preprocessing.StandardScaler()) for col in target if col not in categorical_columns] +
        [],
        sparse=False,
        df_out=True
    )


    

    df = mapper.fit_transform(prosper_biased)
    df['ListingNumber'] = list(prosper_biased.index)
    df = df.set_index('ListingNumber')

    #convert back to old naming scheme

    order_dict = mapper_ordering(mapper, categorical_columns)


    column_names = {}
    for col in df.columns.to_list():
        col_split = col.split("_")
        if col_split[0] in order_dict:
            column_names[col] = col_split[0] + "_x0_" + str(order_dict[col_split[0]][int(col_split[1])])

    df = df.rename(columns=column_names)




    train_X = df[df.index.isin(train.index)].copy()
    train_Y = train_X[['GroupedLoanStatus']]
    train_X = train_X.drop(columns = ['GroupedLoanStatus'])


    test_X = df[df.index.isin(test.index)].copy()
    test_Y = test_X[['GroupedLoanStatus']]
    test_X = test_X.drop(columns = ['GroupedLoanStatus'])



    test_full = df[df.index.isin(test.index)].copy()




    train_X.to_csv("_".join([STUDY_PATH+"/dfs/train_X",condition])+".csv")
    train_Y.to_csv("_".join([STUDY_PATH+"/dfs/train_y",condition])+".csv")
    test_X.to_csv("_".join([STUDY_PATH+"/dfs/test_X",condition])+".csv")
    test_Y.to_csv("_".join([STUDY_PATH+"/dfs/test_Y",condition])+".csv")
    prosper_biased.to_csv("_".join([STUDY_PATH+"/dfs/prosper_biased",condition])+".csv")
    df.to_csv("_".join([STUDY_PATH+"/dfs/df",condition])+".csv")

    test_full = df[df.index.isin(test.index)].copy()


    cor_cols = [x for x in list(train_X.columns) if x[:len(cor_feature)+3] == cor_feature+"_x0"]
    sen_cols = [x for x in list(train_X.columns) if x[:len(sensitive_feature)+3] == sensitive_feature+"_x0"]

    
    bias_type = condition.split("_")[0]



    if bias_type == "protected":
        drop = cor_cols
    elif bias_type == "proxy":
        drop = sen_cols
    else:
        drop = sen_cols+cor_cols

    train_X = train_X.drop(columns = drop)
    test_X = test_X.drop(columns = drop)
    # prosper_Y = train_Y

    prosper_sen = test[[sensitive_feature, 'GroupedLoanStatus']].copy()

    

    return train_X, train_Y, test_X, test_Y, prosper_biased, prosper_sen, mapper, drop, test_full, train #train, test, prosper_biased

def test_feature_name_equiv(feat,cand):
    return (len(feat) == len(cand) and feat == cand) or \
    ("_" in cand and len(feat)<len(cand) and cand[len(feat)] == "_" and cand[:len(feat)] == feat) \
    or (" " in cand and len(feat)<len(cand) and cand[len(feat)] == " " and cand[:len(feat)] == feat)

def in_feats(feat, cur_feats):
    # The binarizer makes some changes to the feature names. This checks that a given binarized feature is in the set of features we're curently considering
    for f in cur_feats:
        if test_feature_name_equiv(f, feat):
            return True
    return False

def mapper_ordering(mapper, categorical_columns):
    d = {}

    for x in mapper.get_params()['features']:
        if x[0][0] in categorical_columns:
            d[x[0][0]] = {a:b for a,b in enumerate(x[1].categories_[0])}

    return d

def get_label_name(feat, key, train, orig_egs):
    abbrev_to_us_state ={'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia', 'AS': 'American Samoa', 'GU': 'Guam', 'MP': 'Northern Mariana Islands', 'PR': 'Puerto Rico', 'UM': 'United States Minor Outlying Islands', 'VI': 'U.S. Virgin Islands'}

    prosper_listing_category = {0: 'Not Available', 1: 'Debt Consolidation', 2: 'Home Improvement', 3: 'Business', 4: 'Personal Loan', 5: 'Student Use', 6: 'Auto', 7: 'Other', 8: 'Baby/Adoption', 9: 'Boat', 10: 'Cosmetic Procedure', 11: 'Engagement Ring', 12: 'Green Loans', 13: 'Household Expenses', 14: 'Large Purchases', 15: 'Medical/Dental', 16: 'Motorcycle', 17: 'RV', 18: 'Taxes', 19: 'Vacation', 20: 'Wedding Loans'}



    if feat == "Overall Prediction":
        value = str(list(orig_egs[orig_egs.index==key]['pred'])[0])
    elif "_x0" in feat:
        feat = feat[:feat.index("_")]
        value = list(orig_egs[orig_egs.index==key][feat])[0]
        p = len(train[train[feat]==value])/len(train)
    
        if "ListingCategory" in feat:
            feat = "ListingCategory"
            value = prosper_listing_category[int(value)]
        elif "BorrowerState" in feat:
            value = abbrev_to_us_state[value]
            
        p = round(p*100)
        if p ==0:
            value = str(value) + " (<1 percent)"
        else:
            value = str(value) + " ("+str(p)+" percent)"
    else:
        value = list(orig_egs[orig_egs.index==key][feat])[0]
        
        
        p = round(stats.percentileofscore(list(train[feat]), value))
        
        value = str(value) + " ("+str(p)+" percentile)"
        
        
        if "LoanOriginalAmount" in feat:
            value = "$"+value
            
    return feat, value

def part_pred(eg, bias_type, coefs, sensitive_feature, cor_feature, feats):

    if bias_type=="protected":
        cur_feats = [sensitive_feature] + feats[:-1]
    elif bias_type=="proxy":
        cur_feats = [cor_feature] + feats[:-1]
    else:
        cur_feats = feats
    eg_dict = {}
    for x in eg.items():
        eg_dict[x[0]] = list(x[1])[0]
    eg_used = {f:v for f,v in eg_dict.items() if ("_x0" not in f) or (v == 1)}
    
    bars = {f:v*eg_used[f] for f,v in coefs.items() if f in eg_used and in_feats(f, cur_feats)}
    return sum(bars.values())

def train_model(train_X, train_Y, test_X, test_Y, condition, prosper_sen, mapper, feats, sensitive_feature="Gender", cor_feature="University"):
    bias_type = condition.split("_")[0]


    estimator = sklearn.linear_model.LogisticRegression(solver='liblinear', penalty='l1')
    estimator.fit(train_X, train_Y)

    feat_names = list(train_X.columns)

    sorted_feats = sorted(zip(estimator.coef_[0], feat_names))

    for val,feat in sorted_feats: 
        if  abs(val) > 1e-1: 
            print(f"{val:-2.3}\t{feat}")
    
    coef = estimator.coef_[0]
    # test_X = test_X.drop(columns = drop_cols[bias_type]).sort_values(by=['ListingNumber'])
    coefs = {n:v for n,v in zip(test_X.columns, coef)}
    if coefs['LoanOriginalAmount'] < 0:
        # We already know that loan original amount should be contribute positively to a "complete" prediction. Here, we double-check that the direction of positive/neg prediction being complete/late hasn't gotten flipped

        coef *= FLIP # since the predictions are backwards of the way I want them to go
        coefs = {n:v for n,v in zip(test_X.columns, coef)}

    prosper_sen["tot"] = prosper_sen.apply(lambda x: part_pred(test_X[test_X.index == x.name], bias_type, coefs, sensitive_feature, cor_feature, feats) ,axis=1)
    prosper_sen[bias_type] = prosper_sen.apply(lambda x: int(x["tot"]>0) ,axis=1)



    model = estimator


    prosper_sen['answer']=test_Y




    positive = list(mapper.get_params()['features'][0][1].classes_).index('Completed')
    if BACKWARDS:
        # again, making sure that positive is "comleted" not "late"
        positive = 1- positive




    sensitive_labels = list(mapper.get_params()['features'][6][1].categories_[0])


    grp1 = len(prosper_sen[(prosper_sen[sensitive_feature] == sensitive_labels[0]) & (prosper_sen[bias_type] == positive )])/ len(prosper_sen[(prosper_sen[sensitive_feature] == sensitive_labels[0])])
    grp2 = len(prosper_sen[(prosper_sen[sensitive_feature] == sensitive_labels[1]) & (prosper_sen[bias_type] == positive )])/ len(prosper_sen[(prosper_sen[sensitive_feature] == sensitive_labels[1])])

    

    print(bias_type, grp1/grp2, grp1/grp2 < .8)

    # numbers["dis_"+bias_type] = grp1/grp2








    print(bias_type)
    for gender in sensitive_labels:
        for outcome in [positive, 1-positive]:
            print(gender, outcome, len(prosper_sen[ (prosper_sen.Gender == gender) & (prosper_sen[bias_type]==outcome)]))



    print(len(prosper_sen[(prosper_sen.answer ==prosper_sen[bias_type])])/len(prosper_sen))



    return model, coefs, positive


def graph(condition, x_lims, eg, explanation, orig_egs, sensitive_feature, cor_feature, feats, coefs, colors, train, save=True, just_extremes=False, tutorial=False):
    bias_type = condition.split("_")[0]


    fig, ax = plt.subplots()
    eg_dict = {}



    for x in eg.items():
        eg_dict[x[0]] = list(x[1])[0]

    key = eg.index[0]
    pred = list(eg['pred'])[0]


    gender = orig_egs.Gender[key]
    univ = "".join(orig_egs.University[key].split(" "))


    eg_used = {f:v for f,v in eg_dict.items() if ("_x0" not in f) or (v == 1)}

    if bias_type=="protected":
        cur_feats = [sensitive_feature] + feats[:-1]
    elif bias_type=="proxy":
        cur_feats = [cor_feature] + feats[:-1]
    else:
        cur_feats = feats


    bars = {f:v*eg_used[f] for f,v in coefs.items() if f in eg_used and in_feats(f, cur_feats)}



    x = list(bars.values())

    
    if just_extremes:
        return max(x), min(x)





    y = [get_label_name(feat,key, train, orig_egs) for feat in bars.keys()]
    y_1= [tup[0] for tup in y]
    y_2= [tup[1] for tup in y]

    barlist = ax.barh(y_1, x)
    for j in range(len(x)):
        if x[j] < 0 :
            barlist[j].set_color(colors["neg"])
        else:
            barlist[j].set_color(colors["pos"])
    ax.axvline(0,color="black")
    

    ax2 = ax.secondary_yaxis('right')
    
    ax2.set_yticklabels([""]+y_2)

    ax.set_xlim(x_lims[0], x_lims[1])

    
    ax.annotate("More likely to complete", xy=(.6,1.1), xytext=(0,0), xycoords='axes fraction', textcoords = "offset points")
    ax.annotate("", xy=(1,1.05), xytext=(.5,1.05), xycoords='axes fraction', arrowprops=dict(arrowstyle="->"))
    ax.annotate("More likely to be late", xy=(.1,1.1), xytext=(0,0), xycoords='axes fraction', textcoords = "offset points")
    ax.annotate("", xy=(0,1.05), xytext=(.5,1.05), xycoords='axes fraction', arrowprops=dict(arrowstyle="->"))

    name = "_".join([str(key),gender,univ,pred,condition, 'e'])+".png"
    plt.draw()

    name = "_".join([str(key),gender,univ,pred, condition, 'e'])+".png"
    if save and max(x) <= x_lims[1]:
        if tutorial:
            name = "TUTORIAL_"+name
            plt.savefig(STUDY_PATH+"/examples/tutorial/"+name, bbox_inches = "tight")
            plt.savefig(STUDY_PATH+"/examples/tutorial/"+name[:-3]+"pdf", bbox_inches = "tight")
        else:
            plt.savefig(STUDY_PATH+"/examples/main/"+name, bbox_inches = "tight")
    


    
        
    if explanation == "ne":
        ax.set_yticklabels(y_1,ha='left')
        y_2 = [":   "+val for val in y_2]
        ax2.set_yticklabels([""]+y_2)
            
        yax = ax.get_yaxis()
        # find the maximum width of the label on the major ticks
        pad = max(T.label.get_window_extent().width for T in yax.majorTicks)

        yax.set_tick_params(pad=pad)
        
        plt.savefig("temp.png", bbox_inches = "tight")
        
        im1 = Image.open("temp.png")
        width, height = im1.size

        im4 = im1.crop((719, 60, width, height-50))
        im3 = im1.crop((0, 60, 160, height-50))

        im3_size = im3.size
        im4_size = im4.size
        new_image = Image.new('RGB',(im3_size[0]+im4_size[0], im4_size[1]), (250,250,250))
        new_image.paste(im3,(0,0))
        new_image.paste(im4,(im3_size[0],0))
        new_image.show()
        name = "_".join([str(key),gender,univ,pred,condition, 'ne'])+".png"
        if tutorial:
            name = "TUTORIAL_"+name
            new_image.save(STUDY_PATH+"/examples/tutorial/"+name)
            plt.savefig(STUDY_PATH+"/examples/tutorial/"+name[:-3]+"pdf", bbox_inches = "tight")
        else:
            new_image.save(STUDY_PATH+"/examples/main/"+name)

        
    return max(x), min(x)
        
    
def graph_all(condition, egs, orig_egs, sensitive_feature, cor_feature, feats, coefs, colors, train, x_lims = (-1,1), save=True, just_extremes=False, tutorial=False, explanation="ne"):
    # change explanation to "e" if you don't want to also generate no explanations
    bias_type = condition.split("_")[0]


    max_x = 0
    min_x = 0
    first = True


    for i in egs.index:
        plt.clf()
        plt.cla()
        plt.close("all")
        





        eg = egs[egs.index == i]
        
        if first:
            #condition, x_lims, eg, explanation, orig_egs, sensitive_feature, cor_feature, feats, coefs, colors, train
            new_max, new_min = graph(condition, x_lims, eg, explanation, orig_egs, sensitive_feature, cor_feature, feats, coefs, colors, train, save, just_extremes, tutorial)
            first = False
        else:
            new_max, new_min = graph(condition, x_lims, eg, "e", orig_egs, sensitive_feature, cor_feature, feats, coefs, colors, train, save, just_extremes, tutorial)
        
        max_x = max(max_x, new_max)
        min_x = min(min_x, new_min)
        
            
    return max_x, min_x



def generate_viz(condition, coefs, test_full, prosper_sen, positive, sensitive_labels, maj_percent, sensitive_feature, cor_feature, feats, prosper_biased, universities, TUTORIAL_IDX, drop, train):
    bias_type = condition.split("_")[0]

    maj = maj_percent
    mi =1-maj

    num_groups=len(sensitive_labels)

    colors = {"pos": "#377eb8", "neg" : "#e41a1c"}

    props = [["Late", 'm', mi],
            ["Late", 'f', maj],
            ["Completed", 'm', maj],
            ["Completed", 'f', mi]]


    props_dict = {"Late":{'m':mi, 'f':maj}, "Completed":{"m":maj, "f":mi}}


    num_egs = 2000



    max_x, min_x = 3, -3 #starting max and min of graphs


    



    df = test_full.copy()
    df['pred'] = prosper_sen[bias_type]
    df.pred=df.apply(lambda x: "Completed" if x['pred']==positive else "Late", axis=1)
    df.GroupedLoanStatus=df.apply(lambda x: "Completed" if x['GroupedLoanStatus']==positive else "Late", axis=1)

    tutorial_eg = df[df.index == TUTORIAL_IDX]
    

    df = df[df.index != TUTORIAL_IDX]

    egs = []

    for label, status in product(sensitive_labels, list(set(df.pred))):
        prop = props_dict[status][label]
        print(label, status, prop, num_egs/num_groups * prop)

        n = round(num_egs/num_groups * prop)
        if GEN_ALL:
            part = df[(df.pred == status) & (df[sensitive_feature+"_x0_"+label] == 1)]

            egs.append(part.sample(len(part)-1))
        else:
            egs.append(df[(df.pred == status) & (df[sensitive_feature+"_x0_"+label] == 1)].sample(n))

    egs = pd.concat(egs).sort_values(by=['ListingNumber'])

    


    prosper_Y = egs[['GroupedLoanStatus']]+tutorial_eg[['GroupedLoanStatus']]
    prosper_X = egs.drop(columns = drop+['GroupedLoanStatus', 'pred'])+tutorial_eg.drop(columns = drop+['GroupedLoanStatus', 'pred'])


    orig_egs = prosper_biased[prosper_biased.index.isin(list(egs.index)+[TUTORIAL_IDX])].copy().sort_values(by=['ListingNumber'])
    orig_egs['pred'] = orig_egs.apply(lambda x:prosper_sen[bias_type][x.name], axis=1)

    


    for outcome, gender, prop in props:
        n = num_egs * .5 * prop # number of egs for study * 1/2 gender * proportion of outcome
        
        
    graph(condition, (min_x, max_x), tutorial_eg, "ne", orig_egs, sensitive_feature, cor_feature, feats, coefs, colors, train, save=True, just_extremes=False, tutorial=True)

    orig_egs = orig_egs[orig_egs.index != TUTORIAL_IDX]
    prosper_Y = egs[['GroupedLoanStatus']]
    prosper_X = egs.drop(columns = drop+['GroupedLoanStatus', 'pred'])




    num_participants = num_egs/20
    
    min_b = 2
    

    cutoff = min_b*2*num_participants #at least min_b per phase per participant


    if bias_type =="proxy":
        match = len(orig_egs[(orig_egs.University.isin(universities['wc'])) & (orig_egs.GroupedLoanStatus == "Late")])
        print("wc, l", match)
        assert(match >= cutoff)
    elif bias_type == "protected":
        # This bit is unnecessary
        match = len(orig_egs[(orig_egs.Gender == "f") & (orig_egs.GroupedLoanStatus == "Late")])
        print("f, l", match)
        assert(match >= cutoff) 
        match = len(orig_egs[(orig_egs.Gender == "m") & (orig_egs.GroupedLoanStatus == "Completed")])
        print("m, c", match)
        assert(match >= cutoff)


    for outcome, eg_dist in props_dict.items():
        if outcome == "Completed":
            label = positive
        else:
            label = 1-positive

        for g, p in eg_dist.items():
            cutoff = round(p*num_egs*.5)
            print(g,label,"cutoff", cutoff, len(orig_egs[(orig_egs.Gender==g) & (orig_egs.pred == label)]))
            assert(len(orig_egs[(orig_egs.Gender==g) & (orig_egs.pred == label)])>=cutoff)




    final_max, final_min =  graph_all(bias_type, egs, orig_egs, sensitive_feature, cor_feature, feats, coefs, colors, train, (min_x, max_x), save=True)


def calc_stats(train, train_X, cor_feature, sensitive_feature, condition, feats, categorical_columns):
    numbers = {}

    binarizer_sen = sklearn.preprocessing.LabelBinarizer()



    mapper_sen = DataFrameMapper(
        [(sensitive_feature, binarizer_sen)] + # target variable    
        [([col], sklearn.preprocessing.OneHotEncoder() ) for col in categorical_columns if col in [cor_feature]] +
        [([col], sklearn.preprocessing.StandardScaler()) for col in [cor_feature] if col not in categorical_columns] +
        [],
        sparse=False,
        df_out=True
    )

    df = mapper_sen.fit_transform(train)

    order_dict = mapper_ordering(mapper_sen, categorical_columns)


    column_names = {}
    for col in df.columns.to_list():
        col_split = col.split("_")
        if col_split[0] in order_dict:
            column_names[col] = col_split[0] + "_x0_" + str(order_dict[col_split[0]][int(col_split[1])])

    df = df.rename(columns=column_names)

    data = df.to_numpy()

    prosper_sen_Y = data[:,0]
    prosper_sen_X = data[:,1:]

    imp_mean = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='median') #, add_indicator=True)
    imp_mean.fit(prosper_sen_X)
    prosper_sen_X = imp_mean.transform(prosper_sen_X)


    



    feat_names = [order_dict["University"][int(x.split("_")[1])] for x in mapper_sen.transformed_names_[1:]]




    sensitive = prosper_sen_Y
    cors = []

    for i in range(len(feat_names)):
        name = feat_names[i]
        values = prosper_sen_X[:,i]
        
        cors.append(pearsonr(values, sensitive)[0])
        




    sorted_feats = sorted(zip(cors, feat_names))

    names = []
    vals = []

    for val,feat in sorted_feats: 

        print(f"{val:-2.3}\t{feat}")
        
        names.append(feat)
        vals.append(round(val,3))
        numbers["r_"+names[-1]] = vals[-1]




    fig, ax = plt.subplots()
    ax.imshow([vals], vmax= -min(vals))
    ax.set_xticks(np.arange(len(names)), labels=names)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    ax.set(yticklabels=[])  
    ax.tick_params(left=False)

    for j in range(len(vals)):
        if vals[j] < 0:
            text = ax.text(j, 0, vals[j], ha="center", va="center", color="yellow")
        else:
            text = ax.text(j, 0, vals[j], ha="center", va="center", color="purple")
            
            
    
    plt.savefig(STUDY_PATH+"/examples/corrs/corrs_"+condition+".png", bbox_inches = "tight")





    for f in [f for f in feats if f not in categorical_columns and f != "ListingCategory"]:
        print(f)
        values = train_X['LoanOriginalAmount']

        
        numbers["r_"+f] = round(pearsonr(values, sensitive)[0],3)





    f = 'DelinquenciesLast7Years'



    for col in feats + ['University', 'Gender']:
        if col in categorical_columns:
            df = train.groupby(['Gender', col])[col].count().unstack(fill_value=0).stack()



            a = np.array([list(df['m']), list(df['f'])])



            # Finding Chi-squared test statistic,
            # sample size, and minimum of rows
            # and columns
            X2 = stats.chi2_contingency(a, correction=False)[0]
            N = np.sum(a)
            minimum_dimension = min(a.shape)-1

            # Calculate Cramer's V
            result = np.sqrt((X2/N) / minimum_dimension)

            # Print the result
            print("$\\texttt{"+col+"}$& $",round(result,3),"$\\\\")
            numbers["cram_"+col] = round(result,3)
        else:
            pass


    with open("_".join([STUDY_PATH+"/dfs/numbers",condition])+".csv", 'w') as f: 
        w = csv.DictWriter(f, numbers.keys())
        w.writeheader()
        w.writerow(numbers)


    
    

def main():

    create_or_clear_directory(STUDY_PATH+"/examples")
    create_or_clear_directory(STUDY_PATH+"/examples/main")
    create_or_clear_directory(STUDY_PATH+"/examples/tutorial")
    create_or_clear_directory(STUDY_PATH+"/examples/corrs")
    create_or_clear_directory(STUDY_PATH+"/dfs")



    maj_percent = .6

    cor_dist =  {
                'f':[.15,.15,.04,.04, .14, .15, .15, .06, .07, .05], 
                'm':[0  ,0  ,.05,.05, .2, .2, .2, .1, .1, .1]
                }


    cor_labels =[
                    'Bryn Mawr College','Mount Holyoke College', 'Denison University', 'Scripps College',
                    'Kenyon College','Bucknell University','Harvey Mudd College',
                    'Lafayette College', 'Trinity College', 'Macalester College'
                ]

    universities = {
        "wc":['Bryn Mawr College','Mount Holyoke College'], #women's colleges
        "co":['Denison University', 'Scripps College',
                'Kenyon College','Bucknell University','Harvey Mudd College',
                'Lafayette College', 'Trinity College', 'Macalester College'], #coed
        "mm":["West Point Academy", "US Air Force Academy"] #majority male
    }

    interp_cols = ["ListingCategory (numeric)","LoanOriginalAmount","MonthlyLoanPayment","Term","BankcardUtilization","DebtToIncomeRatio","DelinquenciesLast7Years","OpenRevolvingMonthlyPayment","StatedMonthlyIncome","BorrowerState","EmploymentStatusDuration","EmploymentStatus","IsBorrowerHomeowner","Occupation"]

    categorical_columns = [
            'ProsperRating (Alpha)',
            'BorrowerState',
            'Occupation',
            'EmploymentStatus',
            'IncomeRange',
            'ListingCategory (numeric)',
            'University',
            'Gender',
        ]
    
    conditions = [
                    'protected_by_cn', # gender with bias disclosure
                    'proxy_by_cn', # university with only bias disclosure
                    'proxy_by_cy' # university with bias and correlation disclosure
                  ] # all have e and ne versions
    
    sensitive_feature = "Gender"
    sensitive_labels= ["m","f"]

    cor_feature = "University"

    feats = [
            'BorrowerState',
            'Occupation',
            'EmploymentStatus',
            'ListingCategory',
            'LoanOriginalAmount',
            'DelinquenciesLast7Years']

    
    for cond in conditions:

        # load prosper data
        prosper = load_and_clean(interp_cols)

        # generate protected and proxy features
        train_X, train_Y, test_X, test_Y, prosper_biased, prosper_sen, mapper, drop, test_full, train = generate_cols(cond, prosper, cor_labels, cor_dist, maj_percent, TUTORIAL_IDX, interp_cols, categorical_columns)

        # train the model
        model, coefs, positive = train_model(train_X, train_Y, test_X, test_Y, cond, prosper_sen, mapper, feats)
        
        # generate the explanations
        generate_viz(cond, coefs, test_full, prosper_sen, positive, sensitive_labels, maj_percent, sensitive_feature, cor_feature, feats, prosper_biased,  universities, TUTORIAL_IDX, drop, train)
        
        # calculate correlations
        calc_stats(train, train_X, cor_feature, sensitive_feature, cond, feats, categorical_columns)


if __name__ == '__main__':
    main()