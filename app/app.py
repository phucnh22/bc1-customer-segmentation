import os
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load
import sys
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # improves plot aesthetics
from matplotlib.figure import Figure
from io import BytesIO
import base64

def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])

def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        assert (y1 <= d <= y2) or (y2 <= d <= y1)
    x1, x2 = ranges[0]
    d = data[0]
    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdata.append((d-y1) / (y2-y1) 
                     * (x2 - x1) + x1)
    return sdata

class ComplexRadar():
    def __init__(self, fig, variables, ranges, n_ordinate_levels=5):
        angles = np.arange(0, 360, 360./len(variables))

        axes = [fig.add_axes([0.1,0.1,0.9,0.9],polar=True,
                label = "axes{}".format(i)) 
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles, 
                                         labels=variables)
        [txt.set_rotation(angle-90) for txt, angle 
             in zip(text, angles)]
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], 
                               num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x)) 
                         for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1] # hack to invert grid
                          # gridlabels aren't reversed
            gridlabel[0] = "" # clean up origin
            ax.set_rgrids(grid, labels=gridlabel,
                         angle=angles[i])
            #ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)


def radar_compare(cus_data, radar_df, min_max_df, variables = ['Age', 'Edu', 'Income', 'LTV', 'WebPurchase']):
    # example data
    #titles = ['Age', 'Edu', 'Income', 'LTV','Avr_purchase']
    data0 = radar_df.iloc[0][variables]
    data1 = radar_df.iloc[1][variables]
    data2 = radar_df.iloc[2][variables]
    data_customer = cus_data
    ranges = [(min_max_df.loc[feat,'0'], min_max_df.loc[feat,'1']) for feat in variables]          
    # plotting
    fig1 = plt.figure(figsize=(8, 5))
    radar = ComplexRadar(fig1, variables, ranges)
    radar.plot(data_customer, "-", lw=4, color="black", alpha=0.9, label="customer")
    radar.plot(data0, "-", lw=2, color="b", alpha=0.4, label="Wine Explorer")
    radar.plot(data1, "-", lw=2, color="r", alpha=0.4, label="Wine Expert")
    radar.plot(data2, "-", lw=2, color="g", alpha=0.4, label="Red Lover")

    radar.fill(data0, alpha=0.2)
    radar.fill(data1, alpha=0.2)
    radar.fill(data2, alpha=0.2)
    radar.ax.legend(loc = 'lower right', bbox_to_anchor=(1.1, 0.1), fontsize = 'xx-small' )
    figfile = BytesIO()
    plt.savefig(figfile,transparent=True, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue()).decode('utf8')
    return figdata_png

PROJECT_ROOT = Path(os.path.abspath('')).resolve()

#sys.path.insert(0, os.path.join(PROJECT_ROOT,'data','radar.py'))
#sys.path.insert(1, os.path.join(PROJECT_ROOT,'data'))
#from myfolder.myfile import myfunc

app = Flask(__name__)  # Initialize the flask App
model = load(os.path.join(PROJECT_ROOT, 'model', 'best_decision_tree22.joblib'))

#cus_data = data.iloc[4,:-2].values
radar_df = pd.read_csv(os.path.join(PROJECT_ROOT,"data","radar.csv"),index_col=0)
min_max_df = pd.read_csv(os.path.join(PROJECT_ROOT,"data","radar_minmax.csv"), index_col=0)
variables = ['Age', 'Edu', 'Income', 'LTV', 'WebPurchase']

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = list(request.form.values())[0].split(',')
    final_features = np.array(int_features).reshape(1, -1)
    prediction = model.predict(final_features)[0]
    if prediction == 2:
        profile = "The Red Lover"
        profile_text = ["- Middle age married couples living with teenage children and sometimes kids in the family",
                        "- Special love for dry red wines ",
                        "- Sophisticated with high level of education",
                        "- Maybe currently saving for children future (buying with promotion deals)"]
        marketing_text = ["- Collaboration with web-food-company (e.g. cheese)",
                          "- Promo-code for partner restaurant for family dining or take-home orders",
                          "- Include rack cross-sale in web-purchase"]
    elif prediction == 1:
        profile = "The Wine-Expert"
        profile_text = ["- The best group",
                        "- The oldest group of customers of the company which spends a considerable amount of time, effort and money about wine-products",
                        "- Highest income, and also the most active considering the frequency of their purchases for the company products",
                        "- Maybe like to discover the different typologies and tastes that the company offers",
                        "- Not interested in discounts"]
        marketing_text = ["- Better to reach him by mail or other physical method, no digital",
                          "- Propose a loyalty card that can be sent at home after a defined good number of purchases completed",
                          "- Include accessories as gift after purchases to express appreciation",
                          "- Invite to some wine-event as a VIP-member"]
    else:
        profile = "The Wine-Explorer"
        profile_text = ["- Beginner level wine enthusiasts",
                        "- Young age and low income",
                        "- Web-oriented, people who grew up in the age of technology",
                        "- Extremely high preference for promotions deals, low Life Time Value",
                        "- Curious about different types of wines, but mostly in the range of sweet and unusual wines, not very interested in dry wines"]
        marketing_text = ["- Advertising through digital platform such as social medial page",
                          "- Encourage subcription to e-mail newsletters with promotions and deals",
                          "- Offer discount to first-time purchase to acquire new customer"]    
    cus_data = final_features[0][radar_df.iloc[:,:-1].columns.isin(variables)]
    new_cus_data = [int(numeric_string) for numeric_string in cus_data]
    #radar_image = cus_data
    radar_image = radar_compare(new_cus_data, radar_df, min_max_df)


    return render_template('Home.html', 
            prediction_text='The customer belongs to cluster ' + profile,
            profile_text = profile_text,
            marketing_text = marketing_text,
            radar = radar_image)

if __name__ == "__main__":
    app.run(use_debugger=False, use_reloader=False, passthrough_errors=True)





