import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff

###################################
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import JsCode

###################################

from functionforDownloadButtons import download_button

###################################

import megaquant_processing as mp

###################################


#__________________________________        META        __________________________________#
st.set_page_config(page_icon="💰", page_title="PD Model | MegaQuant")
st.title("Привет, Алина!")

###################################        META        ###################################
##########################################################################################
##########################################################################################
##########################################################################################


#__________________________________      SIDEBAR       __________________________________#
st.sidebar.header('Выполнить предсказание для')
uploaded_file = st.sidebar.file_uploader(
        "Перетащите csv файл сюда!",
        key="1",
        help="To activate 'wide mode', go to the hamburger menu > Settings > turn on 'wide mode'",
    )

if uploaded_file is not None:
    file_container = st.expander("Был введен следующий csv файл")
    shows = pd.read_csv(uploaded_file, sep=';')
    uploaded_file.seek(0)
    file_container.write(shows)
else:
    st.sidebar.info(
        f"""
            👆 Попробуйте загрузить [df_test.csv](https://hse.kamran.uz/ps22/df_test.csv)
            """
    )

    def user_input_features():
        record_id                   = 0
        ar_revenue                  = st.sidebar.number_input('ar_revenue',                   value=0)
        ar_total_expenses           = st.sidebar.number_input('ar_total_expenses',            value=0)
        ar_sale_cost                = 1
        ar_selling_expenses         = st.sidebar.number_input('ar_selling_expenses',          value=0)
        ar_management_expenses      = st.sidebar.number_input('ar_management_expenses',       value=0)
        ar_sale_profit              = 1
        ar_balance_of_rvns_and_expns= 1
        ar_profit_before_tax        = 1
        ar_taxes                    = 1
        ar_other_profit_and_losses  = 1
        ar_net_profit               = 1
        ab_immobilized_assets       = 1
        ab_mobile_current_assets    = 1
        ab_inventory                = 1
        ab_accounts_receivable      = 1
        ab_other_current_assets     = st.sidebar.number_input('ab_other_current_assets', value=0)
        ab_cash_and_securities      = st.sidebar.number_input('ab_cash_and_securities', value=0)
        ab_losses                   = st.sidebar.number_input('ab_losses', value=0)
        ab_own_capital              = 1
        ab_borrowed_capital         = 1
        ab_long_term_liabilities    = st.sidebar.number_input('ab_long_term_liabilities', value=0)
        ab_short_term_borrowing     = st.sidebar.number_input('ab_short_term_borrowing', value=0)
        ab_accounts_payable         = 1
        ab_other_borrowings         = 1
        bus_age                     = 1
        ogrn_age                    = st.sidebar.slider('ogrn_age',        0, 135, 20)
        adr_actual_age              = 1
        head_actual_age             = 1
        cap_actual_age              = 3 #+st.sidebar.slider('cap_actual_age',  0, 1000, 3)
        ul_staff_range              = st.sidebar.select_slider('ar_sale_cost', ('[1-100]', '(100-500]', '> 500'))
        ul_capital_sum              = 25 #st.sidebar.slider('ul_capital_sum', 0, 130, 25)
        ul_founders_cnt             = 1
        ul_branch_cnt               = 0 #st.sidebar.slider('ul_strategic_flg', 0, 130, 0)
        ul_strategic_flg            = st.sidebar.select_slider('ar_sale_cost', (0,1))
        ul_systematizing_flg        = 1
        data = {
                'record_id'                   : record_id,
                'ar_revenue'                  : ar_revenue,
                'ar_total_expenses'           : ar_total_expenses,
                'ar_sale_cost'                : ar_sale_cost,
                'ar_selling_expenses'         : ar_selling_expenses,
                'ar_management_expenses'      : ar_management_expenses,
                'ar_sale_profit'              : ar_sale_profit,
                'ar_balance_of_rvns_and_expns': ar_balance_of_rvns_and_expns,
                'ar_profit_before_tax'        : ar_profit_before_tax,
                'ar_taxes'                    : ar_taxes,
                'ar_other_profit_and_losses'  : ar_other_profit_and_losses,
                'ar_net_profit'               : ar_net_profit,
                'ab_immobilized_assets'       : ab_immobilized_assets,
                'ab_mobile_current_assets'    : ab_mobile_current_assets,
                'ab_inventory'                : ab_inventory,
                'ab_accounts_receivable'      : ab_accounts_receivable,
                'ab_other_current_assets'     : ab_other_current_assets,
                'ab_cash_and_securities'      : ab_cash_and_securities,
                'ab_losses'                   : ab_losses,
                'ab_own_capital'              : ab_own_capital,
                'ab_borrowed_capital'         : ab_borrowed_capital,
                'ab_long_term_liabilities'    : ab_long_term_liabilities,
                'ab_short_term_borrowing'     : ab_short_term_borrowing,
                'ab_accounts_payable'         : ab_accounts_payable,
                'ab_other_borrowings'         : ab_other_borrowings,
                'bus_age'                     : bus_age,
                'ogrn_age'                    : ogrn_age,
                'adr_actual_age'              : adr_actual_age,
                'head_actual_age'             : head_actual_age,
                'cap_actual_age'              : cap_actual_age,
                'ul_staff_range'              : ul_staff_range,
                'ul_capital_sum'              : ul_capital_sum,
                'ul_founders_cnt'             : ul_founders_cnt,
                'ul_branch_cnt'               : ul_branch_cnt,
                'ul_strategic_flg'            : ul_strategic_flg,
                'ul_systematizing_flg'        : ul_systematizing_flg,
                }

        features = pd.DataFrame(data, index=[0])
        return features
    
    shows = user_input_features()

###################################       SIDEBAR      ###################################
##########################################################################################
##########################################################################################
##########################################################################################

#_______________________________        DISPLAY CSV       _______________________________#

from st_aggrid import GridUpdateMode, DataReturnMode

gb = GridOptionsBuilder.from_dataframe(shows)
## enables pivoting on all columns, however i'd need to change ag grid to allow export of pivoted/grouped data, however it select/filters groups
gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
gb.configure_selection(selection_mode="multiple", use_checkbox=True)
gb.configure_side_bar()  # side_bar is clearly a typo :) should by sidebar
gridOptions = gb.build()




###################################        DISPLAY CSV        ##################################
################################################################################################
################################################################################################
################################################################################################

#__________________________________        SHOW METRICS      __________________________________#
df = shows
df_stats = mp.make_df_ready_for_viz(df)

def scatter_3d(df_stats, x,y,z):
    import plotly.express as px
    fig = px.scatter_3d(df_stats, x,y,z,
                        color='Вероятность Дефолта',
                        symbol='выборка',
                        hover_name='выборка',
                        opacity=0.4,
                        size=df_stats['выборка'].replace({'выборка, на которой проводилось обучение': 0.1, 'введенная выборка':2}))
    fig.update_layout(coloraxis_colorbar_x=-0.15, height=800, width=1000)
    return fig

def scatter_2d(df_stats, x,y):
    import plotly.express as px
    fig = px.scatter(df_stats, x,y, 
                     color='Результат модели', 
                     marginal_x="box", 
                     marginal_y="histogram",
                     symbol='выборка',
                     opacity=0.4,
                     size=df_stats['выборка'].replace({'выборка, на которой проводилось обучение': 0.1, 'введенная выборка':1}))
    fig.update_layout(coloraxis_colorbar_x=-0.3, height=800, width=1000)
    return fig

c1,c2,c3 = st.columns([1, 1, 1])

with c1:
     x = st.selectbox('X',df_stats.columns)
with c2:
    y = st.selectbox('Y',df_stats.columns)
with c3:
    z = st.selectbox('Z',df_stats.columns)
fig_3d = scatter_3d(df_stats, x,y,z)
fig_2d = scatter_2d(df_stats, x,y)

st.plotly_chart(fig_3d)
st.plotly_chart(fig_2d)

###################################      SHOW METRICS         ##################################
################################################################################################
################################################################################################
################################################################################################


#__________________________________       PREDICT  CSV        __________________________________#

st.subheader("Предсказания 👇 ")
@st.cache
def predict(shows):
    return mp.predict_pretty(shows.copy())

def dist_plot(y_pred_probability):
    fig = ff.create_distplot([np.concatenate([[0,1],y_pred_probability])], ['Вероятность Дефолта'],show_hist=False,show_rug=False)
    fig.update_layout(title='Распределение вероятности введенных наблюдений',
                      xaxis_title="Веротяность Дефолта",
                      yaxis_title="Плотность Наблюдений",
                      legend_title="Легенда",
                      showlegend=False,)
    fig.update_layout(height=800, width=1000)
    return fig

def scatter_3d_clust(df_clust, x,y,z):
    import plotly.express as px
    fig = px.scatter_3d(df_clust, x,y,z,
                        color='clusters')
        
    fig.update_layout(height=800, width=1000)
    return fig

def get_clustered(df, k_to_try = 7):
    test_df = df.copy()
    import plotly.graph_objects as go
    from sklearn.cluster import KMeans
    import pingouin as pg
    from collections import Counter
    def try_different_clusters(K, data):
        cluster_values = list(range(1, K+1))
        inertias=[]
        
        for c in cluster_values:
            model = KMeans(n_clusters = c,init='k-means++',max_iter=400,random_state=42)
            model.fit(data)
            inertias.append(model.inertia_)
        
        return inertias

    # # !!!!!!!!!!!!!!______________________________________________________________________________________________________________________________________
    kmeans_model = KMeans(init='k-means++',  max_iter=400, random_state=42)
    kmeans_model.fit(test_df.drop('Вероятность Дефолта', axis = 1))
    outputs = try_different_clusters(k_to_try, test_df)

    # # ГРАФИК TO DO elbow_fig
    
    distances = pd.DataFrame({"clusters": list(range(1, 8)),"sum of squared distances": outputs})
    elbow_fig = go.Figure()
    elbow_fig.add_trace(go.Scatter(x=distances["clusters"], y=distances["sum of squared distances"]))

    elbow_fig.update_layout(xaxis = dict(tick0 = 1,dtick = 1,tickmode = 'linear'),                  
                      xaxis_title="Количество Кластеров",
                      yaxis_title="Сумма Расстояний",
                      title_text="График 'Метод локтя'")
    

    # автоматическое определение количества кластеров 
    difference = []
    for i in range(len(outputs) - 1): 
      difference.append(outputs[i + 1] - outputs[i])
    optimal_clusters = np.argmin(difference) + 3

    kmeans_model_new = KMeans(n_clusters = optimal_clusters, init='k-means++',max_iter=400,random_state=42)
    kmeans_model_new.fit(test_df)

    # определение самых норм признаков для кластеризации 
    cluster_centers = kmeans_model_new.cluster_centers_
    test_df["clusters"] = kmeans_model_new.labels_
    num_clusters = test_df["clusters"].nunique()

    a = []
    for clust1 in range(num_clusters):
      for clust2 in range(clust1 + 1, num_clusters):
        for i in test_df.columns:
          df_clust0 = test_df[test_df['clusters']== clust1][i]
          df_clust1 = test_df[test_df['clusters']== clust2][i]
          p_val = pg.ttest(x= df_clust0, y= df_clust1, correction=False)['p-val'][0]
          if p_val < 0.05:
            a.append(i)

    def printRepeating(arr):
        res = []
        freq = Counter(arr)
        for i in freq:
            if(freq[i] > num_clusters - 1):
                res.append(i)
        return res
    features_for_cluster = printRepeating(a) + ['Вероятность Дефолта'] #+['clusters']

    df_clust_info = test_df[features_for_cluster].groupby('clusters').mean()
    df_clust = test_df
    return df_clust, features_for_cluster,elbow_fig, df_clust_info


# # Типо выводит введеную таблицу и дает возможность поставить галочки. Оказалось что это не надо(
# response = AgGrid(
#     shows,
#     gridOptions=gridOptions,
#     enable_enterprise_modules=True,
#     update_mode=GridUpdateMode.MODEL_CHANGED,
#     data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
#     fit_columns_on_grid_load=False,
# )
# df = pd.DataFrame(response["selected_rows"])

df_to_clust = df_stats[df_stats['выборка'] == 'введенная выборка'].drop(columns=['выборка','Результат модели']) 
prediction = predict(shows)
pred_table = pd.DataFrame(prediction).T

st.dataframe(pred_table.T)
st.write(dist_plot(pred_table['Вероятность Дефолта']))



c29, c30, c31 = st.columns([1, 1, 1])
with c30:
    CSVButton = download_button(
        pred_table,
        'prediction.csv',
        'скачать предсказания'
    )

###################################       PREDICT  CSV        ##################################
################################################################################################
################################################################################################
################################################################################################


#__________________________________       CLASTER  CSV        __________________________________#

st.subheader("Кластерный анализ 🔎")

if len(pred_table) >= 7:
    # st.expander("Кластеризация была проведана по следующим данным").write(df_to_clust)
    df_clust, features_for_cluster, elbow_fig, df_clust_info = get_clustered(df_to_clust)

    st.write('**Средение значения выделенных кластеров по наглядным признакам:**')
    st.write(df_clust_info)

    st.write('**Визуализация кластеров:**')
    st.expander(f"Кластеризация наглядна по cледующим {len(features_for_cluster)-2} признакам").write(features_for_cluster)
    c1,c2,c3 = st.columns([1, 1, 1])
    with c1:
         x = st.selectbox('X',df_clust.columns)
    with c2:
        y = st.selectbox('Y',df_clust.columns)
    with c3:
        z = st.selectbox('Z',df_clust.columns)
    st.write(scatter_3d_clust(df_clust, x,y,z))

    st.write('**Оптимальное количество кластеров находиться по методу локтя:**')
    st.write(elbow_fig)
else:
    st.write(f'Кластерный анализ разумен для более 7 наблюдений. Вы ввели только {len(pred_table)}.')



###################################       CLASTER  CSV        ##################################
################################################################################################
################################################################################################
################################################################################################
