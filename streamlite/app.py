
import requests
import streamlit as st
import pandas as pd


dataset_train = pd.read_parquet('../data/04_feature/base_train.parquet')
dataset_test = pd.read_parquet('../data/04_feature/base_test.parquet')
results = pd.read_parquet('../data/02_intermediate/data_filtered.parquet')
model = pd.read_pickle('../data/06_models/classifier_model/model.pkl')


# Side Bar Header
st.sidebar.title('Escolha das Features')
st.sidebar.markdown(f"""
Selecione as variáveis desejadas para prever se o arremesso será convertido em cesta.                    
""")


train_data = results
features = results.drop(['shot_made_flag'], axis=1).columns
target_col = results['shot_made_flag']
idx_train = dataset_train.index  # train_data.categoria == 'treino'
idx_test = dataset_test.index


# Side Bar Inputs
form = st.sidebar.form("input_form")

input_variables = {'target_col': 0}

for feature in features:
    input_variables[feature] = form.slider(feature.capitalize(),
                                           float(results[feature].min()),
                                           float(results[feature].max()))


form.form_submit_button("Resultado")


# Dashboard Header
st.title(f"""
Predição dos Arremessos de Kobe Bryant
""")

features_text = ''
for x in features:
    features_text = features_text + '\n - ' + str(x)


st.markdown(f"""
Será que é possível prever se os arremessos para cestas de dois pontos do mestre Kobe Bryant serão convertidos em cesta? 

Essa é a proposta do modelo de classificação desenvolvido nesse projeto. Esse modelo treinado com o dataset [Kobe Bryant Shot Selection](https://www.kaggle.com/c/kobe-bryant-shot-selection/data). 
Esse dataset possui um total de {int(results.shape[0]):,} arremessos, sendo que 80% ({ int( len(idx_train) ):,}) dos dados foi utilizado para treinamento do modelo e 20% ({ int( len(idx_test) ):,}) para validação.

Os arremessos são caracterizados por um total de 7 features: 
{features_text}
""")


@st.cache_data
def predict_user(input_variables):

    url = 'http://localhost:5001/invocations'
    data = {
        "dataframe_records":
        [
            input_variables
        ]
    }

    prediction = requests.post(url, json=data)
    result = prediction.json()

    return {
        'classificacao': list(result.values())[0][0]
    }


user = predict_user(input_variables)


if user['classificacao'] == 0:
    st.markdown("""<span style="font-size: 25px;">Classificação:</span>
    <span style="color:red; font-size: 38px;">*Quaaaaase....foi por pouco*</span>.
    """, unsafe_allow_html=True)
else:
    st.markdown("""<span style="font-size: 25px;">Classificação:</span>
    <span style="color:green; font-size: 38px;">*Cesta!*</span>
    """, unsafe_allow_html=True)
