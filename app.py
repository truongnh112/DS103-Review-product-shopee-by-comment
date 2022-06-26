import streamlit as st
import pandas as pd
import numpy as np
import pickle
import preprocessing
from preprocessing import *
from pipeline import *
from build_data import *
from sklearn.feature_extraction.text import CountVectorizer


#load model da train

#gridSVM = pickle.load(open('Model/gridsvm.pkl', 'rb'))

modelSVM = pickle.load(open('Model/model_svm.pkl', 'rb'))




def analyze(result):
    total_score = 0
    num_of_vneg=0
    num_of_sneg=0
    num_of_neu=0
    num_of_spos=0
    num_of_vpos=0
    for pred in result:
        total_score = total_score+pred
        if pred == -2: 
            num_of_vneg = num_of_vneg+1
        elif pred == -1:
            num_of_sneg = num_of_sneg + 1
        elif pred == 0:
            num_of_neu = num_of_neu +1
        elif pred == 1:
            num_of_spos = num_of_spos +1
        elif pred == 2:
            num_of_vpos = num_of_vpos +1
    st.write('Điểm hài lòng: ', total_score)
    
    objects = ("rất tiêu cực", "hơi tiêu cực", "trung tính", "hơi tích cực", "rất tích cực")
    y_pos = np.arange(len(objects))
    performance = [num_of_vneg, num_of_sneg, num_of_neu, num_of_spos, num_of_vpos]
    df = pd.DataFrame(performance,objects )
    st.bar_chart(df)


    if total_score>0:
        st.write("Tốt, yên tâm mua") 
    else:
        st.write("Không tốt, cân nhắc và kỹ lượng trước khi mua" ) 

def precdict_by_link(model,list_cmt):
    X_test = encode_list(list_cmt)
    y_pred = model.predict(X_test)
    return y_pred




def main():
    
    
    st.set_page_config(
    page_title="Evaluate product",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
    )
    
    col1, col2 = st.columns([6, 4])

    with col1:
        st.header("PHÂN TÍCH ĐỘ HÀI LÒNG CỦA KHÁCH HÀNG")
        url = st.text_input("Link sản phẩm: ")

        if url:
            r = re.search(r"i\.(\d+)\.(\d+)", url)
            shop_id, item_id = r[1], r[2]
            st.write("Shop ID: ", shop_id)
            st.write("Product ID: ", item_id)
            try:
                crawl_data(url)
                build_dataset()
            except:
                st.write("Lấy link khác đi")
            data = pd.read_csv('data/dataset.csv')
            with open('data/dataset.csv', 'rb') as csv:
                file_container = st.expander("Check your crawl .csv")
                shows = pd.read_csv('data/dataset.csv')
                file_container.write(shows)
                st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='comment.csv',
                mime='text/csv',
                )

            if st.button('Analyze'):
                    with st.spinner("Analyzing..."):

                        result = precdict_by_link(modelSVM,data['comment'])
                        analyze(result=result)

                        st.success(f'Analysis finished')               
           
       

                
    
       
        

    with col2:
        st.markdown("**DỰ ĐOÁN MỘT BÌNH LUẬN**")
        option = st.selectbox('Select a review form:',
        ['None', 'SVM Kernel RBF'])

        st.write('Options: ',  option)

        with st.form(key="text"):
            raw_review = st.text_area("Review")
            submit = st.form_submit_button(label="Submit")

        st.write("Submit: ", submit)
        if submit:
            with st.spinner("Predicting..."):

                if option == 'SVM Kernel RBF':
                    model = modelSVM
                    st.write(predict_raw(model, raw_review))
           
    
    

if __name__ == "__main__":
    main()