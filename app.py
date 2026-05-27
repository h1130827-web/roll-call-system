import streamlit as st
from deepface import DeepFace
import pandas as pd
from PIL import Image
import numpy as np
import os

st.title("🛡️ AI 自動點名系統")

# 1. 側邊欄設定
st.sidebar.header("系統設定")
db_path = "./db" # 確保 GitHub 專案裡有這個資料夾

# 2. 相機輸入
img_file = st.camera_input("請對準相機進行簽到")

if img_file:
    # 轉換格式
    img = Image.open(img_file)
    img_array = np.array(img)
    
    st.write("正在辨識中...")
    
    try:
        # DeepFace 辨識
        results = DeepFace.find(img_path=img_array, db_path=db_path, model_name='VGG-Face', enforce_detection=False)
        
        if len(results) > 0 and not results[0].empty:
            name = results[0].iloc[0]['identity'].split('/')[-1].split('.')[0]
            st.success(f"✅ 簽到成功！歡迎 {name}")
            
            # 這裡可以加入寫入 CSV 的邏輯
            # ...
        else:
            st.warning("❓ 無法辨識身分，請重試或聯絡管理員。")
            
    except Exception as e:
        st.error(f"發生錯誤: {e}")
