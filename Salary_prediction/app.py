# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 18:12:12 2025

@author: shreya
"""

import streamlit as st
import numpy as np
import pickle         # Used for loading the pre-trained model from disk
import random         # Used for generating random positions and delays for the money animation

# Load the model 👇

model = pickle.load(open(r'D:\nit_prac\assignments\salary_pred\linear_regression_model.pk1', 'rb'))

# Custom CSS + HTML Title

st.markdown("""
    <style>
        .main-header {
            text-align: center;
            margin-top: 20px;
            margin-bottom: 30px;
        }
        .main-title-bold {
            font-size: 45px;
            font-weight: 800;
            display: inline;
            color: #333;
        }
        .main-subtitle {
            font-size: 28px;
            font-weight: 400;
            color: #555;
            display: inline;
        }
        .sub-text {
            font-size: 20px;
            color: #666;
            text-align: center;
            margin-bottom: 20px;
        }
        .result-box {
            background-color: #eaf9ea;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #b8e1b8;
            color: #2e7d32;
            font-size: 18px;
            margin-top: 20px;
            text-align: center;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            font-size: 18px;
            color: #999;
        }

        /* Money animation */
        @keyframes floatUp 
            
            {
            0%   { transform: translateY(0); opacity: 1; }
            100% { transform: translateY(-120vh); opacity: 0; }
            }
        .money {
            position: fixed;
            font-size: 30px;
            animation: floatUp 3s ease-in-out;
            z-index: 9999;
            pointer-events: none;
        }
    </style>

    <div class="main-header">
        <span class="main-title-bold">💼 SalaryCast </span>
        <span class="main-subtitle">– Predict your future Salary</span>
    </div>
""", unsafe_allow_html=True)

# Subtitle
st.markdown('<div class="sub-text">Enter your experience and estimate your salary instantly using ML 🤖</div>', unsafe_allow_html=True)

# Input
yr_exp = st.number_input("📅 Enter years of experience:", min_value=0.0, max_value=50.0, value=1.0, step=0.5)

# Predict Button
if st.button("🔍 Predict Salary"):
    exp_input = np.array([[yr_exp]])
    pred = model.predict(exp_input)[0]
    formatted_salary = "${:,.2f}".format(pred)

    # Display prediction
    st.markdown(f'''
        <div class="result-box">
            ✅ Predicted salary for <b>{yr_exp}</b> year(s) of experience is:<br><b>{formatted_salary}</b>
        </div>
    ''', unsafe_allow_html=True)

    # Money effect – 3 bursts
    money_html = ""
    for _ in range(3):  # repeat 3 times
        for _ in range(10):  # 10 emojis per burst
            left = random.randint(0, 95)  # horizontal position
            emoji = random.choice(['💸', '💵', '💰'])
            delay = random.uniform(0, 2)  # delay to stagger
            money_html += f'''
                <div class="money" style="left:{left}%; top:100%; animation-delay:{delay}s;">{emoji}</div>
            '''

    # Inject animated money HTML
    st.markdown(money_html, unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">📊 Model trained using a dataset of salaries vs years of experience.</div>', unsafe_allow_html=True)
