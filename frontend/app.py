import streamlit as st
import requests

st.title("🎯 ロト6 予測アプリ（FastAPI連携）")

st.markdown("最新の当選番号をカンマ（,）で区切って6個入力してください。")

# 入力欄
numbers_input = st.text_input("例: 1, 5, 10, 20, 30, 40")

if st.button("🎰 予測する"):
    try:
        # 入力値をリストに変換
        numbers = [int(n.strip()) for n in numbers_input.split(",")]
        if len(numbers) != 6:
            st.error("❗ 数字は6個入力してください。")
        elif any(n < 1 or n > 43 for n in numbers):
            st.error("❗ 数字は1〜43の間で入力してください。")
        else:
            # FastAPI にリクエスト送信
            response = requests.post(
                "https://loto6-project.onrender.com/predict",
                json={"last_numbers": numbers}
            )
            if response.status_code == 200:
                result = response.json()
                st.success(f"🔮 予測された当選数字: {result['predicted_numbers']}")
            else:
                st.error("API からエラーが返されました。")
    except Exception as e:
        st.error(f"入力または通信エラー: {e}")
