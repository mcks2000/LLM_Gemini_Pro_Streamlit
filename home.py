import streamlit as st
from PIL import Image
import google.generativeai as genai

st.set_page_config(page_title="Gemini Pro with Streamlit", page_icon="♊")

st.write("欢迎来到 Gemini Pro 聊天机器人。您可以通过提供您的 Google API 密钥来继续。")

with st.expander("提供您的 Google API 密钥"):
     google_api_key = st.text_input("Google API 密钥", key="google_api_key", type="password")
     
if not google_api_key:
    st.info("请输入 Google API 密钥以继续")
    st.stop()

genai.configure(api_key=google_api_key)

st.title("Gemini Pro 与 Streamlit 聊天机器人")

with st.sidebar:
    option = st.selectbox('选择您的模型',('gemini-pro', 'gemini-pro-vision'))

    if 'model' not in st.session_state or st.session_state.model != option:
        st.session_state.chat = genai.GenerativeModel(option).start_chat(history=[])
        st.session_state.model = option
    
    st.write("在此处调整您的参数:")
    temperature = st.number_input("温度", min_value=0.0, max_value= 1.0, value =0.5, step =0.01)
    max_token = st.number_input("最大输出令牌数", min_value=0, value =100)
    gen_config = genai.types.GenerationConfig(max_output_tokens=max_token, temperature=temperature)

    st.divider()
    st.markdown("""<span ><font size=1>与我联系</font></span>""", unsafe_allow_html=True)
    "[领英](https://www.linkedin.com/in/cornellius-yudha-wijaya/)"
    "[GitHub](https://github.com/cornelliusyudhawijaya)"
    
    st.divider()
    
    upload_image = st.file_uploader("在此上传您的图片", accept_multiple_files=False, type = ['jpg', 'png'])
    
    if upload_image:
        image = Image.open(upload_image)
    st.divider()

    if st.button("清除聊天历史"):
        st.session_state.messages.clear()
        st.session_state["messages"] = [{"role": "assistant", "content": "你好。我可以帮助你吗？"}]

 
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "你好。我可以帮助你吗？"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if upload_image:
    if option == "gemini-pro":
        st.info("请切换到 Gemini Pro Vision")
        st.stop()
    if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            response=st.session_state.chat.send_message([prompt,image], stream=True, generation_config=gen_config)
            response.resolve()
            msg=response.text

            st.session_state.chat = genai.GenerativeModel(option).start_chat(history=[])
            st.session_state.messages.append({"role": "assistant", "content": msg})
            
            st.image(image, width=300)
            st.chat_message("assistant").write(msg)

else:
    if prompt := st.chat_input():
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            response=st.session_state.chat.send_message(prompt, stream=True, generation_config=gen_config)
            response.resolve()
            msg=response.text
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
