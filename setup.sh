mkdir -p ~/.streamlit/
cat <<CONFIG > ~/.streamlit/credentials.toml
[general]
email = "letian0102@gmail.com"
CONFIG
cat <<CONFIG > ~/.streamlit/config.toml
[server]
headless = true
maxUploadSize = 10240
enableCORS = false
port = $PORT
address = "0.0.0.0"
CONFIG
