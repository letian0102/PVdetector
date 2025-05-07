mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"letian0102@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
maxUploadSize = 10240\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml