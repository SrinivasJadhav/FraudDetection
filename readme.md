# create env
uv venv .venv


Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process 
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# or Windows cmd
.\.venv\Scripts\activate.bat


uv pip uninstall scikit-learn numpy imbalanced-learn


uv pip install --upgrade --no-cache-dir --only-binary=:all:  "numpy<2.0.0"  "scikit-learn==1.4.2" "imbalanced-learn==0.12.3"


uv pip install --upgrade --no-cache-dir --only-binary=:all: -r requirements.txt


streamlit run app.py


