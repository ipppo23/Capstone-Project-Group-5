CAPSTONE PROJECT GROUP 5
HOW TO RUN ON ANOTHER PC

Recommended folder structure:

Capstone Project Group 5/
│
├── Raw Data/
│   └── RQ1/
│       ├── 1410002001.csv
│       ├── 1410037101.csv
│       ├── 1410044201.csv
│       ├── 1410044301.csv
│       ├── 1410044401.csv
│       └── 3710019601.csv
│
└── src/
    ├── ml_prescriptive_pipeline.py
    ├── streamlit_app.py
    ├── project_paths.py
    └── requirements.txt


STEP 1 — DOWNLOAD / COPY THE PROJECT
Copy or download the folder "Capstone Project Group 5" to the new PC.

STEP 2 — OPEN TERMINAL
Open PowerShell or Command Prompt.
Go to the project folder.

Example:
cd "C:\Users\YourName\Desktop\Capstone Project Group 5"

STEP 3 — CREATE VIRTUAL ENVIRONMENT
Run:

python -m venv .venv

STEP 4 — ACTIVATE VIRTUAL ENVIRONMENT
If using PowerShell:

.venv\Scripts\Activate

If using Command Prompt:

.venv\Scripts\activate.bat

STEP 5 — INSTALL REQUIRED PACKAGES
Run:

pip install -r src\requirements.txt

If requirements.txt does not work, install manually:

pip install pandas numpy scikit-learn streamlit plotly openpyxl matplotlib

STEP 6 — RUN THE PIPELINE
This will create the output files automatically.

Run:

python src\ml_prescriptive_pipeline.py --project_root .

STEP 7 — RUN THE STREAMLIT APP
After the pipeline finishes, run:

streamlit run src\streamlit_app.py

STEP 8 — OPEN IN BROWSER
Streamlit will show a local link, usually:

http://localhost:8501

Open that in the browser if it does not open automatically.


NOTES
1. The CSV files must stay inside:
   Raw Data\RQ1\

2. Do not move the Python files out of:
   src\

3. The folder outputs_capstone will be created automatically after running the pipeline.

4. If the app says a TASK_ file is missing, run the pipeline first.

5. If Python is not recognized, install Python first and make sure it is added to PATH.


QUICK RUN VERSION

cd "C:\Users\YourName\Desktop\Capstone Project Group 5"
python -m venv .venv
.venv\Scripts\Activate
pip install -r src\requirements.txt
python src\ml_prescriptive_pipeline.py --project_root .
streamlit run src\streamlit_app.py