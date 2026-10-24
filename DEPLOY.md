# Deploying the MPG Predictor on Streamlit Community Cloud

## 1. Test locally
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501 and confirm a prediction works.

## 2. Push to GitHub
Folder layout:
```
mpg-predictor/
├── app.py
├── requirements.txt
└── auto-mpg.csv   (optional local copy of the dataset)
```
```bash
git init
git add .
git commit -m "MPG predictor"
git branch -M main
git remote add origin https://github.com/<your-username>/mpg-predictor.git
git push -u origin main
```

## 3. Deploy
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** -> **Deploy a public app from GitHub**.
3. Select your repo, branch `main`, main file path `app.py`.
4. Click **Deploy**. The first build takes a few minutes.
5. You get a public URL like `https://<your-app>.streamlit.app`.

## Troubleshooting
- Build fails on a package: pin the version in requirements.txt.
- Data won't load: the app downloads from the UCI repository; if that is
  unreachable, add `auto-mpg.csv` to the repo or upload a CSV in the sidebar.
- Slow start: training happens once and is cached by Streamlit.
