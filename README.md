# 🔥 Forest Fire Prediction using Flask & Machine Learning

This project predicts the **Forest Fire Weather Index (FWI)** — a numeric rating of fire intensity — based on environmental features. It uses a **Ridge Regression model**, wrapped in a Flask web app for easy use.

---

## 🧠 What Does It Do?

You enter environmental data like temperature, humidity, wind speed, etc., and the model predicts the **FWI value**, which helps estimate how intense a forest fire could be.

---

## 💻 Tech Stack

- Python
- Scikit-learn (ML Model)
- Flask (Web App)
- HTML & CSS (Frontend)

---

## 📊 Features Used

- **Temperature**
- **Relative Humidity (RH)**
- **Wind Speed (WS)**
- **Rain**
- **FFMC** – Fine Fuel Moisture Code
- **DMC** – Duff Moisture Code
- **ISI** – Initial Spread Index
- **Classes** – Fire weather class
- **Region** – Area label (e.g., Bejaia or Sidi-Bel)

---

## 🖥 How to Run Locally

1. Clone the repository  
   `git clone https://github.com/balaji5744/forest-fire-flask-app.git`

2. Navigate to the project folder  
   `cd forest-fire-flask-app`

3. Install the dependencies  
   `pip install -r requirements.txt`

4. Run the Flask app  
   `python application.py`

5. Open in browser  
   Visit `http://localhost:5000`

---

## 🧪 Output

- The output is a **FWI (Forest Fire Weather Index)** value.
- This is a continuous numeric value.  
  Higher the FWI → Higher the fire risk.

---

## 🔮 Future Improvements

- Improve prediction accuracy using ensemble models
- Add data visualizations (e.g., time series)
- Deploy using AWS or Streamlit with autoscaling
- Add alert system for high FWI values

---

## 👤 Author

**Balaji S**  
📧 sayabalaji1@gmail.com  
🌐 [GitHub Profile](https://github.com/balaji5744)

---

## 📝 License

This project is open-source and licensed under the MIT License.
