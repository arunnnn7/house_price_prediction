import joblib
import pandas as pd
import gradio as gr

# -------------------------------
# Load Model + City Encoder
# -------------------------------
model, city_means = joblib.load("model/house_model.pkl")

# -------------------------------
# City Encoding Function
# -------------------------------
def encode_city(city):
    city = city.strip()
    return city_means.get(city, city_means.mean())

# -------------------------------
# Prediction Function
# -------------------------------
def predict_price(
    bedrooms,
    bathrooms,
    sqft_living,
    sqft_lot,
    floors,
    waterfront,
    view,
    condition,
    sqft_above,
    sqft_basement,
    yr_built,
    yr_renovated,
    city
):
    try:
        data = {
            "bedrooms": float(bedrooms),
            "bathrooms": float(bathrooms),
            "sqft_living": float(sqft_living),
            "sqft_lot": float(sqft_lot),
            "floors": float(floors),
            "waterfront": int(waterfront),
            "view": int(view),
            "condition": int(condition),
            "sqft_above": float(sqft_above),
            "sqft_basement": float(sqft_basement),
            "yr_built": int(yr_built),
            "yr_renovated": int(yr_renovated),

            # ✅ Send BOTH
            "city": city,
            "city_encoded": encode_city(city)
        }

        df = pd.DataFrame([data])
        price = model.predict(df)[0]

        return f"🏠 Estimated House Price: ₹{price:,.0f}"

    except Exception as e:
        return f"❌ Error: {str(e)}"

# -------------------------------
# Gradio UI
# -------------------------------
interface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Bedrooms", value=3),
        gr.Number(label="Bathrooms", value=2),
        gr.Number(label="Sqft Living", value=1500),
        gr.Number(label="Sqft Lot", value=5000),
        gr.Number(label="Floors", value=1),
        gr.Number(label="Waterfront (0 = No, 1 = Yes)", value=0),
        gr.Number(label="View (0-4)", value=0),
        gr.Number(label="Condition (1-5)", value=3),
        gr.Number(label="Sqft Above", value=1500),
        gr.Number(label="Sqft Basement", value=0),
        gr.Number(label="Year Built", value=2000),
        gr.Number(label="Year Renovated", value=0),
        gr.Textbox(label="City", placeholder="Enter city name")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="🏠 AI House Price Prediction System",
    description="Enter house details to predict estimated market price using a Boosted ML Model"
)

# -------------------------------
# Run App
# -------------------------------
interface.launch()
