import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from streamlit_folium import st_folium
import folium

#Ladda modell och kolumner
model_package = joblib.load("xgb_log_model.pkl")
model = model_package["model"]
expected_cols = model_package["columns"]

#Titel
st.title("XGBoost-modell - Bostadsprisprognos")
st.markdown("Ange uppgifter om bostaden för att beräkna ett uppskattat utgångspris:")

#Inmatning
col1, col2 = st.columns(2)
with col1:
    living_area = float(st.number_input("Boyta (kvm)", min_value=0.0, step=5.0))
    land_area = float(st.number_input("Tomtarea (kvm)", min_value=0.0, step=5.0))
    rooms = int(st.number_input("Antal rum", min_value=1, max_value=20, step=1))

with col2:
    typology_group_sv = st.selectbox(
        "Bostadstyp",
        ["Villa / Hus", "Lägenhet", "Gårdsfastighet / Mark", "Annat"]
    )

#Interaktiv karta för platsval
default_location = [59.33, 18.06]

st.markdown("Välj bostadens plats på kartan")
st.caption("Klicka på kartan för att placera markören.")

m = folium.Map(location=default_location, zoom_start=6, tiles="OpenStreetMap")

marker = folium.Marker(
    location=default_location,
    draggable=True,
    popup="Dra mig till bostadens plats"
)
marker.add_to(m)

map_data = st_folium(m, width=700, height=400)

if map_data and map_data.get("last_clicked"):
    latitude = map_data["last_clicked"]["lat"]
    longitude = map_data["last_clicked"]["lng"]
else:
    latitude = default_location[0]
    longitude = default_location[1]

st.write(f"**Vald plats:** Lat {latitude:.5f}, Long {longitude:.5f}")

#Omvandling
living_area_sqft = living_area * 10.7639
land_area_sqft = land_area * 10.7639

#Gruppering 
typology_groups = {
    "Villa / Hus": ["House"],
    "Lägenhet": ["Apartment"],
    "Gårdsfastighet / Mark": ["Estate"],
    "Annat": ["Other"]
}

chosen_group = typology_groups[typology_group_sv]

input_dict = {
    "living_area": living_area_sqft,
    "land_area": land_area_sqft,
    "rooms": rooms,
    "latitude": latitude,
    "longitude": longitude,
}

for col in expected_cols:
    if col.startswith("typology_"):
        input_dict[col] = 1 if any(col.endswith(g) for g in chosen_group) else 0
    elif col not in input_dict:
        input_dict[col] = 0 

new_data = pd.DataFrame([input_dict])
new_data = new_data[expected_cols] 

#Prediktion
if st.button("Beräkna uppskattat pris"):
    log_pred = model.predict(new_data)
    price_pred = np.exp(log_pred)[0]

    st.subheader("Uppskattat pris")
    st.write(f"**≈ {price_pred:,.0f} SEK**")
    st.markdown("_Observera: Detta är en uppskattning baserad på historiska data._")

#Viktigaste faktorerna
st.markdown("---")
st.subheader("Viktigaste faktorerna för modellen")

if st.checkbox("Visa feature importance"):
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": expected_cols,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False).head(5)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df["Feature"], importance_df["Importance"], color="steelblue")
    ax.invert_yaxis()
    ax.set_xlabel("Viktighet")
    ax.set_ylabel("Feature")
    ax.set_title("De 5 viktigaste faktorerna enligt XGBoost")
    st.pyplot(fig)
