
import streamlit as st
import numpy as np
import pandas as pd
import pickle


# Load  both models
import joblib
import os
import gdown

# Download new_model.pkl
if not os.path.exists("new_model.pkl"):
    gdown.download("https://drive.google.com/uc?id=1Vgan6YToyiNhYd2fLzEOa-Tt2mF6-TA4", "new_model.pkl", quiet=False)

# Download used_model.pkl
if not os.path.exists("used_model.pkl"):
    gdown.download("https://drive.google.com/uc?id=1KZC3O-vnMfhzj6Rx_sUYEet8S0_uYybL", "used_model.pkl", quiet=False)

# Download make_model_to_styles.pkl
if not os.path.exists("make_model_to_styles.pkl"):
    gdown.download("https://drive.google.com/uc?id=17HDzSlZiP8Qlwgb18ETpuuhUIO4xNTzK", "make_model_to_styles.pkl", quiet=False)


# Load both models and style mapping using joblib
new_model = joblib.load("new_model.pkl")
used_model = joblib.load("used_model.pkl")
# make_model_to_styles = joblib.load("make_model_to_styles.pkl")

# Load style mapping using pickle (not joblib!)
with open("make_model_to_styles.pkl", "rb") as f:
    make_model_to_styles = pickle.load(f)

# --- Model choice ---
car_type = st.radio("Select Vehicle Type", ["Used", "New"])
model = used_model if car_type == "Used" else new_model
# Mileage input
mileage = st.number_input("Mileage", value=0)

# Threshold for new car
NEW_CAR_MILEAGE_THRESHOLD = 1000
USED_CAR_MIN_MILEAGE_THRESHOLD = 1000

if car_type == "New" and mileage > NEW_CAR_MILEAGE_THRESHOLD:
    st.warning("⚠️ A new car typically should not have more than 1,000 km. Please verify this is a new vehicle.")
elif car_type == "Used" and mileage < USED_CAR_MIN_MILEAGE_THRESHOLD:
    st.warning("⚠️ A used car typically has more than 1,000 km. Please verify this is not a new vehicle.")

st.title(" 🚗 Vehicle Price Prediction App")

make_to_models = {
    "Audi": {"A3": 12, "A4": 13,"A4 Allroad": 14, "A5": 15, "A6": 16, "A6 Allroad": 17,
             "Q3": 156, "Q4 e-tron": 157, "Q5": 158, "Q7": 160, "Q8": 161, "Q8 e-tron": 162,
             "S3": 182, "S4": 183, "S5": 184, "SQ5": 186},
    "Acura": {"Integra": 121, "MDX": 131, "RDX": 168, "TLX": 206,"ZDX": 248},
    "Alfa Romeo": { "Giulia": 107, "Stelvio": 204, "Tonale": 214},
    "BMW": {"2-Series": 3, "3-Series": 5, "4-Series": 8, "5-Series": 10, "M2": 130,
            "X1": 234, "X2": 235, "X3": 236, "X4": 237, "X5": 238},
    "Buick": {"Enclave": 74, "Encore GX": 75, "Envision": 76, "Envista": 77},
    "Cadillac": {"CT4": 39, "CT5": 40, "Lyriq": 129, "XT4": 242, "XT5": 243, "XT6": 244},
    "Chevrolet": {
        "Blazer": 28,
        "Blazer EV": 29,
        "Bolt EUV": 30,
        "Bolt EV": 31,
        "Camaro": 46,
        "Colorado": 54,
        "Equinox": 78,
        "Equinox EV": 79,
        "Express Cargo": 84,
        "Malibu": 134,
        "Silverado 1500": 196,
        "Silverado EV": 197,
        "Suburban": 205,
        "Tahoe": 209,
        "Trailblazer": 215,
        "Traverse": 220,
        "Trax": 221},
    "Chrysler": {
        "300": 6,
        "Grand Caravan": 110,
        "Pacifica": 148},
    "Dodge": {
        "Challenger": 50,
        "Charger": 51,
        "Durango": 67,
        "Hornet": 117},
    "Fiat": {
        "500e": 11},
    "Ford": {
        "Bronco": 32,
        "Bronco Sport": 33,
        "E-Transit 350": 69,
        "Edge": 72,
        "Escape": 80,
        "Expedition": 81,
        "Expedition Max": 82,
        "Explorer": 83,
        "F150": 86,
        "F150 Lightning": 87,
        "Maverick": 135,
        "Mustang": 140,
        "Mustang MACH-E": 141,
        "Ranger": 175,
        "Transit 150": 216,
        "Transit 250": 217,
        "Transit 350": 218,
        "Transit 350HD": 219},
    "Genesis": {
        "G70": 92,
        "G80": 93,
        "GV70": 104,
        "GV80": 105},
    "GMC": {
        "Acadia": 18,
        "Canyon": 48,
        "Savana": 189,
        "Savana Cargo": 190,
        "Sierra 1500": 195,
        "Terrain": 212,
        "Yukon": 245,
        "Yukon XL": 246},
    "Honda": {
        "Accord": 19,
        "CR-V": 38,
        "Civic": 53,
        "HR-V": 115,
        "Integra": 121,
        "Odyssey": 145,
        "Passport": 150,
        "Pilot": 152,
        "Ridgeline": 179},
    "Hyundai": {
        "Elantra": 73,
        "Kona": 124,
        "Kona EV": 125,
        "Palisade": 149,
        "Santa Cruz": 187,
        "Santa Fe": 188,
        "Sonata": 198,
        "Tucson": 222,
        "Venue": 227},
    "Jaguar": {
        "F-Pace": 85},
    "Jeep": {
        "Cherokee": 52,
        "Compass": 55,
        "Gladiator": 108,
        "Grand Cherokee": 111,
        "Grand Cherokee L": 112,
        "Renegade": 178,
        "Wagoneer": 231,
        "Wagoneer L": 232,
        "Wrangler": 233},
    "Kia": {
        "Carnival": 49,
        "Forte": 89,
        "Forte5": 90,
        "Niro": 144,
        "Rio": 180,
        "Seltos": 191,
        "Sorento": 199,
        "Soul": 200,
        "Sportage": 201,
        "Telluride": 211},
    "Land Rover": {
        "Defender": 64,
        "Discovery": 65,
        "Discovery Sport": 66,
        "Range Rover Evoque": 173,
        "Range Rover Velar": 174},
    "Lexus": {
        "ES-Series": 70,
        "GX-Series": 106,
        "IS-Series": 119,
        "NX-Series": 142,
        "RX-Series": 170,
        "RX350": 171,
        "RX350h": 172,
        "TX-Series": 207,
        "UX-Series": 224},
    "Lincoln": {
        "Aviator": 26,
        "Corsair": 61,
        "Nautilus": 143},
    "Maserati": {
        "Grecale": 114},
    "Mazda": {
        "CX-30": 41,
        "CX-5": 42,
        "CX-50": 43,
        "CX-9": 44,
        "CX-90": 45,
        "MX-5": 132,
        "MX5": 133,
        "Mazda3": 137},
    "Mercedes-Benz": {
        "C-Class": 34,
        "CLA": 35,
        "CLA-Class": 36,
        "CLE": 37,
        "E-Class": 68,
        "GLA": 94,
        "GLB": 95,
        "GLB-Class": 96,
        "GLC": 97,
        "GLC-Class": 98,
        "GLE-Class": 99,
        "Sprinter 2500": 202,
        "Sprinter 2500 Cargo": 203},
    "Mini": {
        "Convertible": 56,
        "Cooper": 57,
        "Cooper Countryman": 58},
    "Mitsubishi": {
        "Eclipse Cross": 71,
        "Mirage": 138,
        "Outlander": 147,
        "RVR": 169},
    "Nissan": {
        "Altima": 20,
        "Ariya": 21,
        "Armada": 22,
        "Frontier": 91,
        "Kicks": 123,
        "Leaf": 127,
        "Maxima": 136,
        "Murano": 139,
        "Pathfinder": 151,
        "Qashqai": 167,
        "Rogue": 181,
        "Sentra": 192,
        "Versa": 229,
        "Z": 247},
    "Ram": {
        "1500": 0,
        "1500 Classic": 1,
        "1500 ProMaster": 2,
        "2500 ProMaster": 4,
        "3500 ProMaster": 7,
        "ProMaster": 155},
    "Subaru": {
        "Ascent": 23,
        "BRZ": 27,
        "Crosstrek": 62,
        "Forester": 88,
        "Impreza": 120,
        "Legacy": 128,
        "Outback": 146,
        "WRX": 230},
    "Toyota": {
        "4Runner": 9,
        "Camry": 47,
        "Corolla": 59,
        "Corolla Cross": 60,
        "Crown": 63,
        "GR Corolla": 100,
        "GR Supra": 101,
        "GR86": 102,
        "Grand Highlander": 113,
        "Highlander": 116,
        "Land Cruiser": 126,
        "Prius": 153,
        "Prius Prime": 154,
        "Rav4": 176,
        "Rav4 Prime": 177,
        "Sequoia": 193,
        "Sienna": 194,
        "Tacoma": 208,
        "Tundra": 223,
        "Venza": 228},
    "Volkswagen": {
        "Atlas": 24,
        "Atlas Cross Sport": 25,
        "Golf R": 109,
        "GTI": 103,
        "ID.4": 118,
        "Jetta": 122,
        "Taos": 210,
        "Tiguan": 213},
    "Volvo": {
        "S60": 185,
        "V60": 225,
        "V90": 226,
        "XC40": 239,
        "XC60": 240,
        "XC90": 241}
}


# ----------------- MAPPING DICTIONARIES -----------------
make_mapping = {
    'Acura': 0, 'Alfa Romeo': 1, 'Audi': 2, 'BMW': 3, 'Buick': 4, 'Cadillac': 5,
    'Chevrolet': 6, 'Chrysler': 7, 'Dodge': 8, 'Fiat': 9, 'Ford': 10, 'GMC': 11,
    'Genesis': 12, 'Honda': 13, 'Hyundai': 14, 'Infiniti': 15, 'Jaguar': 16, 'Jeep': 17,
    'Kia': 18, 'Land Rover': 19, 'Lexus': 20, 'Lincoln': 21, 'Maserati': 22, 'Mazda': 23,
    'Mercedes-Benz': 24, 'Mini': 25, 'Mitsubishi': 26, 'Nissan': 27, 'Ram': 28, 'Subaru': 29,
    'Toyota': 30, 'Volkswagen': 31, 'Volvo': 32
}

model_mapping = {'1500': 0, '1500 Classic': 1,
                 '1500 ProMaster': 2, '2-Series': 3, '2500 ProMaster': 4, '3-Series': 5, '300': 6,
                 '3500 ProMaster': 7, '4-Series': 8, '4Runner': 9, '5-Series': 10, '500e': 11, 'A3': 12,
                 'A4': 13, 'A4 Allroad': 14, 'A5': 15, 'A6': 16, 'A6 Allroad': 17, 'Acadia': 18,
                 'Accord': 19, 'Altima': 20, 'Ariya': 21, 'Armada': 22, 'Ascent': 23,
                 'Atlas': 24, 'Atlas Cross Sport': 25, 'Aviator': 26, 'BRZ': 27, 'Blazer': 28,
                 'Blazer EV': 29, 'Bolt EUV': 30, 'Bolt EV': 31, 'Bronco': 32, 'Bronco Sport': 33,
                 'C-Class': 34, 'CLA': 35, 'CLA-Class': 36, 'CLE': 37,
                 'CR-V': 38, 'CT4': 39, 'CT5': 40, 'CX-30': 41,
                 'CX-5': 42, 'CX-50': 43, 'CX-9': 44, 'CX-90': 45,
                 'Camaro': 46, 'Camry': 47, 'Canyon': 48, 'Carnival': 49,
                 'Challenger': 50, 'Charger': 51, 'Cherokee': 52, 'Civic': 53,
                 'Colorado': 54, 'Compass': 55, 'Convertible': 56, 'Cooper': 57,
                 'Cooper Countryman': 58, 'Corolla': 59, 'Corolla Cross': 60,
                 'Corsair': 61, 'Crosstrek': 62, 'Crown': 63, 'Defender': 64,
                 'Discovery': 65, 'Discovery Sport': 66, 'Durango': 67,
                 'E-Class': 68, 'E-Transit 350': 69, 'ES-Series': 70,
                 'Eclipse Cross': 71, 'Edge': 72, 'Elantra': 73,
                 'Enclave': 74, 'Encore GX': 75, 'Envision': 76,
                 'Envista': 77, 'Equinox': 78, 'Equinox EV': 79,
                 'Escape': 80, 'Expedition': 81, 'Expedition Max': 82,
                 'Explorer': 83, 'Express Cargo': 84, 'F-Pace': 85,
                 'F150': 86, 'F150 Lightning': 87, 'Forester': 88,
                 'Forte': 89, 'Forte5': 90, 'Frontier':91,
                 'G70': 92, 'G80': 93, 'GLA': 94,
                 'GLB': 95, 'GLB-Class': 96, 'GLC':97,
                 'GLC-Class': 98, 'GLE-Class': 99,
                 'GR Corolla': 100, 'GR Supra': 101,
                 'GR86': 102, 'GTI': 103, 'GV70': 104,
                 'GV80': 105, 'GX-Series': 106, 'Giulia': 107,
                 'Gladiator': 108, 'Golf R': 109, 'Grand Caravan': 110,
                 'Grand Cherokee': 111, 'Grand Cherokee L': 112,
                 'Grand Highlander': 113, 'Grecale': 114, 'HR-V': 115,
                 'Highlander': 116, 'Hornet': 117, 'ID.4': 118,
                 'IS-Series': 119, 'Impreza': 120, 'Integra': 121,
                 'Jetta': 122, 'Kicks': 123, 'Kona': 124,
                 'Kona EV': 125, 'Land Cruiser': 126, 'Leaf': 127,
                 'Legacy': 128, 'Lyriq': 129, 'M2': 130,
                 'MDX': 131, 'MX-5': 132, 'MX5': 133,
                 'Malibu': 134, 'Maverick': 135, 'Maxima': 136,
                 'Mazda3': 137, 'Mirage': 138, 'Murano': 139,
                 'Mustang': 140, 'Mustang MACH-E': 141, 'NX-Series': 142,
                 'Nautilus': 143, 'Niro': 144, 'Odyssey': 145,
                 'Outback': 146, 'Outlander': 147, 'Pacifica': 148,
                 'Palisade': 149, 'Passport': 150, 'Pathfinder': 151,
                 'Pilot': 152, 'Prius': 153, 'Prius Prime': 154,
                 'ProMaster': 155, 'Q3': 156, 'Q4 e-tron': 157,
                 'Q5': 158, 'Q50': 159, 'Q7': 160, 'Q8': 161,
                 'Q8 e-tron': 162, 'QX50': 163, 'QX55': 164,
                 'QX60': 165, 'QX80': 166, 'Qashqai': 167,
                 'RDX': 168, 'RVR': 169, 'RX-Series': 170,
                 'RX350': 171, 'RX350h': 172, 'Range Rover Evoque': 173,
                 'Range Rover Velar': 174, 'Ranger': 175,
                 'Rav4': 176, 'Rav4 Prime': 177, 'Renegade': 178,
                 'Ridgeline': 179, 'Rio': 180, 'Rogue': 181,
                 'S3': 182, 'S4': 183, 'S5': 184,
                 'S60': 185, 'SQ5': 186, 'Santa Cruz': 187,
                 'Santa Fe': 188, 'Savana': 189, 'Savana Cargo': 190,
                 'Seltos': 191, 'Sentra': 192, 'Sequoia': 193,
                 'Sienna': 194, 'Sierra 1500': 195,
                 'Silverado 1500': 196, 'Silverado EV': 197,
                 'Sonata': 198, 'Sorento': 199, 'Soul': 200,
                 'Sportage': 201, 'Sprinter 2500': 202,
                 'Sprinter 2500 Cargo': 203, 'Stelvio': 204,
                 'Suburban': 205, 'TLX': 206, 'TX-Series': 207,
                 'Tacoma': 208, 'Tahoe': 209, 'Taos': 210,
                 'Telluride': 211, 'Terrain': 212, 'Tiguan': 213,
                 'Tonale': 214, 'Trailblazer': 215, 'Transit 150': 216,
                 'Transit 250': 217, 'Transit 350': 218, 'Transit 350HD': 219,
                 'Traverse': 220, 'Trax': 221, 'Tucson': 222,
                 'Tundra': 223, 'UX-Series': 224, 'V60': 225, 'V90': 226,
                 'Venue': 227, 'Venza': 228, 'Versa': 229, 'WRX': 230,
                 'Wagoneer': 231, 'Wagoneer L': 232, 'Wrangler': 233,
                 'X1': 234, 'X2': 235, 'X3': 236, 'X4': 237,
                 'X5': 238, 'XC40': 239, 'XC60': 240, 'XC90': 241,
                 'XT4': 242, 'XT5': 243, 'XT6': 244, 'Yukon': 245,
                 'Yukon XL': 246, 'Z': 247, 'ZDX': 248

                 }

style_mapping = {
    '2D Cabrio Qtro': 0, '2D Cabriolet': 1, '2D Cabriolet 4MATIC': 2, '2D Conv 6sp': 3, '2D Conv at': 4,
    '2D Convertible': 5, '2D Convertible at': 6, '2D Coupe': 7, '2D Coupe 4MATIC': 8, '2D Coupe 6sp': 9,
    '2D Coupe AWD': 10, '2D Coupe Qtro': 11, '2D Coupe at': 12, '2D Utility 4WD': 13, '3D Hatchback': 14,
    '4D 4WD': 15, '4D Coupe': 16, '4D Coupe 4MATIC': 17, '4D Coupe 4MATIC +': 18, '4D Coupe 4Matic': 19,
    '4D Hatchback': 20, '4D Hatchback AWD': 21, '4D Sedan': 22, '4D Sedan 2.0T AWD': 23, '4D Sedan 2.5': 24,
    '4D Sedan 3.0TT AWD': 25, '4D Sedan 4MATIC': 26, '4D Sedan 4Matic': 27, '4D Sedan 6MT': 28, '4D Sedan 6sp': 29,
    '4D Sedan AWD': 30, '4D Sedan AWD at': 31, '4D Sedan FWD': 32, '4D Sedan Qtro': 33, '4D Sedan RWD': 34,
    '4D Sedan V6 AWD': 35, '4D Sedan V8': 36, '4D Sedan at': 37, '4D Sportback Qtro': 38, '4D Utility': 39,
    '4D Utility 2.0': 40, '4D Utility 3.6': 41, '4D Utility 4MATIC': 42, '4D Utility 4Motion': 43,
    '4D Utility 4WD': 44, '4D Utility 5 Pass': 45, '4D Utility 7 Pass': 46, '4D Utility 7P': 47,
    '4D Utility 8P': 48, '4D Utility AWD': 49, '4D Utility AWD 2.5T': 50, '4D Utility AWD 6P': 51,
    '4D Utility AWD 7P': 52, '4D Utility AWD 8P': 53, '4D Utility FWD': 54, '4D Utility Hyb AWD': 55,
    '4D Utility PHEV AWD': 56, '4D Utility Qtro': 57, '4D Utility RWD': 58, '4D Utility at': 59,
    '4D Utility at AWD 6P': 60, '4D Utility e-4orce': 61, '4D Wagon': 62, '4D Wagon AWD': 63,
    '4D Wagon Qtro': 64, '5D Hatchback': 65, '5D Hatchback 6MT': 66, '5D Hatchback 6sp': 67,
    '5D Hatchback AWD': 68, '5D Hatchback at': 69, '5D Hbk 6sp': 70, '5D Hbk AWD at': 71,
    '5D Hbk at': 72, '5D Wagon': 73, '5D Wagon 7 Pass AWD': 74, '5D Wagon 8 Pass AWD': 75,
    '5D Wagon 8 Pass FWD': 76, 'CC AWD 2.0L': 77, 'CC Hybrid FWD 2.5L': 78, 'Cargo HR DRW AWD 11K': 79,
    'Cargo HR Sld 148 AWD': 80, 'Cargo HR Slide 148WB': 81, 'Cargo HR Slide 148eL': 82,
    'Cargo LR Sld 148 AWD': 83, 'Cargo LR Slide 130WB': 84, 'Cargo LR Slide 148WB': 85,
    'Cargo MR Sld 130 AWD': 86, 'Cargo MR Slide 130WB': 87, 'Cargo MR Slide 148WB': 88,
    'Cargo Van 135WB': 89, 'Cargo Van 155WB': 90, 'Cargo Van 159 WB': 91, 'Cargo Van I4 HO LWB': 92,
    'Cargo Van I4 HO SWB': 93, 'Cargo Van I4 LWB': 94, 'Cargo Van I4 SWB': 95, 'Cargo Van LWB 4WD Ex': 96,
    'Crew Cab 4WD': 97, 'Crew Cab 4X4 at': 98, 'Crew Cab AWD': 99, 'Crew Cab AWD 2.0L': 100,
    'Crew Cab FWD 2.5L': 101, 'Crew Cab LWB 4WD': 102, 'Crew Cab SWB 2WD': 103,
    'Crew Cab SWB 4WD': 104, 'Crew Cab e4WD': 105, 'Crew I4 HO LWB': 106, 'Crew I4 LWB': 107,
    'Crew LR Sld 148 AWD': 108, 'Crew Max 4WD': 109, 'Crew Max LB 4WD': 110,
    'Crg HR Sld 148 AWD': 111, 'Crg HR Sld 148EL': 112, 'Crg HR Sld 148EL AWD': 113,
    'Crg HR Sld 148WB': 114, 'Crg LR Sld 130 AWD': 115, 'Crg LR Sld 130WB': 116,
    'Crg LR Sld 148 AWD': 117, 'Crg LR Sld 148WB': 118, 'Crg MR Sld 148 AWD': 119,
    'Crg MR Sld 148WB': 120, 'Crg Van 118 WB': 121, 'Crg Van 136 WB': 122,
    'Crg Van 159 WB': 123, 'Crg Van Ext 159 WB': 124, 'Dbl Cab 4WD': 125, 'Dbl Cab LB 4WD': 126,
    'Dbl Cab LWB 4X4 at': 127, 'Dbl Cab SWB 4WD': 128, 'Dbl Cab SWB 4X4 at': 129,
    'Dbl Cab SWB 4X4 mt': 130, 'Double Cab SWB 4WD': 131, 'King Cab 4X4 at': 132,
    'Pass 148EL DRW AWD': 133, 'Pass LR Sld 148 AWD': 134, 'Pass MR Sld 148 AWD': 135,
    'Pass MR Sld 148WB': 136, 'Quad Cab SWB 2WD': 137, 'Quad Cab SWB 4WD': 138,
    'Reg Cab LWB 2WD': 139, 'Reg Cab LWB 4WD': 140, 'Reg Cab SWB 2WD': 141,
    'Reg Cab SWB 4WD': 142, 'Supercab LWB 4WD': 143, 'Supercab SWB 4WD': 144,
    'Supercrew 4WD': 145, 'Supercrew LWB 4WD': 146, 'Supercrew SWB 4WD': 147,
    'Wagon 155WB': 148
}


color_mapping = {
    'beige': 0, 'beige,black': 1, 'beige,brown': 2, 'beige,red': 3,
    'black': 4, 'blue': 5, 'blue,beige': 6, 'blue,black': 7, 'blue,gray': 8,
    'blue,silver': 9, 'blue,silver,black': 10, 'brown': 11, 'brown,black': 12,
    'brown,red': 13, 'gold': 14, 'gold,beige': 15, 'gold,beige,black': 16,
    'gold,beige,brown': 17, 'gold,brown': 18, 'gray': 19, 'gray,beige': 20,
    'gray,beige,black': 21, 'gray,black': 22, 'gray,brown': 23,
    'gray,brown,silver': 24, 'gray,green': 25, 'gray,red': 26,
    'gray,silver': 27, 'gray,silver,black': 28, 'gray,white': 29,
    'gray,white,black': 30, 'gray,white,red': 31, 'gray,white,red,black': 32,
    'green': 33, 'green,black': 34, 'green,brown': 35, 'green,silver': 36,
    'green,white': 37, 'orange': 38, 'orange,black': 39, 'orange,blue': 40,
    'orange,brown': 41, 'purple': 42, 'red': 43, 'red,black': 44,
    'red,silver': 45, 'silver': 46, 'silver,black': 47, 'white': 48,
    'white,black': 49, 'white,red': 50, 'white,silver': 51, 'yellow': 52,
    'yellow,beige,silver': 53, 'yellow,gray': 54, 'yellow,gray,beige,silver': 55
}


interior_color_mapping = {
    'black': 0,
    'brown': 1,
    'brown,black': 2,
    'brown,red,black': 3,
    'gray': 4,
    'gray,black': 5,
    'gray,brown,black': 6,
    'gray,red': 7,
    'gray,white': 8,
    'gray,white,black': 9,
    'gray,white,brown': 10,
    'gray,white,brown,black': 11,
    'other': 12,
    'other,gray': 13,
    'other,red': 14,
    'other,red,black': 15,
    'red': 16,
    'red,black': 17,
    'white': 18,
    'white,black': 19,
    'white,brown': 20
}


drivetrain_mapping = {
    '4WD': 0, 'AWD': 1, 'FWD': 2, 'RWD': 3
}

transmission_mapping = {
    'Automatic': 0, 'Manual': 1
}

fuel_type_mapping = {
    'Diesel': 0, 'Electric': 1, 'Gas': 2, 'Hybrid': 3, 'PHEV': 4
}

# Select make
selected_make = st.selectbox("Make", list(make_to_models.keys()))
make = make_mapping[selected_make]

# Get only the models for the selected make
filtered_model_mapping = make_to_models[selected_make]
selected_model_name = st.selectbox("Model", list(filtered_model_mapping.keys()))
model_code = filtered_model_mapping[selected_model_name]

# ---------------------------------------------------------

# Input fields
days_on_market = st.number_input("Days on Market", value=30)
msrp = st.number_input("MSRP ($)", value=25000)
model_year = st.number_input("Model Year", value=2020)

# Dropdowns with readable labels (internally use encoded values)


# Dynamic style list from the nested dictionary
style_options = make_model_to_styles.get(selected_make, {}).get(selected_model_name, [])

if style_options:
    selected_style_name = st.selectbox("Style", style_options)
    style = style_mapping[selected_style_name]
else:
    st.warning("No styles found for selected make/model.")
    selected_style_name = None  # Or set a default/fallback
    style = None

exterior_color = color_mapping[st.selectbox("Exterior Color", list(color_mapping.keys()))]
interior_color = interior_color_mapping[st.selectbox("Interior Color", list(interior_color_mapping.keys()))]
drivetrain = drivetrain_mapping[st.selectbox("Drivetrain", list(drivetrain_mapping.keys()))]
transmission = transmission_mapping[st.selectbox("Transmission", list(transmission_mapping.keys()))]
fuel_type = fuel_type_mapping[st.selectbox("Fuel Type", list(fuel_type_mapping.keys()))]

number_price_changes = st.number_input("Number of Price Changes", value=1)
engine_size = st.number_input("Engine Size (L)", min_value=0.0, max_value=6.0, value=2.0)


# Interaction term
mileage_year_interaction = mileage * model_year

# Feature order (must match model training)
columns = [
    'days_on_market',
    'mileage',
    'msrp',
    'model_year',
    'make',
    'model',
    'style',
    'exterior_color_category',
    'interior_color_category',
    'drivetrain_from_vin',
    'transmission_from_vin',
    'fuel_type_from_vin',
    'number_price_changes',
    'engine_size',
    'mileage_year_interaction'
]

# Predict button
if st.button("Predict Price"):
    if style is None:
        st.error("Please select a valid style before predicting.")
    else:
        input_df = pd.DataFrame([[
            days_on_market,
            mileage,
            msrp,
            model_year,
            make,
            model_code,
            style,
            exterior_color,
            interior_color,
            drivetrain,
            transmission,
            fuel_type,
            number_price_changes,
            engine_size,
            mileage_year_interaction
        ]], columns=columns)

        log_pred = model.predict(input_df)[0]
        st.write("Input going to model:", input_df)
        prediction = np.expm1(log_pred)
        st.success(f"💰 Predicted Price: ${prediction:,.2f}")