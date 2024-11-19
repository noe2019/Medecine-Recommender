from flask import Flask, request, render_template
from joblib import load
import os
import ast 
import numpy as np
import pandas as pd

# Flask app
app = Flask(__name__)

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load datasets
def load_data(file_name):
    """Helper function to load a CSV file."""
    file_path = os.path.join(DATASETS_DIR, file_name)
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset '{file_name}' not found in {DATASETS_DIR}. Please ensure the file exists.")
    except Exception as e:
        raise RuntimeError(f"Error loading dataset '{file_name}': {e}")

sym_des = load_data("symtoms_df.csv")
precautions = load_data("precautions_df.csv")
workout = load_data("workout_df.csv")
description = load_data("description.csv")
medications = load_data("medications.csv")
diets = load_data("diets.csv")

# Extract unique symptoms for the autocomplete feature
columns_to_check = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']
unique_symptoms = (
    sym_des[columns_to_check]
    .values.ravel('K')  # Flatten DataFrame
    .tolist()
)
unique_symptoms = sorted(set(s.strip() for s in unique_symptoms if isinstance(s, str) and s.strip()))

# Load model
try:
    model_path = os.path.join(MODELS_DIR, 'random_forest_model.joblib')
    svc = load(model_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file not found. Please ensure '{model_path}' exists.")
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

# Symptom dictionary
symptoms_dict = {
    'itching': 0,
    'skin_rash': 1,
    'nodal_skin_eruptions': 2,
    'continuous_sneezing': 3,
    'shivering': 4,
    'chills': 5,
    'joint_pain': 6,
    'stomach_pain': 7,
    'acidity': 8,
    'ulcers_on_tongue': 9,
    'muscle_wasting': 10,
    'vomiting': 11,
    'burning_micturition': 12,
    'spotting_urination': 13,
    'fatigue': 14,
    'weight_gain': 15,
    'anxiety': 16,
    'cold_hands_and_feets': 17,
    'mood_swings': 18,
    'weight_loss': 19,
    'restlessness': 20,
    'lethargy': 21,
    'patches_in_throat': 22,
    'irregular_sugar_level': 23,
    'cough': 24,
    'high_fever': 25,
    'sunken_eyes': 26,
    'breathlessness': 27,
    'sweating': 28,
    'dehydration': 29,
    'indigestion': 30,
    'headache': 31,
    'yellowish_skin': 32,
    'dark_urine': 33,
    'nausea': 34,
    'loss_of_appetite': 35,
    'pain_behind_the_eyes': 36,
    'back_pain': 37,
    'constipation': 38,
    'abdominal_pain': 39,
    'diarrhoea': 40,
    'mild_fever': 41,
    'yellow_urine': 42,
    'yellowing_of_eyes': 43,
    'acute_liver_failure': 44,
    'fluid_overload': 45,
    'swelling_of_stomach': 46,
    'swelled_lymph_nodes': 47,
    'malaise': 48,
    'blurred_and_distorted_vision': 49,
    'phlegm': 50,
    'throat_irritation': 51,
    'redness_of_eyes': 52,
    'sinus_pressure': 53,
    'runny_nose': 54,
    'congestion': 55,
    'chest_pain': 56,
    'weakness_in_limbs': 57,
    'fast_heart_rate': 58,
    'pain_during_bowel_movements': 59,
    'pain_in_anal_region': 60,
    'bloody_stool': 61,
    'irritation_in_anus': 62,
    'neck_pain': 63,
    'dizziness': 64,
    'cramps': 65,
    'bruising': 66,
    'obesity': 67,
    'swollen_legs': 68,
    'swollen_blood_vessels': 69,
    'puffy_face_and_eyes': 70,
    'enlarged_thyroid': 71,
    'brittle_nails': 72,
    'swollen_extremities': 73,
    'excessive_hunger': 74,
    'extra_marital_contacts': 75,
    'drying_and_tingling_lips': 76,
    'slurred_speech': 77,
    'knee_pain': 78,
    'hip_joint_pain': 79,
    'muscle_weakness': 80,
    'stiff_neck': 81,
    'swelling_joints': 82,
    'movement_stiffness': 83,
    'spinning_movements': 84,
    'loss_of_balance': 85,
    'unsteadiness': 86,
    'weakness_of_one_body_side': 87,
    'loss_of_smell': 88,
    'bladder_discomfort': 89,
    'foul_smell_of_urine': 90,
    'continuous_feel_of_urine': 91,
    'passage_of_gases': 92,
    'internal_itching': 93,
    'toxic_look_typhos': 94,
    'depression': 95,
    'irritability': 96,
    'muscle_pain': 97,
    'altered_sensorium': 98,
    'red_spots_over_body': 99,
    'belly_pain': 100,
    'abnormal_menstruation': 101,
    'dischromic_patches': 102,
    'watering_from_eyes': 103,
    'increased_appetite': 104,
    'polyuria': 105,
    'family_history': 106,
    'mucoid_sputum': 107,
    'rusty_sputum': 108,
    'lack_of_concentration': 109,
    'visual_disturbances': 110,
    'receiving_blood_transfusion': 111,
    'receiving_unsterile_injections': 112,
    'coma': 113,
    'stomach_bleeding': 114,
    'distention_of_abdomen': 115,
    'history_of_alcohol_consumption': 116,
    'fluid_overload.1': 117,
    'blood_in_sputum': 118,
    'prominent_veins_on_calf': 119,
    'palpitations': 120,
    'painful_walking': 121,
    'pus_filled_pimples': 122,
    'blackheads': 123,
    'scurring': 124,
    'skin_peeling': 125,
    'silver_like_dusting': 126,
    'small_dents_in_nails': 127,
    'inflammatory_nails': 128,
    'blister': 129,
    'red_sore_around_nose': 130,
    'yellow_crust_ooze': 131
}

diseases_list = {
    0: '(vertigo) Paroymsal Positional Vertigo',
    1: 'AIDS',
    2: 'Acne',
    3: 'Alcoholic hepatitis',
    4: 'Allergy',
    5: 'Arthritis',
    6: 'Bronchial Asthma',
    7: 'Cervical spondylosis',
    8: 'Chicken pox',
    9: 'Chronic cholestasis',
    10: 'Common Cold',
    11: 'Dengue',
    12: 'Diabetes ',
    13: 'Dimorphic hemmorhoids(piles)',
    14: 'Drug Reaction',
    15: 'Fungal infection',
    16: 'GERD',
    17: 'Gastroenteritis',
    18: 'Heart attack',
    19: 'Hepatitis B',
    20: 'Hepatitis C',
    21: 'Hepatitis D',
    22: 'Hepatitis E',
    23: 'Hypertension ',
    24: 'Hyperthyroidism',
    25: 'Hypoglycemia',
    26: 'Hypothyroidism',
    27: 'Impetigo',
    28: 'Jaundice',
    29: 'Malaria',
    30: 'Migraine',
    31: 'Osteoarthritis',
    32: 'Paralysis (brain hemorrhage)',
    33: 'Peptic ulcer disease',
    34: 'Pneumonia',
    35: 'Psoriasis',
    36: 'Tuberculosis',
    37: 'Typhoid',
    38: 'Urinary tract infection',
    39: 'Varicose veins',
    40: 'hepatitis A'
}

# Helper function
def helper(dis, description, precautions, medications, diets, workout):
    """Retrieve additional details for a disease."""
    try:
        desc = description.loc[description['Disease'] == dis, 'Description']
        desc = desc.values[0] if not desc.empty else "Description not available"

        pre = precautions.loc[precautions['Disease'] == dis, ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
        pre = pre.values.flatten().tolist() if not pre.empty else ["No precautions available"]

        med = medications.loc[medications['Disease'] == dis, 'Medication']
        med = med.tolist() if not med.empty else ["No medications available"]
        if isinstance(med, str):
            med = ast.literal_eval(med)
        else:
            med = med

        diet = diets.loc[diets['Disease'] == dis, 'Diet']
        diet = diet.tolist() if not diet.empty else ["No diet recommendations"]

        wrkout = workout.loc[workout['disease'] == dis, 'workout']
        wrkout = wrkout.tolist() if not wrkout.empty else ["No workout suggestions"]

        return desc, pre, med, diet, wrkout
    except Exception as e:
        print(f"Error retrieving details for disease '{dis}': {e}")
        return "Error retrieving details", [], [], [], []

# Prediction function
def get_predicted_value(patient_symptoms, symptoms_dict, diseases_list, model):
    """Predict the disease based on symptoms."""
    try:
        input_vector = np.zeros(len(symptoms_dict))

        # Map symptoms to input vector
        for symptom in patient_symptoms:
            clean_symptom = symptom.strip().lower().replace(" ", "_")
            if clean_symptom in symptoms_dict:
                symptom_index = symptoms_dict[clean_symptom]
                input_vector[symptom_index] = 1
            else:
                print(f"Warning: Symptom '{symptom}' not recognized.")

        # Predict the disease
        predicted_index = model.predict(input_vector.reshape(1, -1))[0]
        return diseases_list.get(predicted_index, "Unknown Disease")
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Prediction Failed"

# Routes
@app.route("/")
def index():
    return render_template("index.html", unique_symptoms=unique_symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form.get('symptoms', '').split(',')
    symptoms = [s for s in symptoms if s.strip()]

    if not symptoms:
        return render_template('index.html', message="No symptoms provided. Please select symptoms.", unique_symptoms=unique_symptoms)

    predicted_disease = get_predicted_value(symptoms, symptoms_dict, diseases_list, svc)
    if predicted_disease == "Prediction Failed":
        return render_template('index.html', message="Prediction failed. Please try again.", unique_symptoms=unique_symptoms)

    # Pass global `workout` explicitly
    dis_des, pre, med, diet, wrkout = helper(predicted_disease, description, precautions, medications, diets, workout)

    # Sanitize medications data
    if isinstance(med, str):
        med = ast.literal_eval(med)
    med = [m.strip() for m in med if isinstance(m, str)]

    return render_template(
        'index.html',
        predicted_disease=predicted_disease,
        dis_des=dis_des,
        #precautions=pre,
        medications=med,
        #diet=diet,
        workout=wrkout,
        unique_symptoms=unique_symptoms
    )

if __name__ == "__main__":
    # Bind to the PORT environment variable if defined, otherwise default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    app.run(debug=True)