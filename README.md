# My AI Doctor

This documentation explains the backend, frontend, and application installation process for the AI Health Care Center web app, which predicts diseases based on user-inputted symptoms and provides recommendations.

## 1. Backend

### Framework and Tools
- **Flask**: The backend framework used for routing and application logic.
- **joblib**: For loading the pre-trained machine learning model.
- **pandas**: For handling datasets like symptoms, descriptions, medications, etc.
- **numpy**: For numerical computation, particularly to build input vectors for the model.
- **Bootstrap**: Integrated with Flask templates for responsive and styled web pages.

### Key Features
- **Symptom Autocomplete**: Backend provides a list of unique symptoms extracted from datasets to support the autocomplete functionality.
- **Disease Prediction**: A Random Forest model predicts the disease based on user symptoms.
- **Recommendations**:
  - **Medications**
  - **Workout suggestions**
  - **Disease descriptions**

### File Structure
- `app.py`: The main Flask application with routes for rendering the interface and handling predictions.
- `datasets/`: Contains CSV files used to extract symptoms, precautions, medications, etc.
  - `symptoms_df.csv`
  - `precautions_df.csv`
  - `workout_df.csv`
  - `description.csv`
  - `medications.csv`
  - `diets.csv`
- `models/`: Contains the pre-trained model file (`random_forest_model.joblib`).

### Backend Workflow
1. **Dataset Loading**: CSV files are loaded during app initialization.
2. **Symptom Mapping**: Symptoms are mapped to input indices using a dictionary for vector construction.
3. **Prediction Logic**: The input vector is passed to the Random Forest model for disease prediction.
4. **Data Retrieval**: Based on the predicted disease, the app fetches the associated:
   - Description
   - Medications
   - Recommendations (workout, precautions, diets)

### Routes
- `/`: Displays the homepage with the symptom selection form.
- `/predict`: Accepts symptoms from the form, predicts the disease, and displays the result along with recommendations.


## 2. Frontend

### Framework and Tools
- **HTML5, CSS, and JavaScript**: For structure, styling, and dynamic behavior.
- **Bootstrap**: Ensures responsive design.
- **Jinja2 Templating**: Used in Flask to dynamically inject data into HTML templates.

### Key Features
- **Autocomplete Dropdown**: A dynamic dropdown helps users select symptoms.
- **Dynamic Form**: Selected symptoms are displayed in a list and sent as a hidden input to the backend.
- **Results Display**:
  - Predicted disease
  - Description
  - Medication recommendations
  - Suggested workouts

### User Interaction Workflow
1. The user starts typing symptoms in the text box.
2. An autocomplete dropdown suggests matching symptoms.
3. The user selects symptoms, which are added to a list.
4. On form submission, the app displays the predicted disease and recommendations.

### Frontend Structure
- **Homepage (`index.html`)**:
  - Symptom input field with autocomplete.
  - Diagnosis results section that dynamically updates based on predictions.
- **Styling**:
  - Modern gradient backgrounds and shadowed containers for aesthetics.
  - Responsive layout for different devices.

---

## 3. Installation

### Prerequisites
1. **Python 3.8+** installed on your system.
2. **pip**: Ensure Python's package manager is available.
3. Install the required Python libraries:
   ```bash
   pip install flask pandas numpy joblib
   ```

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ai-health-care-center.git
   cd ai-health-care-center
   ```

2. Create necessary directories and place datasets and models:
   - `datasets/`:
     - Place `symptoms_df.csv`, `precautions_df.csv`, `workout_df.csv`, `description.csv`, `medications.csv`, `diets.csv`.
   - `models/`:
     - Place the `random_forest_model.joblib` file.

3. Run the Flask app:
   ```bash
   python app.py
   ```

4. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

### Deploying on a Server
1. Install a web server like **Gunicorn**:
   ```bash
   pip install gunicorn
   ```
2. Run the app with Gunicorn:
   ```bash
   gunicorn -w 4 -b 0.0.0.0:8000 main:app
   ```

### Notes
- Ensure that all CSV files and the model are correctly placed in their respective directories.
- Modify file paths in `main.py` if deploying on a different environment.

For questions or feedback, contact noecaremee@gmail.com.
