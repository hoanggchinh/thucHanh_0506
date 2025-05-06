import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from scipy.stats import chi2_contingency

# Đọc dữ liệu từ file CSV (thay đường dẫn phù hợp)
file_path = r'C:\Users\hoang\PycharmProjects\thucHanh_0506\data\Data_Number_7.csv'
data = pd.read_csv(file_path)

# --- 1. Tạo chỉ số "nguy cơ biến chứng" ---
# Ví dụ trọng số: BMI (0.3), đường huyết (0.5), số lần nhập viện (0.2)
data['risk_index'] = 0.3*data['bmi'] + 0.5*data['blood_glucose'] + 0.2*data['hospitalizations']

# --- 2. Kiểm định chi-squared biến chứng theo nhóm tuổi ---
def age_group(x):
    if x < 40:
        return '<40'
    elif 40 <= x <= 60:
        return '40-60'
    else:
        return '>60'

data['age_group'] = data['age'].apply(age_group)
contingency = pd.crosstab(data['age_group'], data['complication'])
chi2, p, _, _ = chi2_contingency(contingency)
print("Bảng chéo biến chứng theo nhóm tuổi:\n", contingency)
print(f"Kiểm định chi-squared: chi2={chi2:.3f}, p-value={p:.4f}")

# --- 3. Tạo đặc trưng "xu hướng đường huyết" ---
# Giả lập 3 lần đo đường huyết trước (ví dụ thêm nhiễu ngẫu nhiên)
for i in range(1,4):
    data[f'blood_glucose_prev_{i}'] = data['blood_glucose'] + np.random.normal(0, 10, len(data))

def glucose_trend(row):
    past_mean = np.mean([row['blood_glucose_prev_1'], row['blood_glucose_prev_2'], row['blood_glucose_prev_3']])
    diff = row['blood_glucose'] - past_mean
    if diff > 5:
        return 'increasing'
    elif diff < -5:
        return 'decreasing'
    else:
        return 'stable'

data['glucose_trend'] = data.apply(glucose_trend, axis=1)
le = LabelEncoder()
data['glucose_trend_enc'] = le.fit_transform(data['glucose_trend'])

# --- 4. Tạo đặc trưng "mức độ nghiêm trọng" ---
def severity(row):
    if row['hospitalizations'] > 3 and row['blood_glucose'] > 180:
        return 'high'
    elif (1 <= row['hospitalizations'] <= 3) or (120 < row['blood_glucose'] <= 180):
        return 'medium'
    else:
        return 'low'

data['severity'] = data.apply(severity, axis=1)
data['severity_enc'] = le.fit_transform(data['severity'])

# --- 5. Chuẩn bị dữ liệu ---
features = ['age', 'bmi', 'blood_glucose', 'hospitalizations', 'risk_index', 'glucose_trend_enc', 'severity_enc']
X = data[features]
y = data['complication']

# --- 6. Xử lý mất cân bằng với SMOTE ---
print("Tỷ lệ biến chứng trước SMOTE:\n", y.value_counts(normalize=True))
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print("Tỷ lệ biến chứng sau SMOTE:\n", pd.Series(y_res).value_counts(normalize=True))

# --- 7. Chia train/test ---
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# --- 8. Logistic Regression ---
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("\nLogistic Regression:\n", classification_report(y_test, y_pred_lr))
print("AUC:", roc_auc_score(y_test, lr.predict_proba(X_test)[:,1]))

# --- 9. Random Forest và tuning ---
rf = RandomForestClassifier(random_state=42)
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 20, None]}
grid = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)
print("Best RF params:", grid.best_params_)
best_rf = grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)
print("\nRandom Forest:\n", classification_report(y_test, y_pred_rf))
print("AUC:", roc_auc_score(y_test, best_rf.predict_proba(X_test)[:,1]))
